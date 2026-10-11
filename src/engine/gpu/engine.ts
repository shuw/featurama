// GPT-2 on WebGPU with every activation kept in a buffer we own, so any of
// them can be read back or overwritten mid-run.

import type { ModelConfig, Weights } from "../model.ts";
import type { Intervention, Lens, RunRequest, Trace } from "../hooks.ts";
import { EDIT, EMBED, LAYERNORM, LINEAR, MIX, SCORES, TOPK, UNEMBED } from "./kernels.ts";

const S = GPUBufferUsage.STORAGE;
const LENS_CHUNK = 64; // rows of vocabulary logits held at once (64 x 50,257 floats)
export const MAX_TOKENS = 256;

type Kernel = "embed" | "layernorm" | "linear" | "scores" | "mix" | "edit" | "unembed" | "topk";
const SOURCES: Record<Kernel, string> = {
  embed: EMBED, layernorm: LAYERNORM, linear: LINEAR, scores: SCORES, mix: MIX, edit: EDIT, unembed: UNEMBED, topk: TOPK,
};

interface Activations {
  cap: number;
  tokens: GPUBuffer;
  identity: GPUBuffer;
  resid: GPUBuffer;
  residMid: GPUBuffer;
  ln: GPUBuffer;
  qkv: GPUBuffer;
  z: GPUBuffer;
  mlp: GPUBuffer;
  pattern: GPUBuffer;
  lensRows: GPUBuffer;
  lensLn: GPUBuffer;
  logits: GPUBuffer;
}

export class GpuEngine {
  private acts?: Activations;
  private dummy: GPUBuffer;
  private headScale: GPUBuffer;

  private constructor(
    readonly device: GPUDevice,
    readonly config: ModelConfig,
    private weights: Map<string, GPUBuffer>,
    private pipelines: Record<Kernel, GPUComputePipeline>,
    readonly adapterInfo: string,
  ) {
    this.dummy = device.createBuffer({ size: 16, usage: S });
    this.headScale = device.createBuffer({ size: config.n_layers * config.n_heads * 4, usage: S | GPUBufferUsage.COPY_DST });
  }

  static async create(w: Weights): Promise<GpuEngine> {
    if (!("gpu" in navigator)) throw new Error("This browser has no WebGPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter.");
    const largest = Math.max(...w.names().map((n) => w.raw(n).byteLength));
    if (adapter.limits.maxStorageBufferBindingSize < largest) {
      throw new Error(`This GPU can bind ${adapter.limits.maxStorageBufferBindingSize} bytes; the embedding needs ${largest}.`);
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const weights = new Map<string, GPUBuffer>();
    for (const name of w.names()) {
      const raw = w.raw(name);
      const buf = device.createBuffer({ size: raw.byteLength, usage: S, mappedAtCreation: true });
      new Uint16Array(buf.getMappedRange()).set(raw);
      buf.unmap();
      weights.set(name, buf);
    }
    const entries = await Promise.all(
      (Object.keys(SOURCES) as Kernel[]).map(async (k) => {
        const module = device.createShaderModule({ code: SOURCES[k], label: k });
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" }, label: k });
        return [k, pipeline] as const;
      }),
    );
    const info = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" ") : "";
    return new GpuEngine(device, w.config, weights, Object.fromEntries(entries) as Record<Kernel, GPUComputePipeline>, info);
  }

  async run(req: RunRequest): Promise<Trace> {
    const start = performance.now();
    const { n_layers: L, n_heads: H, d_model: d, d_mlp, d_vocab: V, ln_eps } = this.config;
    const tokens = req.tokens;
    const T = tokens.length;
    if (T < 1 || T > MAX_TOKENS) throw new Error(`Prompts must be 1 to ${MAX_TOKENS} tokens.`);
    const a = this.activations(T);
    const dev = this.device;
    const temp: GPUBuffer[] = [];

    dev.queue.writeBuffer(a.tokens, 0, new Uint32Array(tokens));
    const scale = new Float32Array(L * H).fill(1);
    for (const iv of req.interventions ?? []) if (iv.kind === "head") scale[iv.layer * H + iv.head] = iv.scale;
    dev.queue.writeBuffer(this.headScale, 0, scale);

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (k: Kernel, params: number[], buffers: GPUBuffer[], groups: [number, number?]) => {
      const u = this.uniform(params, k === "layernorm" ? [2] : []);
      temp.push(u);
      const bind = dev.createBindGroup({
        layout: this.pipelines[k].getBindGroupLayout(0),
        entries: [u, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      pass.setPipeline(this.pipelines[k]);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups[0], groups[1] ?? 1);
    };
    const W = (name: string) => this.weights.get(name)!;
    const linear = (x: GPUBuffer, xOff: number, n: number, m: number, prefix: string, flags: number, res: GPUBuffer, resOff: number, out: GPUBuffer, outOff: number) =>
      run("linear", [T, n, m, flags, resOff, outOff, xOff], [x, W(`${prefix}.weight`), W(`${prefix}.bias`), res, out], [Math.ceil(m / 128), Math.ceil(T / 8)]);
    const edits = (hook: string, layer: number) => {
      for (const iv of req.interventions ?? []) {
        if (iv.kind !== "resid") continue;
        const atPre = iv.hook === "resid_pre" ? iv.layer : iv.hook === "resid_post" ? iv.layer + 1 : -1;
        const here = hook === "mid" ? iv.hook === "resid_mid" && iv.layer === layer : atPre === layer;
        if (!here) continue;
        const target = hook === "mid" ? a.residMid : a.resid;
        const value = this.expand(iv, T);
        const vb = dev.createBuffer({ size: value.byteLength, usage: S | GPUBufferUsage.COPY_DST });
        dev.queue.writeBuffer(vb, 0, value.buffer as ArrayBuffer, value.byteOffset, value.byteLength);
        temp.push(vb);
        const at = layer * T * d + (iv.position === "all" || value.length === T * d ? 0 : iv.position * d);
        run("edit", [at, value.length, iv.mode === "set" ? 0 : 1, 0], [vb, target], [Math.ceil(value.length / 256)]);
      }
    };

    run("embed", [T, d], [a.tokens, W("wte.weight"), W("wpe.weight"), a.resid], [Math.ceil((T * d) / 256)]);
    edits("pre", 0);
    for (let l = 0; l < L; l++) {
      const off = l * T * d;
      const patOff = l * H * T * T;
      run("layernorm", [T, d, ln_eps, off], [a.resid, a.identity, W(`h.${l}.ln_1.weight`), W(`h.${l}.ln_1.bias`), a.ln], [T]);
      linear(a.ln, 0, d, 3 * d, `h.${l}.attn.c_attn`, 0, this.dummy, 0, a.qkv, 0);
      run("scores", [T, d, d / H, patOff], [a.qkv, a.pattern], [T, H]);
      run("mix", [T, d, d / H, patOff, l * H, off], [a.qkv, a.pattern, this.headScale, a.z], [Math.ceil((T * d) / 256)]);
      linear(a.z, off, d, d, `h.${l}.attn.c_proj`, 2, a.resid, off, a.residMid, off);
      edits("mid", l);
      run("layernorm", [T, d, ln_eps, off], [a.residMid, a.identity, W(`h.${l}.ln_2.weight`), W(`h.${l}.ln_2.bias`), a.ln], [T]);
      linear(a.ln, 0, d, d_mlp, `h.${l}.mlp.c_fc`, 1, this.dummy, 0, a.mlp, 0);
      linear(a.mlp, 0, d_mlp, d, `h.${l}.mlp.c_proj`, 2, a.residMid, off, a.resid, off + T * d);
      edits("pre", l + 1);
    }

    let lensInfo: { positions: number[]; K: number; targets: number[]; rows: number } | undefined;
    let ids: GPUBuffer | undefined;
    let lps: GPUBuffer | undefined;
    if (req.lens) {
      const positions = req.lens.positions ?? [T - 1];
      const K = req.lens.topK ?? 5;
      const targets = req.lens.targets ?? [];
      const rows: number[] = [];
      for (let l = 0; l <= L; l++) for (const p of positions) rows.push(l * T + p);
      dev.queue.writeBuffer(a.lensRows, 0, new Uint32Array(rows));
      const tb = dev.createBuffer({ size: Math.max(16, targets.length * 4), usage: S | GPUBufferUsage.COPY_DST });
      if (targets.length) dev.queue.writeBuffer(tb, 0, new Uint32Array(targets));
      ids = dev.createBuffer({ size: rows.length * K * 4, usage: S | GPUBufferUsage.COPY_SRC });
      lps = dev.createBuffer({ size: rows.length * (K + targets.length) * 4, usage: S | GPUBufferUsage.COPY_SRC });
      temp.push(tb, ids, lps);
      run("layernorm", [rows.length, d, ln_eps, 0], [a.resid, a.lensRows, W("ln_f.weight"), W("ln_f.bias"), a.lensLn], [rows.length]);
      for (let r0 = 0; r0 < rows.length; r0 += LENS_CHUNK) {
        const R = Math.min(LENS_CHUNK, rows.length - r0);
        run("unembed", [R, d, V, r0 * d], [a.lensLn, W("wte.weight"), a.logits], [Math.ceil(V / 64), Math.ceil(R / 4)]);
        run("topk", [R, V, K, targets.length, r0], [a.logits, tb, ids, lps], [R]);
      }
      lensInfo = { positions, K, targets, rows: rows.length };
    }
    pass.end();

    const reads: [keyof Trace | "ids" | "lps", GPUBuffer, number][] = [];
    if (req.read?.resid) reads.push(["resid", a.resid, (L + 1) * T * d * 4]);
    if (req.read?.residMid) reads.push(["residMid", a.residMid, L * T * d * 4]);
    if (req.read?.pattern) reads.push(["pattern", a.pattern, L * H * T * T * 4]);
    if (req.read?.z) reads.push(["z", a.z, L * T * d * 4]);
    if (ids && lps) reads.push(["ids", ids, ids.size], ["lps", lps, lps.size]);
    const staged = reads.map(([key, src, size]) => {
      const dst = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(src, 0, dst, 0, size);
      temp.push(dst);
      return [key, dst] as const;
    });
    dev.queue.submit([enc.finish()]);
    await Promise.all([dev.queue.onSubmittedWorkDone(), ...staged.map(([, b]) => b.mapAsync(GPUMapMode.READ))]);

    const trace: Trace = { tokens, ms: 0 };
    const out: Record<string, ArrayBuffer> = {};
    for (const [key, b] of staged) out[key] = b.getMappedRange().slice(0);
    if (out.resid) trace.resid = new Float32Array(out.resid);
    if (out.residMid) trace.residMid = new Float32Array(out.residMid);
    if (out.pattern) trace.pattern = new Float32Array(out.pattern);
    if (out.z) trace.z = new Float32Array(out.z);
    if (lensInfo) {
      const { positions, K, targets, rows } = lensInfo;
      const all = new Float32Array(out.lps);
      const logprobs = new Float32Array(rows * K);
      const targetLogprobs = new Float32Array(rows * targets.length);
      for (let r = 0; r < rows; r++) {
        logprobs.set(all.subarray(r * (K + targets.length), r * (K + targets.length) + K), r * K);
        targetLogprobs.set(all.subarray(r * (K + targets.length) + K, (r + 1) * (K + targets.length)), r * targets.length);
      }
      trace.lens = { positions, topK: K, targets, ids: new Uint32Array(out.ids), logprobs, targetLogprobs } satisfies Lens;
    }
    for (const b of temp) b.destroy();
    trace.ms = performance.now() - start;
    return trace;
  }

  private expand(iv: Extract<Intervention, { kind: "resid" }>, T: number) {
    const d = this.config.d_model;
    if (iv.value.length === T * d) return iv.value;
    if (iv.value.length !== d) throw new Error(`An intervention needs ${d} or ${T * d} numbers, not ${iv.value.length}.`);
    if (iv.position !== "all") return iv.value;
    const out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) out.set(iv.value, t * d);
    return out;
  }

  // Uniform block of 4-byte slots; `floats` lists the slots that are f32.
  private uniform(values: number[], floats: number[]) {
    const data = new ArrayBuffer(Math.ceil((values.length * 4) / 16) * 16);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    values.forEach((v, i) => (floats.includes(i) ? (f[i] = v) : (u[i] = v)));
    const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  private activations(T: number): Activations {
    if (this.acts && this.acts.cap >= T) return this.acts;
    if (this.acts) for (const b of Object.values(this.acts)) if (b instanceof GPUBuffer) b.destroy();
    const cap = Math.min(MAX_TOKENS, Math.max(32, 2 ** Math.ceil(Math.log2(T))));
    const { n_layers: L, n_heads: H, d_model: d, d_mlp, d_vocab: V } = this.config;
    const f32 = (n: number, extra = 0) => this.device.createBuffer({ size: n * 4, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra });
    const identity = f32(cap);
    this.device.queue.writeBuffer(identity, 0, new Uint32Array(cap).map((_, i) => i));
    this.acts = {
      cap,
      tokens: f32(cap),
      identity,
      resid: f32((L + 1) * cap * d),
      residMid: f32(L * cap * d),
      ln: f32(cap * d),
      qkv: f32(cap * 3 * d),
      z: f32(L * cap * d),
      mlp: f32(cap * d_mlp),
      pattern: f32(L * H * cap * cap),
      lensRows: f32((L + 1) * cap),
      lensLn: f32((L + 1) * cap * d),
      logits: f32(LENS_CHUNK * V),
    };
    return this.acts;
  }
}
