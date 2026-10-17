// The arithmetic for looking inside parts: inside a head and an MLP, who voted
// for the answer, and tracing it back by unplugging parts. Everything is
// computed from a real run and the real weights.

import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention, Trace } from "../engine/hooks.ts";
import { BOS, type ModelConfig, type Weights } from "../engine/model.ts";
import { headOutput, mlpOutput } from "./heads.ts";

export interface HeadInside {
  q: Float32Array; // the query token's question, 64 numbers
  keys: Float32Array[]; // each position's label, up to the query
  values: Float32Array[]; // what each position offers
  scores: number[]; // q · k / 8
  pattern: number[]; // the scores squashed into shares that add up to 1
  z: Float32Array; // the blend of values, 64 numbers
}

export function insideHead(trace: Trace, cfg: ModelConfig, l: number, h: number, query: number): HeadInside {
  const { n_heads: H, d_model: d } = cfg;
  const dh = d / H;
  const T = trace.tokens.length;
  const at = (t: number, part: number) => (l * T + t) * 3 * d + part * d + h * dh;
  const slice = (o: number) => trace.qkv!.slice(o, o + dh);
  const q = slice(at(query, 0));
  const keys: Float32Array[] = [];
  const values: Float32Array[] = [];
  const scores: number[] = [];
  const pattern: number[] = [];
  for (let j = 0; j <= query; j++) {
    const k = slice(at(j, 1));
    keys.push(k);
    values.push(slice(at(j, 2)));
    let s = 0;
    for (let c = 0; c < dh; c++) s += q[c] * k[c];
    scores.push(s / Math.sqrt(dh));
    pattern.push(trace.pattern![((l * H + h) * T + query) * T + j]);
  }
  const z = trace.z!.slice((l * T + query) * d + h * dh, (l * T + query) * d + (h + 1) * dh);
  return { q, keys, values, scores, pattern, z };
}

// How much each earlier position's value adds to what the head writes: its
// share of attention times the size of its value after the output projection.
export function headSources(trace: Trace, w: Weights, l: number, h: number, query: number): number[] {
  const { n_heads: H, d_model: d } = w.config;
  const dh = d / H;
  const T = trace.tokens.length;
  const WO = w.f32(`h.${l}.attn.c_proj.weight`);
  const out: number[] = [];
  const acc = new Float64Array(d);
  for (let j = 0; j <= query; j++) {
    acc.fill(0);
    const vo = (l * T + j) * 3 * d + 2 * d + h * dh;
    for (let k = 0; k < dh; k++) {
      const v = trace.qkv![vo + k];
      const row = (h * dh + k) * d;
      for (let o = 0; o < d; o++) acc[o] += v * WO[row + o];
    }
    let s = 0;
    for (let o = 0; o < d; o++) s += acc[o] * acc[o];
    out.push(trace.pattern![((l * H + h) * T + query) * T + j] * Math.sqrt(s));
  }
  const sum = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((v) => v / sum);
}

// --- Who voted: the answer's score split across every part ----------------

export type PartId = string; // "9.9" for a head, "m7" for layer 7's MLP, "embed", "bias"

export interface Vote {
  id: PartId;
  value: number; // how much it adds to (target's score − rival's score)
}

let F16: Float32Array | null = null;
const f16 = () => (F16 ??= (() => {
  const t = new Float32Array(65536);
  for (let x = 0; x < 65536; x++) {
    const sign = x & 0x8000 ? -1 : 1;
    const e = (x >> 10) & 0x1f;
    const f = x & 0x3ff;
    t[x] = e === 0 ? sign * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : sign * Infinity) : sign * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})());

export function tokenVector(w: Weights, id: number): Float32Array {
  const d = w.config.d_model;
  const raw = w.raw("wte.weight").subarray(id * d, (id + 1) * d);
  const t = f16();
  return Float32Array.from(raw, (x) => t[x]);
}

// The direction in the final residual stream that raises target over rival,
// through the final LayerNorm at its real scale for this position. Every
// part's vote is its output dotted with this; the LayerNorm's own bias adds
// a constant. Votes plus that constant are exactly the score difference.
function voteDirection(trace: Trace, w: Weights, position: number, target: number, rival: number) {
  const { n_layers: L, d_model: d, ln_eps } = w.config;
  const T = trace.tokens.length;
  const x = trace.resid!.subarray((L * T + position) * d, (L * T + position + 1) * d);
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) v += (x[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(v / d + ln_eps);
  const gain = w.f32("ln_f.weight");
  const bias = w.f32("ln_f.bias");
  const a = tokenVector(w, target);
  const b = tokenVector(w, rival);
  const u = new Float32Array(d);
  let um = 0;
  let constant = 0;
  for (let i = 0; i < d; i++) {
    u[i] = scale * gain[i] * (a[i] - b[i]);
    um += u[i];
    constant += bias[i] * (a[i] - b[i]);
  }
  um /= d;
  for (let i = 0; i < d; i++) u[i] -= um;
  return { u, constant };
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>, n = a.length) => {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
};

export function votes(trace: Trace, w: Weights, position: number, target: number, rival: number) {
  const { n_layers: L, n_heads: H, d_model: d } = w.config;
  const dh = d / H;
  const T = trace.tokens.length;
  const { u, constant } = voteDirection(trace, w, position, target, rival);
  const parts: Vote[] = [{ id: "embed", value: dot(trace.resid!.subarray(position * d, (position + 1) * d), u) }];
  let bias = 0;
  for (let l = 0; l < L; l++) {
    const WO = w.f32(`h.${l}.attn.c_proj.weight`);
    bias += dot(w.f32(`h.${l}.attn.c_proj.bias`), u);
    for (let h = 0; h < H; h++) {
      let s = 0;
      for (let k = 0; k < dh; k++) {
        const row = (h * dh + k) * d;
        let g = 0;
        for (let o = 0; o < d; o++) g += WO[row + o] * u[o];
        s += trace.z![(l * T + position) * d + h * dh + k] * g;
      }
      parts.push({ id: `${l}.${h}`, value: s });
    }
    parts.push({ id: `m${l}`, value: dot(mlpOutput(trace, d, l, position), u) });
  }
  parts.push({ id: "bias", value: bias + constant });
  const total = parts.reduce((s, p) => s + p.value, 0);
  return { parts, total };
}

// --- Tracing back by unplugging -------------------------------------------

export const partLayer = (id: PartId) => (id === "embed" ? -1 : id.startsWith("m") ? Number(id.slice(1)) : Number(id.split(".")[0]));

export function unplug(id: PartId): Intervention {
  if (id.startsWith("m")) return { kind: "mlp", layer: Number(id.slice(1)), scale: 0 };
  const [layer, head] = id.split(".").map(Number);
  return { kind: "head", layer, head, scale: 0 };
}

// Every head and MLP that runs before `id` can affect it: heads and MLPs
// in earlier layers, and for an MLP, also the heads in its own layer.
export function partsBefore(id: PartId | "answer", cfg: ModelConfig): PartId[] {
  const { n_layers: L, n_heads: H } = cfg;
  const out: PartId[] = [];
  const limit = id === "answer" ? L : partLayer(id);
  for (let l = 0; l < L; l++) {
    const heads = l < limit || (id !== "answer" && id.startsWith("m") && l === limit);
    if (heads) for (let h = 0; h < H; h++) out.push(`${l}.${h}`);
    if (l < limit) out.push(`m${l}`);
  }
  return out;
}

// The answer's score difference with one part unplugged at a time.
export async function scanAnswer(engine: GpuEngine, tokens: number[], position: number, target: number, rival: number, parts: PartId[], onProgress: (done: number) => void) {
  const L = engine.config.n_layers;
  const diff = async (interventions: Intervention[]) => {
    const t = await engine.run({ tokens, interventions, lens: { positions: [position], topK: 1, targets: [target, rival] } });
    return t.lens!.targetLogprobs[L * 2] - t.lens!.targetLogprobs[L * 2 + 1];
  };
  const base = await diff([]);
  const out = new Map<PartId, number>();
  for (let i = 0; i < parts.length; i++) {
    out.set(parts[i], (await diff([unplug(parts[i])])) - base);
    if (i % 6 === 5) onProgress(i + 1);
  }
  onProgress(parts.length);
  return { base, change: out };
}

// What a part writes at a position.
function partOutput(trace: Trace, w: Weights, id: PartId, position: number) {
  if (id.startsWith("m")) return mlpOutput(trace, w.config.d_model, Number(id.slice(1)), position);
  const [l, h] = id.split(".").map(Number);
  return headOutput(trace, w, l, h, position);
}

// How much a part's output at `position` changes when each earlier part is
// unplugged, as a share of the output's size.
export async function scanPart(engine: GpuEngine, w: Weights, tokens: number[], position: number, id: PartId, parts: PartId[], onProgress: (done: number) => void) {
  const read = id.startsWith("m") ? { resid: true, residMid: true } : { z: true };
  const base = partOutput(await engine.run({ tokens, read }), w, id, position);
  const size = Math.sqrt(dot(base, base)) || 1;
  const out = new Map<PartId, number>();
  for (let i = 0; i < parts.length; i++) {
    const v = partOutput(await engine.run({ tokens, read, interventions: [unplug(parts[i])] }), w, id, position);
    let s = 0;
    for (let k = 0; k < v.length; k++) s += (v[k] - base[k]) ** 2;
    out.set(parts[i], Math.sqrt(s) / size);
    if (i % 6 === 5) onProgress(i + 1);
  }
  onProgress(parts.length);
  return out;
}

// --- Inside an MLP ---------------------------------------------------------

// One neuron's input before GELU: the LayerNormed stream dotted with its
// column of the input weights, plus its bias. Returns the 768 products too.
export function neuronInput(trace: Trace, w: Weights, l: number, position: number, n: number) {
  const { d_model: d, d_mlp, ln_eps } = w.config;
  const T = trace.tokens.length;
  const x = trace.residMid!.subarray((l * T + position) * d, (l * T + position + 1) * d);
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) v += (x[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(v / d + ln_eps);
  const gain = w.f32(`h.${l}.ln_2.weight`);
  const lnb = w.f32(`h.${l}.ln_2.bias`);
  const Win = w.f32(`h.${l}.mlp.c_fc.weight`);
  const products = new Float32Array(d);
  let sum = 0;
  for (let i = 0; i < d; i++) {
    products[i] = ((x[i] - mean) * scale * gain[i] + lnb[i]) * Win[i * d_mlp + n];
    sum += products[i];
  }
  const bias = w.f32(`h.${l}.mlp.c_fc.bias`)[n];
  return { products, bias, pre: sum + bias };
}

export const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)));

// The direction a neuron writes, per unit of activation.
export function neuronDirection(w: Weights, l: number, n: number) {
  const d = w.config.d_model;
  return w.f32(`h.${l}.mlp.c_proj.weight`).slice(n * d, (n + 1) * d);
}

export function neuronsAt(trace: Trace, cfg: ModelConfig, l: number, position: number) {
  const T = trace.tokens.length;
  return trace.neurons!.subarray((l * T + position) * cfg.d_mlp, (l * T + position + 1) * cfg.d_mlp);
}

export interface NeuronInfo {
  perMillion: number;
  top: number;
  firing: number[];
  passages: { ids: number[]; acts: number[]; at: number }[];
}

const neuronFiles = new Map<number, Promise<NeuronInfo[]>>();
export function neuronData(layer: number) {
  let p = neuronFiles.get(layer);
  if (!p) {
    p = fetch(new URL(`neurons/${String(layer).padStart(2, "0")}.json`, document.baseURI))
      .then((r) => {
        if (!r.ok) throw new Error(`neuron data: ${r.status}`);
        return r.json();
      })
      .then((d: { neurons: [number, number, number[], [number[], number[], number][]][] }) =>
        d.neurons.map(([perMillion, top, firing, passages]) => ({ perMillion, top, firing, passages: passages.map(([ids, acts, at]) => ({ ids, acts, at })) })));
    p.catch(() => neuronFiles.delete(layer));
    neuronFiles.set(layer, p);
  }
  return p;
}

// --- Weights ---------------------------------------------------------------

// The top three directions the 1,024 position embeddings vary along (PCA by
// power iteration), and each position's place along them.
export function positionPca(w: Weights, k = 3) {
  const d = w.config.d_model;
  const P = w.f32("wpe.weight");
  const n = P.length / d;
  const X = new Float32Array(n * d);
  const mean = new Float64Array(d);
  for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) mean[i] += P[p * d + i] / n;
  for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) X[p * d + i] = P[p * d + i] - mean[i];
  let total = 0;
  for (let i = 0; i < X.length; i++) total += X[i] * X[i];
  const comps: Float64Array[] = [];
  for (let c = 0; c < k; c++) {
    let v = new Float64Array(d).map((_, i) => Math.sin(i * (c + 1) + 1));
    for (let it = 0; it < 100; it++) {
      const xv = new Float64Array(n);
      for (let p = 0; p < n; p++) xv[p] = dot(X.subarray(p * d, (p + 1) * d), v);
      const next = new Float64Array(d);
      for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) next[i] += X[p * d + i] * xv[p];
      for (const u of comps) {
        const s = dot(next, u);
        for (let i = 0; i < d; i++) next[i] -= s * u[i];
      }
      const norm = Math.sqrt(dot(next, next)) || 1;
      v = next.map((x) => x / norm);
    }
    comps.push(v);
  }
  const explained: number[] = [];
  const coords = comps.map((v) => {
    const out = new Float32Array(n);
    for (let p = 0; p < n; p++) out[p] = dot(X.subarray(p * d, (p + 1) * d), v);
    explained.push(dot(out, out) / total);
    return out;
  });
  return { coords, explained };
}

// A head's copying table: for each token in `ids`, what the head would write
// if it attended fully to that token, scored against every token in `ids`.
// Inputs are the tokens as GPT-2 sees them after layer 0 (in effect their
// embedding), read from real one-token runs. Each row is centred and scaled,
// so a bright diagonal means "copies the token it looks at".
export async function copyingTable(engine: GpuEngine, w: Weights, l: number, h: number, ids: number[]) {
  const { n_heads: H, d_model: d, ln_eps } = w.config;
  const dh = d / H;
  const Wqkv = w.f32(`h.${l}.attn.c_attn.weight`);
  const bqkv = w.f32(`h.${l}.attn.c_attn.bias`);
  const WO = w.f32(`h.${l}.attn.c_proj.weight`);
  const g1 = w.f32(`h.${l}.ln_1.weight`);
  const b1 = w.f32(`h.${l}.ln_1.bias`);
  const gf = w.f32("ln_f.weight");
  const targets = ids.map((id) => tokenVector(w, id));
  const ln = (x: Float32Array, g: Float32Array, b: Float32Array | null) => {
    let m = 0;
    for (let i = 0; i < d; i++) m += x[i];
    m /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[i] - m) ** 2;
    const s = 1 / Math.sqrt(v / d + ln_eps);
    return Float32Array.from(x, (xi, i) => (xi - m) * s * g[i] + (b ? b[i] : 0));
  };
  const rows: number[][] = [];
  for (const id of ids) {
    // resid is [layer, position, d]; after layer 0 at position 1.
    const t = await engine.run({ tokens: [BOS, id], read: { resid: true } });
    const x = ln(t.resid!.slice((1 * 2 + 1) * d, (1 * 2 + 2) * d), g1, b1);
    const v = new Float32Array(dh);
    for (let k = 0; k < dh; k++) {
      let s = bqkv[2 * d + h * dh + k];
      for (let i = 0; i < d; i++) s += x[i] * Wqkv[i * 3 * d + 2 * d + h * dh + k];
      v[k] = s;
    }
    const out = new Float32Array(d);
    for (let k = 0; k < dh; k++) for (let o = 0; o < d; o++) out[o] += v[k] * WO[(h * dh + k) * d + o];
    const y = ln(out, gf, null);
    const row = targets.map((tv) => dot(y, tv));
    const m = row.reduce((a, b) => a + b, 0) / row.length;
    const c = row.map((r) => r - m);
    const n = Math.sqrt(c.reduce((a, b) => a + b * b, 0)) || 1;
    rows.push(c.map((r) => r / n));
  }
  return rows;
}

// The heads that most strongly feed `id`'s query, key or value: up to three,
// each at least twice the typical score.
export function strongestSources(w: Wiring, kind: "Q" | "K" | "V", id: string) {
  const k = w.kinds[kind];
  return (k.heads[id] ?? []).filter(([, s]) => s >= 2 * k.median).slice(0, 3);
}

export interface Wiring {
  source: string;
  kinds: Record<"Q" | "K" | "V", { median: number; p99: number; heads: Record<string, [string, number][]> }>;
}

let wiring: Promise<Wiring> | null = null;
export const loadWiring = () => (wiring ??= fetch(new URL("zoom-data/wiring.json", document.baseURI)).then((r) => r.json()));

let tokenMap: Promise<Int16Array> | null = null;
export const loadTokenMap = () => (tokenMap ??= fetch(new URL("zoom-data/token-map.bin", document.baseURI)).then((r) => r.arrayBuffer()).then((b) => new Int16Array(b)));
