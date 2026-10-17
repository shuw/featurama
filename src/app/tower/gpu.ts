// Draws the weight tower with WebGPU, on the engine's own device, reading
// the weights straight from the buffers the forward pass uses. Far away, a
// pyramid of block averages stands in for the weights (built once on the
// GPU, and again for the forward pass whenever the prompt changes); close up,
// each weight is read individually.

import type { GpuEngine } from "../../engine/gpu/engine.ts";
import { FLOOR, type Floor, type Tile } from "./layout.ts";

const LEVELS = [4, 16, 64]; // block sizes of the pyramid
const TILE_WORDS = 32; // 8 vec4s per tile

const F16 = /* wgsl */ `
fn w_at(k: u32) -> f32 {
  let p = unpack2x16float(weights[k >> 1u]);
  return select(p.x, p.y, (k & 1u) == 1u);
}`;

const PYRAMID = /* wgsl */ `
struct P { rows: u32, cols: u32, stride: u32, colStart: u32, f: u32, outOff: u32, outCols: u32, kind: u32, mode: u32, xOff: u32, pick: u32, actOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> acts: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
${F16}
fn value(r: u32, c: u32) -> f32 {
  var v: f32;
  if (p.kind == 1u) { v = acts[p.actOff + r * p.cols + c]; } else { v = w_at(r * p.stride + p.colStart + c); }
  if (p.mode == 1u) { v *= acts[p.xOff + r]; }
  if (p.mode == 2u) { v *= acts[p.xOff + c]; }
  if (p.mode == 3u && r != p.pick) { v = 0.0; }
  return v;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let outRows = (p.rows + p.f - 1u) / p.f;
  let i = g.x + g.y * 65535u * 64u;
  if (i >= outRows * p.outCols) { return; }
  let pr = i / p.outCols;
  let pc = i % p.outCols;
  var s = 0.0;
  var s2 = 0.0;
  var n = 0.0;
  for (var a = 0u; a < p.f; a++) {
    let r = pr * p.f + a;
    if (r >= p.rows) { break; }
    for (var b = 0u; b < p.f; b++) {
      let c = pc * p.f + b;
      if (c >= p.cols) { break; }
      let v = value(r, c);
      s += v;
      s2 += v * v;
      n += 1.0;
    }
  }
  out[p.outOff + i] = pack2x16float(vec2f(s / max(n, 1.0), sqrt(s2 / max(n, 1.0))));
}`;

// Tile record, 8 vec4s:
//  0 origin (xyz, relative to the eye), floor
//  1 w, d, rows, cols
//  2 kind, stride, colStart, actOff            (u32)
//  3 foldRows, foldGap, foldCount, heads       (f32)
//  4 mode, xOff, pick, 0                       (u32)
//  5 raw pyramid offsets L1 L2 L3, 0           (u32)
//  6 forward pyramid offsets L1 L2 L3, has-fwd (u32)
//  7 rawScale, fwdScale, progress, 0         (progress: how far the slow forward pass has got through this tile, 0..1)
const RENDER = /* wgsl */ `
struct Cam { viewProj: mat4x4f, screen: vec4f, mode: vec4f } // mode: forward, slow, focus floor, focus strength
struct T { o: vec4f, size: vec4f, src: vec4u, fold: vec4f, fwd: vec4u, pyr: vec4u, pyrF: vec4u, scale: vec4f }
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> tiles: array<T>;
@group(0) @binding(2) var<storage, read> pyramid: array<u32>;
@group(0) @binding(3) var<storage, read> acts: array<f32>;
@group(1) @binding(0) var<storage, read> weights: array<u32>;
${F16}
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) t: u32 }
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ti: u32) -> V {
  let corners = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1), vec2f(0, 1));
  let uv = corners[vi];
  let t = tiles[ti];
  let world = t.o.xyz + vec3f(uv.x * t.size.x, 0.0, uv.y * t.size.y);
  var out: V;
  out.pos = cam.viewProj * vec4f(world, 1.0);
  out.uv = uv;
  out.t = ti;
  return out;
}

const POS = vec3f(0.48, 0.58, 1.0);
const NEG = vec3f(1.0, 0.48, 0.27);
const BASE = vec3f(0.09, 0.10, 0.13);

fn pooled(off: u32, r: u32, c: u32, k: u32, cols: u32) -> vec2f {
  let shift = 2u * (k + 1u);
  let pc = (cols + (1u << shift) - 1u) >> shift;
  return unpack2x16float(pyramid[off + (r >> shift) * pc + (c >> shift)]);
}

fn shade(t: T, r: u32, c: u32, level: u32, cols: u32, forward: bool) -> vec3f {
  var mean: f32;
  var rms: f32;
  if (level == 0u) {
    var v: f32;
    if (t.src.x == 1u) { v = acts[t.src.w + r * cols + c]; } else { v = w_at(r * t.src.y + t.src.z + c); }
    if (forward) {
      if (t.fwd.x == 1u) { v *= acts[t.fwd.y + r]; }
      if (t.fwd.x == 2u) { v *= acts[t.fwd.y + c]; }
      if (t.fwd.x == 3u && r != t.fwd.z) { v = 0.0; }
    }
    mean = v;
    rms = abs(v);
  } else {
    let off = select(t.pyr, t.pyrF, forward);
    let m = pooled(off[level - 1u], r, c, level - 1u, cols);
    mean = m.x;
    rms = m.y;
  }
  let scale = select(t.scale.x, t.scale.y, forward);
  let strength = pow(clamp(rms / (scale * 1.6), 0.0, 1.0), 0.55);
  let lean = clamp(mean / max(rms, 1e-9), -1.0, 1.0);
  let hue = mix(vec3f(0.6, 0.62, 0.68), select(NEG, POS, lean > 0.0), abs(lean));
  return mix(BASE, hue, strength);
}

// Floors above the focus fade out by dropping a growing share of pixels in a
// fixed pattern, which reads as a smooth dissolve without sorting.
fn dissolve(px: vec2f, level: f32) -> bool {
  let above = clamp((level - cam.mode.z - 0.4) / 0.4, 0.0, 1.0) * smoothstep(0.15, 0.6, cam.mode.w);
  let n = fract(sin(dot(floor(px), vec2f(12.9898, 78.233))) * 43758.5453);
  return n < above;
}

@fragment
fn fs(in: V) -> @location(0) vec4f {
  let t = tiles[in.t];
  // Floors above the one in focus would sit between it and the camera: they dissolve.
  if (dissolve(in.pos.xy, t.o.w)) { discard; }
  let rows = u32(t.size.z);
  let cols = u32(t.size.w);
  // Continuous position in the tensor; the token table is wrapped into columns.
  var rr = in.uv.y * select(t.size.z, t.fold.x, t.fold.z > 0.0);
  var cc = in.uv.x * select(t.size.w, t.size.x, t.fold.z > 0.0);
  let density = max(length(vec2f(dpdx(rr), dpdy(rr))), length(vec2f(dpdx(cc), dpdy(cc))));
  if (t.fold.z > 0.0) {
    let span = t.size.w + t.fold.y;
    let fold = floor(cc / span);
    cc = cc - fold * span;
    if (cc >= t.size.w || fold >= t.fold.z) { discard; }
    rr = rr + fold * t.fold.x;
  }
  let r = u32(rr);
  let c = u32(cc);
  if (r >= rows) { discard; }

  // How far the animated pass has lit this spot: a soft front moving down
  // the tile. Off, everything is lit.
  let progress = t.scale.z;
  let lit = select(clamp((progress - in.uv.y) / 0.12 + 0.5, 0.0, 1.0), 1.0, cam.mode.y < 0.5);
  let canForward = cam.mode.x > 0.5 && t.pyrF.w == 1u;
  var level = 0u;
  if (density > 1.5) { level = 1u; }
  if (density > 6.0) { level = 2u; }
  if (density > 24.0) { level = 3u; }

  // Lit: the forward pass (or the weight); not yet lit: the plain weight, dimmed.
  let litCol = shade(t, r, c, level, cols, canForward);
  var col = litCol;
  if (lit < 1.0) { col = mix(mix(BASE, shade(t, r, c, level, cols, false), 0.32), litCol, lit); }
  let forward = canForward && lit > 0.5;
  // The glow at the front of the pass.
  if (cam.mode.y > 0.5 && progress > -0.1 && progress < 1.1) {
    let d = abs(in.uv.y - progress);
    col = mix(col, vec3f(1.0, 0.9, 0.55), 0.7 * exp(-d * d / 0.0015));
  }
  // Close to a floor, the others fade back, more the further away they are.
  let away = clamp(abs(t.o.w - cam.mode.z) - 0.35, 0.0, 1.0);
  col = mix(col, vec3f(0.03, 0.035, 0.045), 0.88 * cam.mode.w * away);
  // The token's (or position's) row, when only that row is used.
  if (forward && t.fwd.x == 3u) {
    let shift = select(0u, 2u * level, level > 0u);
    if ((r >> shift) == (t.fwd.z >> shift)) { col = mix(col, vec3f(1.0, 0.85, 0.3), 0.7); }
  }
  // Up close, a fine grid between weights; a line between heads.
  if (density < 0.08) {
    let e = min(min(fract(rr), 1.0 - fract(rr)), min(fract(cc), 1.0 - fract(cc)));
    if (e < density * 1.2) { col *= 0.55; }
  }
  if (t.fold.w > 0.5 && density < 8.0) {
    let hc = fract(cc / 64.0) * 64.0;
    if (min(hc, 64.0 - hc) < density * 1.5) { col = mix(col, vec3f(0.9), 0.25); }
  }
  return vec4f(col, 1.0);
}

// The floor plates under the tiles.
struct FV { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) floor: f32 }
struct Plate { o: vec4f, size: vec4f }
@group(0) @binding(4) var<storage, read> plates: array<Plate>;
@vertex
fn vsPlate(@builtin(vertex_index) vi: u32, @builtin(instance_index) pi: u32) -> FV {
  let corners = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1), vec2f(0, 1));
  let uv = corners[vi];
  let p = plates[pi];
  var out: FV;
  out.pos = cam.viewProj * vec4f(p.o.xyz + vec3f(uv.x * p.size.x, 0.0, uv.y * p.size.y), 1.0);
  out.uv = uv;
  out.floor = p.o.w;
  return out;
}
@fragment
fn fsPlate(in: FV) -> @location(0) vec4f {
  if (dissolve(in.pos.xy, in.floor)) { discard; }
  let edge = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
  let glow = smoothstep(0.004, 0.0, edge);
  var a = mix(0.45, 0.9, glow);
  if (cam.mode.w > 0.0 && abs(in.floor - cam.mode.z) > 0.5) { a *= 1.0 - 0.85 * cam.mode.w; }
  return vec4f(mix(vec3f(0.07, 0.08, 0.1), vec3f(0.35, 0.38, 0.46), glow), a);
}`;

export interface Acts {
  data: Float32Array;
  offsets: Map<string, number>;
  token: number;
  position: number;
}

export class TowerRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private tileBuf: GPUBuffer;
  private plateBuf: GPUBuffer;
  private camBuf: GPUBuffer;
  private pyramid: GPUBuffer;
  private actBuf: GPUBuffer;
  private dummy: GPUBuffer;
  private pipeline: GPURenderPipeline;
  private platePipeline: GPURenderPipeline;
  private compute: GPUComputePipeline;
  private depth?: GPUTexture;
  private group0?: GPUBindGroup;
  private plateGroup?: GPUBindGroup;
  private weightGroups = new Map<string, GPUBindGroup>();
  private raw: number[][] = []; // per tile: pyramid offsets for the plain weights
  private fwd: (number[] | null)[] = []; // and for the forward pass
  private tileData: ArrayBuffer;
  readonly scales: { raw: number; fwd: number }[];

  constructor(private engine: GpuEngine, private canvas: HTMLCanvasElement, readonly tiles: Tile[], readonly floors: Floor[], actSize: number) {
    this.device = engine.device;
    this.context = canvas.getContext("webgpu")!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.tileData = new ArrayBuffer(tiles.length * TILE_WORDS * 4);
    this.tileBuf = this.device.createBuffer({ size: this.tileData.byteLength, usage: S });
    this.plateBuf = this.device.createBuffer({ size: floors.length * 32, usage: S });
    this.camBuf = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.actBuf = this.device.createBuffer({ size: Math.max(16, actSize * 4), usage: S });
    this.dummy = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });

    // Room in the pyramid for every tile, plain and forward.
    let words = 0;
    for (const t of tiles) {
      const levels = LEVELS.map((f) => Math.ceil(t.rows / f) * Math.ceil(t.cols / f));
      const at = (n: number) => {
        const o = words;
        words += n;
        return o;
      };
      this.raw.push(levels.map(at));
      this.fwd.push(t.forward.mode === "none" ? null : levels.map(at));
    }
    this.pyramid = this.device.createBuffer({ size: Math.max(16, words * 4), usage: GPUBufferUsage.STORAGE });
    this.scales = tiles.map(() => ({ raw: 1, fwd: 1 }));

    const module = this.device.createShaderModule({ code: RENDER, label: "tower" });
    const target = { format: this.format };
    const depthStencil: GPUDepthStencilState = { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" };
    this.pipeline = this.device.createRenderPipeline({ layout: "auto", vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [target] }, primitive: { topology: "triangle-list" }, depthStencil });
    this.platePipeline = this.device.createRenderPipeline({
      layout: "auto", vertex: { module, entryPoint: "vsPlate" },
      fragment: { module, entryPoint: "fsPlate", targets: [{ format: this.format, blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one" } } }] },
      primitive: { topology: "triangle-list" }, depthStencil: { format: "depth32float", depthWriteEnabled: false, depthCompare: "greater" },
    });
    this.compute = this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code: PYRAMID, label: "tower pyramid" }), entryPoint: "main" } });
    this.buildPyramid(false);
  }

  private weightGroup(pipeline: GPURenderPipeline, t: Tile) {
    const key = t.source.kind === "weight" ? t.source.tensor : "";
    let g = this.weightGroups.get(key);
    if (!g) {
      const buffer = t.source.kind === "weight" ? this.engine.weightBuffer(t.source.tensor) : this.dummy;
      g = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer } }] });
      this.weightGroups.set(key, g);
    }
    return g;
  }

  // Block averages of every tile's values: plain, or as used by the forward pass.
  buildPyramid(forward: boolean, acts?: Acts) {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.compute);
    const temp: GPUBuffer[] = [];
    this.tiles.forEach((t, i) => {
      const offs = forward ? this.fwd[i] : this.raw[i];
      if (!offs) return;
      if (t.source.kind === "act" && !acts) return;
      LEVELS.forEach((f, k) => {
        const outCols = Math.ceil(t.cols / f);
        const count = Math.ceil(t.rows / f) * outCols;
        const fw = t.forward;
        const mode = !forward ? 0 : fw.mode === "row" ? 1 : fw.mode === "col" ? 2 : fw.mode === "pick" ? 3 : 0;
        const xOff = forward && (fw.mode === "row" || fw.mode === "col") ? acts!.offsets.get(fw.act)! : 0;
        const pick = forward && fw.mode === "pick" ? (fw.pick === "token" ? acts!.token : acts!.position) : 0;
        const src = t.source;
        const u = new Uint32Array([t.rows, t.cols, src.kind === "weight" ? src.stride : 1, src.kind === "weight" ? src.colStart : 0, f, offs[k], outCols,
          src.kind === "act" ? 1 : 0, mode, xOff, pick, src.kind === "act" ? acts!.offsets.get(src.act)! : 0, 0, 0, 0, 0]);
        const ub = this.device.createBuffer({ size: u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(ub, 0, u);
        temp.push(ub);
        const bind = this.device.createBindGroup({
          layout: this.compute.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: ub } },
            { binding: 1, resource: { buffer: src.kind === "weight" ? this.engine.weightBuffer(src.tensor) : this.dummy } },
            { binding: 2, resource: { buffer: this.actBuf } },
            { binding: 3, resource: { buffer: this.pyramid } },
          ],
        });
        pass.setBindGroup(0, bind);
        const groups = Math.ceil(count / 64);
        pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      });
    });
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.device.queue.onSubmittedWorkDone().then(() => temp.forEach((b) => b.destroy()));
  }

  // New activations: upload them and rebuild the forward-pass pyramid.
  setActs(acts: Acts, scales: { raw: number; fwd: number }[]) {
    this.device.queue.writeBuffer(this.actBuf, 0, acts.data.buffer as ArrayBuffer, acts.data.byteOffset, acts.data.byteLength);
    scales.forEach((s, i) => (this.scales[i] = s));
    this.buildPyramid(false, acts);
    this.buildPyramid(true, acts);
  }

  // progress: per tile, how far the slow forward pass has got (1 when not in slow motion).
  // focus: the height in view, in floors (fractional), and how strongly to fade the others (0..1).
  render(viewProj: Float32Array, eye: [number, number, number], mode: { forward: boolean; slow: boolean; focus: number; fade: number }, acts: Acts | null, progress: Float32Array) {
    const dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    if (!this.depth || this.depth.width !== w || this.depth.height !== h) {
      this.depth?.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const cam = new Float32Array(24);
    cam.set(viewProj, 0);
    cam.set([w, h, 0, 0], 16);
    cam.set([mode.forward ? 1 : 0, mode.slow ? 1 : 0, mode.focus, mode.fade], 20);
    this.device.queue.writeBuffer(this.camBuf, 0, cam);

    // Positions relative to the eye, computed in double precision, so a
    // single weight stays sharp even 40,000 units from the origin.
    const f = new Float32Array(this.tileData);
    const u = new Uint32Array(this.tileData);
    this.tiles.forEach((t, i) => {
      const b = i * TILE_WORDS;
      const y = t.floor * FLOOR;
      f.set([t.x - eye[0], y - eye[1], t.z - eye[2], t.floor], b);
      f.set([t.w, t.d, t.rows, t.cols], b + 4);
      const src = t.source;
      u.set([src.kind === "act" ? 1 : 0, src.kind === "weight" ? src.stride : 1, src.kind === "weight" ? src.colStart : 0, src.kind === "act" && acts ? acts.offsets.get(src.act)! : 0], b + 8);
      f.set([t.fold?.rows ?? 0, t.fold?.gap ?? 0, t.fold?.count ?? 0, t.heads ? 1 : 0], b + 12);
      const fw = t.forward;
      u.set([
        fw.mode === "row" ? 1 : fw.mode === "col" ? 2 : fw.mode === "pick" ? 3 : 0,
        acts && (fw.mode === "row" || fw.mode === "col") ? acts.offsets.get(fw.act)! : 0,
        acts && fw.mode === "pick" ? (fw.pick === "token" ? acts.token : acts.position) : 0,
        0,
      ], b + 16);
      u.set([...this.raw[i], 0], b + 20);
      u.set(this.fwd[i] ? [...this.fwd[i]!, acts ? 1 : 0] : [0, 0, 0, 0], b + 24);
      f.set([this.scales[i].raw, this.scales[i].fwd, progress[i], 0], b + 28);
    });
    this.device.queue.writeBuffer(this.tileBuf, 0, this.tileData);
    const plates = new Float32Array(this.floors.length * 8);
    this.floors.forEach((fl, i) => plates.set([fl.x0 - eye[0], fl.y - 2 - eye[1], fl.z0 - eye[2], fl.index, fl.x1 - fl.x0, fl.z1 - fl.z0, 0, 0], i * 8));
    this.device.queue.writeBuffer(this.plateBuf, 0, plates);

    this.group0 ??= this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.tileBuf } },
        { binding: 2, resource: { buffer: this.pyramid } },
        { binding: 3, resource: { buffer: this.actBuf } },
      ],
    });
    this.plateGroup ??= this.device.createBindGroup({
      layout: this.platePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 4, resource: { buffer: this.plateBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.03, g: 0.035, b: 0.045, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group0);
    this.tiles.forEach((t, i) => {
      pass.setBindGroup(1, this.weightGroup(this.pipeline, t));
      pass.draw(6, 1, 0, i);
    });
    pass.setPipeline(this.platePipeline);
    pass.setBindGroup(0, this.plateGroup);
    pass.draw(6, this.floors.length, 0, 0);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  destroy() {
    for (const b of [this.tileBuf, this.plateBuf, this.camBuf, this.pyramid, this.actBuf, this.dummy]) b.destroy();
    this.depth?.destroy();
  }
}
