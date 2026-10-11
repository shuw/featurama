// WGSL compute kernels. Weights are fp16, packed two to a u32 and unpacked
// with unpack2x16float, so no shader-f16 support is needed. All arithmetic is
// fp32: GPT-2's residual stream holds values near 3,000, which fp16 can't
// square for LayerNorm.

const F16_AT = (name: string) => /* wgsl */ `
fn ${name}_at(k: u32) -> f32 {
  let w = unpack2x16float(${name}[k >> 1u]);
  return select(w.x, w.y, (k & 1u) == 1u);
}`;

export const EMBED = /* wgsl */ `
struct P { T: u32, d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<u32>;
@group(0) @binding(3) var<storage, read> wpe: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${F16_AT("wte")}
${F16_AT("wpe")}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.T * p.d) { return; }
  let t = g.x / p.d;
  let i = g.x % p.d;
  out[g.x] = wte_at(tokens[t] * p.d + i) + wpe_at(t * p.d + i);
}`;

// One workgroup per row. Row r reads input row rows[r], so the logit lens can
// pick any (layer, position) out of the stored residual stream.
export const LAYERNORM = /* wgsl */ `
struct P { R: u32, d: u32, eps: f32, xOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> gain: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
${F16_AT("gain")}
${F16_AT("bias")}
var<workgroup> red: array<f32, 256>;
fn reduce(v: f32, l: u32) -> f32 {
  red[l] = v;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l < k) { red[l] += red[l + k]; }
    workgroupBarrier();
  }
  let total = red[0];
  workgroupBarrier();
  return total;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let l = lid.x;
  let src = p.xOff + rows[wg.x] * p.d;
  let dst = wg.x * p.d;
  var s = 0.0;
  for (var i = l; i < p.d; i += 256u) { s += x[src + i]; }
  let mean = reduce(s, l) / f32(p.d);
  var v = 0.0;
  for (var i = l; i < p.d; i += 256u) { let c = x[src + i] - mean; v += c * c; }
  let inv = inverseSqrt(reduce(v, l) / f32(p.d) + p.eps);
  for (var i = l; i < p.d; i += 256u) {
    out[dst + i] = (x[src + i] - mean) * inv * gain_at(i) + bias_at(i);
  }
}`;

// out[t, o] = x[t, :] . W[:, o] + bias[o], then optionally GELU, then
// optionally plus a residual. W is [n, m] (Hugging Face Conv1D layout). Each
// thread computes two adjacent outputs (one packed u32 of W) for 8 tokens.
export const LINEAR = /* wgsl */ `
struct P { T: u32, n: u32, m: u32, flags: u32, resOff: u32, outOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<u32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
const TT = 8u;
const CH = 64u;
var<workgroup> xs: array<f32, 512>;
fn gelu(v: vec2f) -> vec2f {
  let u = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), vec2f(-15.0), vec2f(15.0));
  return 0.5 * v * (1.0 + tanh(u));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let l = lid.x;
  let o = (wg.x * 64u + l) * 2u;
  let t0 = wg.y * TT;
  var acc: array<vec2f, 8>;
  for (var i0 = 0u; i0 < p.n; i0 += CH) {
    for (var k = 0u; k < TT; k++) {
      let t = t0 + k;
      var v = 0.0;
      if (t < p.T) { v = x[t * p.n + i0 + l]; }
      xs[k * CH + l] = v;
    }
    workgroupBarrier();
    if (o < p.m) {
      for (var c = 0u; c < CH; c++) {
        let w = unpack2x16float(W[((i0 + c) * p.m + o) >> 1u]);
        for (var k = 0u; k < TT; k++) { acc[k] += xs[k * CH + c] * w; }
      }
    }
    workgroupBarrier();
  }
  if (o >= p.m) { return; }
  let b = unpack2x16float(bias[o >> 1u]);
  for (var k = 0u; k < TT; k++) {
    let t = t0 + k;
    if (t >= p.T) { break; }
    var y = acc[k] + b;
    if ((p.flags & 1u) != 0u) { y = gelu(y); }
    let j = t * p.m + o;
    if ((p.flags & 2u) != 0u) { y += vec2f(res[p.resOff + j], res[p.resOff + j + 1u]); }
    out[p.outOff + j] = y.x;
    out[p.outOff + j + 1u] = y.y;
  }
}`;

// Causal attention pattern for one (query position, head) per workgroup.
export const SCORES = /* wgsl */ `
struct P { T: u32, d: u32, dh: u32, patOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> pattern: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let l = lid.x;
  let i = wg.x;
  let h = wg.y;
  let row = p.patOff + (h * p.T + i) * p.T;
  let q = i * 3u * p.d + h * p.dh;
  let scale = inverseSqrt(f32(p.dh));
  var mx = -3.0e38;
  for (var j = l; j < p.T; j += 64u) {
    if (j <= i) {
      let k = j * 3u * p.d + p.d + h * p.dh;
      var s = 0.0;
      for (var c = 0u; c < p.dh; c++) { s += qkv[q + c] * qkv[k + c]; }
      s *= scale;
      pattern[row + j] = s;
      mx = max(mx, s);
    }
  }
  red[l] = mx;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (l < k) { red[l] = max(red[l], red[l + k]); }
    workgroupBarrier();
  }
  mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = l; j < p.T; j += 64u) {
    if (j <= i) {
      let e = exp(pattern[row + j] - mx);
      pattern[row + j] = e;
      sum += e;
    }
  }
  red[l] = sum;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (l < k) { red[l] += red[l + k]; }
    workgroupBarrier();
  }
  sum = red[0];
  for (var j = l; j < p.T; j += 64u) {
    if (j <= i) { pattern[row + j] /= sum; } else { pattern[row + j] = 0.0; }
  }
}`;

// z[i, h*dh + c] = headScale[h] * sum_j pattern[h, i, j] * v[j, h*dh + c].
// headScale is how ablation reaches a single head.
export const MIX = /* wgsl */ `
struct P { T: u32, d: u32, dh: u32, patOff: u32, scaleOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> pattern: array<f32>;
@group(0) @binding(3) var<storage, read> headScale: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.T * p.d) { return; }
  let i = g.x / p.d;
  let c = g.x % p.d;
  let h = c / p.dh;
  let row = p.patOff + (h * p.T + i) * p.T;
  var s = 0.0;
  for (var j = 0u; j <= i; j++) { s += pattern[row + j] * qkv[j * 3u * p.d + 2u * p.d + c]; }
  z[g.x] = s * headScale[p.scaleOff + h];
}`;

// Set or add a vector into an activation buffer: activation patching and steering.
export const EDIT = /* wgsl */ `
struct P { offset: u32, n: u32, mode: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.n) { return; }
  let k = p.offset + g.x;
  if (p.mode == 0u) { dst[k] = value[g.x]; } else { dst[k] += value[g.x]; }
}`;

// logits[r, v] = x[r, :] . wte[v, :] (the unembedding is the token embedding
// transposed). Each workgroup covers 64 tokens of the vocabulary for 4 rows.
export const UNEMBED = /* wgsl */ `
struct P { R: u32, d: u32, V: u32, xOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> wte: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
const RT = 4u;
var<workgroup> xs: array<f32, 3072>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let l = lid.x;
  let r0 = wg.y * RT;
  for (var k = l; k < RT * p.d; k += 64u) {
    let r = r0 + k / p.d;
    var v = 0.0;
    if (r < p.R) { v = x[p.xOff + r * p.d + k % p.d]; }
    xs[k] = v;
  }
  workgroupBarrier();
  let v = wg.x * 64u + l;
  if (v >= p.V) { return; }
  var acc = vec4f(0.0);
  let base = (v * p.d) >> 1u;
  for (var i = 0u; i < p.d; i += 2u) {
    let w = unpack2x16float(wte[base + (i >> 1u)]);
    acc += vec4f(
      xs[i] * w.x + xs[i + 1u] * w.y,
      xs[p.d + i] * w.x + xs[p.d + i + 1u] * w.y,
      xs[2u * p.d + i] * w.x + xs[2u * p.d + i + 1u] * w.y,
      xs[3u * p.d + i] * w.x + xs[3u * p.d + i + 1u] * w.y);
  }
  for (var k = 0u; k < RT; k++) {
    if (r0 + k < p.R) { out[(r0 + k) * p.V + v] = acc[k]; }
  }
}`;

// Per row: log-softmax, the K most likely tokens, and the log-probability of
// each requested target token. Ties go to the lower token id.
export const TOPK = /* wgsl */ `
struct P { R: u32, V: u32, K: u32, nTargets: u32, rowOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> lps: array<f32>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let l = lid.x;
  let base = wg.x * p.V;
  var mx = -3.0e38;
  for (var v = l; v < p.V; v += 256u) { mx = max(mx, logits[base + v]); }
  rv[l] = mx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l < k) { rv[l] = max(rv[l], rv[l + k]); }
    workgroupBarrier();
  }
  mx = rv[0];
  workgroupBarrier();
  var s = 0.0;
  for (var v = l; v < p.V; v += 256u) { s += exp(logits[base + v] - mx); }
  rv[l] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l < k) { rv[l] += rv[l + k]; }
    workgroupBarrier();
  }
  let lse = mx + log(rv[0]);
  workgroupBarrier();
  var lastV = 3.0e38;
  var lastI = 0u;
  for (var k = 0u; k < p.K; k++) {
    var bv = -3.0e38;
    var bi = 0xffffffffu;
    for (var v = l; v < p.V; v += 256u) {
      let x = logits[base + v];
      let eligible = x < lastV || (x == lastV && v > lastI);
      if (eligible && (x > bv || (x == bv && v < bi))) { bv = x; bi = v; }
    }
    rv[l] = bv;
    ri[l] = bi;
    workgroupBarrier();
    for (var s2 = 128u; s2 > 0u; s2 >>= 1u) {
      if (l < s2) {
        let ov = rv[l + s2];
        let oi = ri[l + s2];
        if (ov > rv[l] || (ov == rv[l] && oi < ri[l])) { rv[l] = ov; ri[l] = oi; }
      }
      workgroupBarrier();
    }
    lastV = rv[0];
    lastI = ri[0];
    if (l == 0u) {
      ids[(p.rowOff + wg.x) * p.K + k] = lastI;
      lps[(p.rowOff + wg.x) * (p.K + p.nTargets) + k] = lastV - lse;
    }
    workgroupBarrier();
  }
  for (var t = l; t < p.nTargets; t += 256u) {
    lps[(p.rowOff + wg.x) * (p.K + p.nTargets) + p.K + t] = logits[base + targets[t]] - lse;
  }
}`;
