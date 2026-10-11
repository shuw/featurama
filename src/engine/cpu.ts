// A plain, slow forward pass on the CPU. It exists to check the GPU engine
// against, so it favours being obviously right over being fast.

import type { Weights } from "./model.ts";

export interface CpuTrace {
  tokens: number[];
  resid: Float32Array[]; // n_layers + 1 snapshots, each [T, d_model]: before layer 0, then after each layer
  residMid: Float32Array[]; // after each layer's attention, [T, d_model]
  pattern: Float32Array[]; // each [n_heads, T, T]
}

function layerNorm(x: Float32Array, T: number, d: number, g: Float32Array, b: Float32Array, eps: number) {
  const out = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[t * d + i];
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[t * d + i] - mean) ** 2;
    const s = 1 / Math.sqrt(v / d + eps);
    for (let i = 0; i < d; i++) out[t * d + i] = (x[t * d + i] - mean) * s * g[i] + b[i];
  }
  return out;
}

// x [T, n] times W [n, m] (Hugging Face Conv1D layout) plus bias.
function linear(x: Float32Array, T: number, n: number, W: Float32Array, m: number, bias: Float32Array) {
  const out = new Float32Array(T * m);
  const acc = new Float64Array(m);
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < m; o++) acc[o] = bias[o];
    for (let i = 0; i < n; i++) {
      const xi = x[t * n + i];
      const row = i * m;
      for (let o = 0; o < m; o++) acc[o] += xi * W[row + o];
    }
    out.set(acc, t * m);
  }
  return out;
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

export function runCpu(w: Weights, tokens: number[]): CpuTrace {
  const { n_layers: L, n_heads: H, d_model: d, d_mlp, ln_eps } = w.config;
  const T = tokens.length;
  const dh = d / H;
  const wte = w.f32("wte.weight");
  const wpe = w.f32("wpe.weight");

  let x = new Float32Array(T * d);
  for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) x[t * d + i] = wte[tokens[t] * d + i] + wpe[t * d + i];
  const trace: CpuTrace = { tokens, resid: [x.slice()], residMid: [], pattern: [] };

  for (let l = 0; l < L; l++) {
    const p = (n: string) => w.f32(`h.${l}.${n}`);
    const a = layerNorm(x, T, d, p("ln_1.weight"), p("ln_1.bias"), ln_eps);
    const qkv = linear(a, T, d, p("attn.c_attn.weight"), 3 * d, p("attn.c_attn.bias"));
    const z = new Float32Array(T * d);
    const pattern = new Float32Array(H * T * T);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < T; i++) {
        const scores = new Float64Array(i + 1);
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let k = 0; k < dh; k++) s += qkv[i * 3 * d + h * dh + k] * qkv[j * 3 * d + d + h * dh + k];
          scores[j] = s / Math.sqrt(dh);
          max = Math.max(max, scores[j]);
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) sum += scores[j] = Math.exp(scores[j] - max);
        for (let j = 0; j <= i; j++) {
          const pj = scores[j] / sum;
          pattern[(h * T + i) * T + j] = pj;
          for (let k = 0; k < dh; k++) z[i * d + h * dh + k] += pj * qkv[j * 3 * d + 2 * d + h * dh + k];
        }
      }
    }
    trace.pattern.push(pattern);
    const attnOut = linear(z, T, d, p("attn.c_proj.weight"), d, p("attn.c_proj.bias"));
    for (let i = 0; i < x.length; i++) x[i] += attnOut[i];
    trace.residMid.push(x.slice());

    const m = layerNorm(x, T, d, p("ln_2.weight"), p("ln_2.bias"), ln_eps);
    const pre = linear(m, T, d, p("mlp.c_fc.weight"), d_mlp, p("mlp.c_fc.bias"));
    for (let i = 0; i < pre.length; i++) pre[i] = gelu(pre[i]);
    const mlpOut = linear(pre, T, d_mlp, p("mlp.c_proj.weight"), d, p("mlp.c_proj.bias"));
    for (let i = 0; i < x.length; i++) x[i] += mlpOut[i];
    trace.resid.push(x.slice());
  }
  return trace;
}

// Final LayerNorm then unembedding, for one residual vector: the logit lens
// when applied to an earlier layer's residual.
export function unembedCpu(w: Weights, resid: Float32Array): Float32Array {
  const { d_model: d, d_vocab, ln_eps } = w.config;
  const x = layerNorm(resid, 1, d, w.f32("ln_f.weight"), w.f32("ln_f.bias"), ln_eps);
  const wte = w.f32("wte.weight");
  const logits = new Float32Array(d_vocab);
  for (let v = 0; v < d_vocab; v++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += x[i] * wte[v * d + i];
    logits[v] = s;
  }
  return logits;
}
