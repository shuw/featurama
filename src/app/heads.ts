// What each attention head does: measured on the model in this browser, or
// (for the IOI roles) taken from the paper that found them.

import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import { BOS, type Weights } from "../engine/model.ts";

export type Job = "previous" | "induction" | "duplicate";

export interface HeadJobs {
  // [layer * n_heads + head] -> score 0..1 for each measured job
  scores: Record<Job, Float32Array>;
}

export const JOB_THRESHOLD: Record<Job, number> = { previous: 0.5, induction: 0.6, duplicate: 0.4 };

export const JOB_TEXT: Record<Job, { name: string; text: string }> = {
  previous: { name: "Previous-token head", text: "Each word looks at the word just before it." },
  induction: { name: "Induction head", text: "Spots a repeat and looks at what came after the earlier copy, to predict it comes again." },
  duplicate: { name: "Duplicate-token head", text: "Looks back at earlier copies of the same word." },
};

// Roles found by Wang et al. (2022), "Interpretability in the Wild", for
// sentences like "When Mary and John went to the store, John gave a drink to".
export const IOI_ROLES: Record<string, string> = {
  "9.9": "Name mover: copies the right name toward the answer.",
  "9.6": "Name mover: copies the right name toward the answer.",
  "10.0": "Name mover: copies the right name toward the answer.",
  "10.7": "Negative name mover: pushes against the name it looks at.",
  "11.10": "Negative name mover: pushes against the name it looks at.",
  "7.3": "S-inhibition: tells the name movers which name to ignore.",
  "7.9": "S-inhibition: tells the name movers which name to ignore.",
  "8.6": "S-inhibition: tells the name movers which name to ignore.",
  "8.10": "S-inhibition: tells the name movers which name to ignore.",
};
export const IOI_SOURCE = "Wang et al. 2022, Interpretability in the Wild";

// Scores every head on a sequence of 20 random tokens shown twice: the
// standard test for previous-token, duplicate-token and induction heads.
export async function measureJobs(engine: GpuEngine): Promise<HeadJobs> {
  const { n_layers: L, n_heads: H } = engine.config;
  let seed = 1939;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const N = 20;
  const half = Array.from({ length: N }, () => 1000 + Math.floor(rand() * 19000));
  const tokens = [BOS, ...half, ...half];
  const T = tokens.length;
  const { pattern } = await engine.run({ tokens, read: { pattern: true } });
  const at = (l: number, h: number, i: number, j: number) => pattern![((l * H + h) * T + i) * T + j];
  const scores = { previous: new Float32Array(L * H), induction: new Float32Array(L * H), duplicate: new Float32Array(L * H) };
  for (let l = 0; l < L; l++) {
    for (let h = 0; h < H; h++) {
      let prev = 0;
      for (let i = 2; i < T; i++) prev += at(l, h, i, i - 1);
      let ind = 0;
      let dup = 0;
      for (let i = N + 1; i < T - 1; i++) {
        ind += at(l, h, i, i - N + 1);
        dup += at(l, h, i, i - N);
      }
      scores.previous[l * H + h] = prev / (T - 2);
      scores.induction[l * H + h] = ind / (N - 1);
      scores.duplicate[l * H + h] = dup / (N - 1);
    }
  }
  return { scores };
}

export function jobsOf(jobs: HeadJobs, l: number, h: number, H: number): Job[] {
  return (Object.keys(JOB_THRESHOLD) as Job[]).filter((j) => jobs.scores[j][l * H + h] >= JOB_THRESHOLD[j]);
}

// How much each head writes into the residual stream at one position: the
// length of its output after the output projection. [layer * n_heads + head]
export function headActivity(trace: Trace, w: Weights, position: number): Float32Array {
  const { n_layers: L, n_heads: H, d_model: d } = w.config;
  const dh = d / H;
  const T = trace.tokens.length;
  const out = new Float32Array(L * H);
  const acc = new Float64Array(d);
  for (let l = 0; l < L; l++) {
    const WO = w.f32(`h.${l}.attn.c_proj.weight`);
    for (let h = 0; h < H; h++) {
      acc.fill(0);
      for (let k = 0; k < dh; k++) {
        const zk = trace.z![(l * T + position) * d + h * dh + k];
        const row = (h * dh + k) * d;
        for (let o = 0; o < d; o++) acc[o] += zk * WO[row + o];
      }
      let s = 0;
      for (let o = 0; o < d; o++) s += acc[o] * acc[o];
      out[l * H + h] = Math.sqrt(s);
    }
  }
  return out;
}

// How much each MLP writes at one position. [layer]
export function mlpActivity(trace: Trace, d: number, L: number, position: number): Float32Array {
  const T = trace.tokens.length;
  const out = new Float32Array(L);
  for (let l = 0; l < L; l++) {
    let s = 0;
    for (let i = 0; i < d; i++) {
      const v = trace.resid![((l + 1) * T + position) * d + i] - trace.residMid![(l * T + position) * d + i];
      s += v * v;
    }
    out[l] = Math.sqrt(s);
  }
  return out;
}

// Attention from `query` to each earlier position, for one head.
export function attentionRow(trace: Trace, H: number, l: number, h: number, query: number): number[] {
  const T = trace.tokens.length;
  return Array.from({ length: T }, (_, j) => (j <= query ? trace.pattern![((l * H + h) * T + query) * T + j] : 0));
}

// Average attention each position pays to the one before it (from position 2 on).
export function previousScore(trace: Trace, H: number, l: number, h: number): number {
  const T = trace.tokens.length;
  if (T < 4) return 0;
  let s = 0;
  for (let i = 2; i < T; i++) s += trace.pattern![((l * H + h) * T + i) * T + i - 1];
  return s / (T - 2);
}

// Where a token repeats an earlier one, how much it attends to the token that
// followed the earlier copy. Needs at least three repeats to count.
export function inductionScore(trace: Trace, H: number, l: number, h: number): number {
  const T = trace.tokens.length;
  let s = 0;
  let n = 0;
  for (let i = 2; i < T; i++) {
    let j = -1;
    for (let k = i - 1; k >= 1; k--) if (trace.tokens[k] === trace.tokens[i]) { j = k; break; }
    if (j < 0 || j + 1 >= i) continue;
    s += trace.pattern![((l * H + h) * T + i) * T + j + 1];
    n++;
  }
  return n >= 3 ? s / n : 0;
}
