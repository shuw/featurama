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

// Each job has a one-letter badge for the head grids.
export const JOB_TEXT: Record<Job, { letter: string; name: string; text: string }> = {
  previous: { letter: "P", name: "Previous-token head", text: "Looks at the token just before." },
  duplicate: { letter: "D", name: "Duplicate-token head", text: "Looks back at earlier copies of the same token." },
  induction: { letter: "I", name: "Induction head", text: "When a token has appeared before, it looks at what came next last time, so GPT-2 can predict the same again." },
};

// Roles found by Wang et al. (2022), "Interpretability in the Wild", for
// sentences like "When Mary and John went to the store, John gave a drink to".
export type Role = "N" | "B" | "S" | "−N";
export const ROLES: Record<Role, { letter: Role; name: string; text: string }> = {
  N: { letter: "N", name: "Name mover", text: "Copies the right name into the answer." },
  B: { letter: "B", name: "Backup name mover", text: "Takes over copying the name when the name movers are switched off." },
  S: { letter: "S", name: "Name inhibitor", text: "Tells the name movers which name to skip (the paper calls this S-inhibition)." },
  "−N": { letter: "−N", name: "Negative name mover", text: "Pushes the answer away from the name it looks at, which keeps GPT-2 from being overconfident." },
};
export const IOI_ROLE: Record<string, Role> = {
  "9.9": "N", "9.6": "N", "10.0": "N",
  "9.0": "B", "9.7": "B", "10.1": "B", "10.2": "B", "10.6": "B", "10.10": "B", "11.2": "B", "11.9": "B",
  "10.7": "−N", "11.10": "−N",
  "7.3": "S", "7.9": "S", "8.6": "S", "8.10": "S",
};
export const IOI_ROLES: Record<string, string> = Object.fromEntries(
  Object.entries(IOI_ROLE).map(([id, r]) => [id, `${ROLES[r].name}: ${ROLES[r].text.charAt(0).toLowerCase()}${ROLES[r].text.slice(1)}`]),
);
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

// The vector a head writes at one position: its mix of values through its
// output projection.
export function headOutput(trace: Trace, w: Weights, l: number, h: number, position: number): Float32Array {
  const { n_heads: H, d_model: d } = w.config;
  const dh = d / H;
  const T = trace.tokens.length;
  const WO = w.f32(`h.${l}.attn.c_proj.weight`);
  const out = new Float32Array(d);
  for (let k = 0; k < dh; k++) {
    const zk = trace.z![(l * T + position) * d + h * dh + k];
    const row = (h * dh + k) * d;
    for (let o = 0; o < d; o++) out[o] += zk * WO[row + o];
  }
  return out;
}

// The vector an MLP writes at one position.
export function mlpOutput(trace: Trace, d: number, l: number, position: number): Float32Array {
  const T = trace.tokens.length;
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = trace.resid![((l + 1) * T + position) * d + i] - trace.residMid![(l * T + position) * d + i];
  return out;
}

let F16: Float32Array | null = null;

// Direct effect: what one part's output, on its own, does to the next-token
// scores at a position, through the final LayerNorm (using that position's
// real scale) and the unembedding. Later layers can change the real outcome.
export function directEffect(trace: Trace, w: Weights, vec: Float32Array, position: number, k = 5, allowed?: Uint8Array) {
  const { n_layers: L, d_model: d, d_vocab: V } = w.config;
  const T = trace.tokens.length;
  if (!F16) F16 = f16Table();
  const x = trace.resid!.subarray((L * T + position) * d, (L * T + position + 1) * d);
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) v += (x[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(v / d + w.config.ln_eps);
  const gain = w.f32("ln_f.weight");
  let cm = 0;
  for (let i = 0; i < d; i++) cm += vec[i];
  cm /= d;
  const eff = new Float32Array(d);
  for (let i = 0; i < d; i++) eff[i] = (vec[i] - cm) * scale * gain[i];
  const wte = w.raw("wte.weight");
  const logits = new Float32Array(V);
  for (let t = 0; t < V; t++) {
    let s = 0;
    const base = t * d;
    for (let i = 0; i < d; i++) s += eff[i] * F16[wte[base + i]];
    logits[t] = s;
  }
  const order = (sign: 1 | -1) => {
    const best: number[] = [];
    for (let t = 0; t < V; t++) {
      if (allowed && !allowed[t]) continue;
      if (best.length < k || sign * logits[t] > sign * logits[best[best.length - 1]]) {
        best.push(t);
        best.sort((a, b) => sign * (logits[b] - logits[a]));
        if (best.length > k) best.pop();
      }
    }
    return best.map((id) => ({ id, value: logits[id] }));
  };
  return { up: order(1), down: order(-1) };
}

function f16Table() {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    t[h] = exp === 0 ? sign * 2 ** -14 * (frac / 1024) : exp === 31 ? (frac ? NaN : sign * Infinity) : sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return t;
}
