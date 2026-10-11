// Turns a raw trace into what the Signal view draws. Everything here is
// computed from the run; nothing is tuned to look good.

import type { Trace } from "../engine/hooks.ts";
import type { ModelConfig } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";

export interface Reading {
  piece: string;
  p: number; // probability of the top guess
  top: { piece: string; p: number }[];
  finalP: number; // probability given to the model's final answer
}

export interface Analysis {
  pieces: string[];
  position: number;
  arcs: { layer: number; from: number; weight: number }[]; // attention into `position`, averaged over heads
  mlp: number[][]; // [layer][token] size of the MLP's write, 0..1
  finals: string[]; // the model's final answer at each position
  lens: Reading[]; // n_layers + 1 readings at `position`: before layer 0, then after each layer
  final: string;
  first: number; // first reading whose top guess is the final answer
  changes: number; // times the top guess changes after the first layer
}

export function analyse(trace: Trace, finalsTrace: Trace, cfg: ModelConfig, tok: Tokenizer, position: number): Analysis {
  const { n_layers: L, n_heads: H, d_model: d } = cfg;
  const T = trace.tokens.length;
  const pieces = trace.tokens.map((id, i) => (i === 0 ? "⟨start⟩" : tok.piece(id)));
  const lens = trace.lens!;
  const K = lens.topK;

  const arcs: Analysis["arcs"] = [];
  for (let l = 0; l < L; l++) {
    for (let j = 0; j <= position; j++) {
      if (j === position) continue;
      let w = 0;
      for (let h = 0; h < H; h++) w += trace.pattern![((l * H + h) * T + position) * T + j];
      w /= H;
      if (w >= 0.03) arcs.push({ layer: l, from: j, weight: w });
    }
  }

  // Size of each MLP write. The start token's are huge and would flatten the
  // rest, so the scale comes from the other tokens and the start is capped.
  const raw: number[][] = [];
  let max = 1e-9;
  for (let l = 0; l < L; l++) {
    raw.push([]);
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let i = 0; i < d; i++) {
        const v = trace.resid![((l + 1) * T + t) * d + i] - trace.residMid![(l * T + t) * d + i];
        s += v * v;
      }
      raw[l].push(Math.sqrt(s));
      if (t > 0) max = Math.max(max, Math.sqrt(s));
    }
  }
  const mlp = raw.map((row) => row.map((v) => Math.min(1, v / max)));

  const row = (layer: number, pos: number) => layer * T + pos;
  const finalId = (pos: number) => lens.ids[row(L, pos) * K];
  const finals = trace.tokens.map((_, t) => tok.piece(finalId(t)));
  const targetIndex = finalsTrace.lens!.targets.indexOf(finalId(position));
  const readings: Reading[] = [];
  for (let l = 0; l <= L; l++) {
    const r = row(l, position);
    const top = Array.from({ length: K }, (_, k) => ({ piece: tok.piece(lens.ids[r * K + k]), p: Math.exp(lens.logprobs[r * K + k]) }));
    const nT = finalsTrace.lens!.targets.length;
    readings.push({ piece: top[0].piece, p: top[0].p, top, finalP: Math.exp(finalsTrace.lens!.targetLogprobs[r * nT + targetIndex]) });
  }
  const final = readings[L].piece;
  let changes = 0;
  for (let l = 2; l <= L; l++) if (readings[l].piece !== readings[l - 1].piece) changes++;
  return { pieces, position, arcs, mlp, finals, lens: readings, final, first: readings.findIndex((r) => r.piece === final), changes };
}

// Readings are numbered as layers: reading 0 is before layer 0 (just the
// embedding), reading k is after layer k - 1.
export const readingLabel = (k: number) => (k === 0 ? "before layer 0" : `after layer ${k - 1}`);

export const show = (piece: string) => piece.replace(/ /g, "␣").replace(/\n/g, "↵");

// For prose: the word without its leading space, unless that's all it is.
export const word = (piece: string) => piece.trim() || show(piece);
