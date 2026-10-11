// The layer-7 sparse autoencoder's features: loading, naming and steering.

import { useEffect, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention } from "../engine/hooks.ts";
import { loadSae } from "../engine/load.ts";
import { BOS } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { LABELS } from "./feature-labels.ts";

export const SAE_LAYER = 7;

export interface FeatureInfo {
  perMillion: number; // how often it fires on Wikipedia text
  top: number; // its strongest activation there
  firing: number[]; // tokens it fires on most
  pushes: number[]; // tokens it pushes toward most
}

export interface FeatureDetail {
  up: number[];
  down: number[];
  passages: { ids: number[]; acts: number[]; at: number }[];
}

type Sae = Awaited<ReturnType<typeof loadSae>>;
export type SaeState = { status: "loading" } | { status: "ready"; sae: Sae; index: FeatureInfo[] } | { status: "error"; message: string };

const SAE_BASE = new URL("model/sae-l7/", document.baseURI).href;
const DATA_BASE = new URL("features/", document.baseURI).href;

let state: SaeState = { status: "loading" };
let started: Promise<void> | null = null;
let listeners: ((s: SaeState) => void)[] = [];

function start(engine: GpuEngine) {
  return (async () => {
    try {
      const [sae, index] = await Promise.all([
        loadSae(SAE_BASE),
        fetch(new URL("index.json", DATA_BASE)).then((r) => r.json()),
      ]);
      engine.attachSae(sae.manifest.layer, sae.manifest.d_sae, sae.encoder, sae.bias);
      state = {
        status: "ready",
        sae,
        index: (index.features as [number, number, number[], number[]][]).map(([perMillion, top, firing, pushes]) => ({ perMillion, top, firing, pushes })),
      };
    } catch (e) {
      state = { status: "error", message: e instanceof Error ? e.message : String(e) };
      started = null;
    }
    listeners.forEach((l) => l(state));
  })();
}

// Loads the feature dictionary the first time a page asks for it.
export function useSae(engine: GpuEngine): SaeState {
  const [s, setS] = useState(state);
  useEffect(() => {
    listeners.push(setS);
    started ??= start(engine);
    return () => {
      listeners = listeners.filter((l) => l !== setS);
    };
  }, [engine]);
  return s;
}

const details = new Map<number, Promise<FeatureDetail[]>>();
export function featureDetail(id: number): Promise<FeatureDetail> {
  const n = Math.floor(id / 256);
  if (!details.has(n)) details.set(n, fetch(new URL(`${String(n).padStart(2, "0")}.json`, DATA_BASE)).then((r) => r.json()));
  return details.get(n)!.then((rows) => rows[id % 256]);
}

// A feature's name: a hand-written label, or the tokens it fires on most.
export function featureName(id: number, info: FeatureInfo | undefined, tok: Tokenizer) {
  if (LABELS[id]) return LABELS[id];
  if (!info || info.firing.length === 0) return "Never fired on our sample";
  return "Fires on " + info.firing.map((t) => `“${tok.piece(t).trim() || tok.piece(t)}”`).join(", ");
}

// Features whose name, firing tokens or pushed words match a word.
export function searchFeatures(index: FeatureInfo[], tok: Tokenizer, query: string, limit = 12) {
  const s = query.trim().toLowerCase();
  const out: number[] = [];
  if (s.length < 2) return out;
  for (let id = 0; id < index.length && out.length < limit; id++) {
    const f = index[id];
    if (!f.firing.length) continue;
    const words = [...f.firing, ...f.pushes].map((t) => tok.piece(t).trim().toLowerCase());
    if ((LABELS[id] ?? "").toLowerCase().includes(s) || words.includes(s)) out.push(id);
  }
  return out;
}

// The words a feature pushes toward, as a short hint next to its name.
export function featureHint(info: FeatureInfo | undefined, tok: Tokenizer) {
  return info?.pushes.length ? "→ " + info.pushes.map((t) => tok.piece(t).trim()).join(", ") : "";
}

// The strongest features at one token, strongest first.
export function topAt(features: Float32Array, size: number, t: number, k = 8) {
  const row = features.subarray(t * size, (t + 1) * size);
  const best: number[] = [];
  for (let i = 0; i < size; i++) {
    if (row[i] <= 0) continue;
    if (best.length < k || row[i] > row[best[best.length - 1]]) {
      best.push(i);
      best.sort((a, b) => row[b] - row[a]);
      if (best.length > k) best.pop();
    }
  }
  return best.map((id) => ({ id, value: row[id] }));
}

// Add `strength` times a feature's direction to the stream entering layer 7,
// at every token but the start token.
export function steering(direction: Float32Array, strength: number, T: number): Intervention {
  const d = direction.length;
  const value = new Float32Array(T * d);
  for (let t = 1; t < T; t++) for (let i = 0; i < d; i++) value[t * d + i] = strength * direction[i];
  return { kind: "resid", hook: "resid_pre", layer: SAE_LAYER, position: "all", mode: "add", value };
}

function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Write `n` tokens, optionally steered. The same seed gives the same dice
// rolls, so two runs differ only because of the steering.
export async function write(engine: GpuEngine, tok: Tokenizer, prompt: number[], n: number, seed: number, steer?: { direction: Float32Array; strength: number }, onToken?: (text: string) => void) {
  const L = engine.config.n_layers;
  const K = 40;
  const roll = rng(seed);
  const tokens = prompt.slice();
  let text = "";
  for (let i = 0; i < n; i++) {
    const interventions = steer ? [steering(steer.direction, steer.strength, tokens.length)] : [];
    const trace = await engine.run({ tokens, interventions, lens: { positions: [tokens.length - 1], topK: K } });
    const ids = trace.lens!.ids.subarray(L * K, (L + 1) * K);
    const lps = trace.lens!.logprobs.subarray(L * K, (L + 1) * K);
    const weights = Array.from(lps, (lp) => Math.exp(lp / 0.8));
    let r = roll() * weights.reduce((a, b) => a + b, 0);
    let pick = 0;
    while (r > weights[pick] && pick < K - 1) r -= weights[pick++];
    const id = ids[pick];
    if (id === BOS) break;
    tokens.push(id);
    text += tok.piece(id);
    onToken?.(text);
  }
  return text;
}
