import { useEffect, useState } from "react";
import { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import { loadModel, type LoadProgress } from "../engine/load.ts";
import { BOS, type Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { measureJobs, type HeadJobs } from "./heads.ts";

export type ModelState =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "ready"; engine: GpuEngine; tokenizer: Tokenizer; weights: Weights; jobs: HeadJobs }
  | { status: "error"; message: string };

// Where the weights live; a <meta name="featurama-model"> in index.html can point elsewhere.
const MODEL_BASE = new URL(document.querySelector<HTMLMetaElement>('meta[name="featurama-model"]')?.content ?? "model/gpt2/", location.href).href;

let listeners: ((s: ModelState) => void)[] = [];
let state: ModelState = { status: "loading", progress: null };
let started = false;

function set(s: ModelState) {
  state = s;
  listeners.forEach((l) => l(s));
}

async function start() {
  if (!("gpu" in navigator)) {
    set({ status: "error", message: "Featurama runs GPT-2 on your graphics card with WebGPU, which this browser doesn't offer. Try a recent Chrome, Edge or Safari on a desktop or laptop." });
    return;
  }
  try {
    const { weights, tokenizer } = await loadModel(MODEL_BASE, (progress) => set({ status: "loading", progress }));
    const engine = await GpuEngine.create(weights);
    const jobs = await measureJobs(engine);
    set({ status: "ready", engine, tokenizer, weights, jobs });
  } catch (e) {
    set({ status: "error", message: e instanceof Error ? e.message : String(e) });
  }
}

export function useModel(): ModelState {
  const [s, setS] = useState(state);
  useEffect(() => {
    listeners.push(setS);
    if (!started) {
      started = true;
      start();
    }
    return () => {
      listeners = listeners.filter((l) => l !== setS);
    };
  }, []);
  return s;
}

export const MAX_PROMPT_TOKENS = 48;

// The start token plus the prompt, cut to fit; `cut` says whether text was dropped.
export function tokenize(tok: Tokenizer, text: string) {
  const ids = [BOS, ...tok.encode(text)];
  const tokens = ids.slice(0, MAX_PROMPT_TOKENS);
  return Object.assign(tokens, { cut: ids.length > tokens.length });
}

// One run reads everything the Signal view needs, with the lens at every
// position; a second asks how likely each position's final answer was at
// every layer.
export async function runPrompt(engine: GpuEngine, tokens: number[]): Promise<{ trace: Trace; finals: Trace }> {
  const positions = tokens.map((_, i) => i);
  const trace = await engine.run({ tokens, read: { resid: true, residMid: true, pattern: true }, lens: { positions, topK: 5 } });
  const L = engine.config.n_layers;
  const finalIds = [...new Set(positions.map((p) => trace.lens!.ids[(L * tokens.length + p) * 5]))];
  const finals = await engine.run({ tokens, lens: { positions, topK: 1, targets: finalIds } });
  return { trace, finals };
}
