// "Let it write": GPT-2 picks the next token, adds it, and repeats.

import { useEffect, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { tokenize } from "./model.ts";

export type Wildness = "calm" | "lively" | "wild";
// Sampling temperature: calm always takes the top guess.
const TEMPERATURE: Record<Wildness, number> = { calm: 0, lively: 0.8, wild: 1.3 };
const MAX_STEPS = 40;
const TOP_K = 40;

export interface Writer {
  writing: boolean;
  toggle: () => void;
  stop: () => void;
  wildness: Wildness;
  setWildness: (w: Wildness) => void;
}

export function useWriter(engine: GpuEngine | null, tokenizer: Tokenizer | null, prompt: string, append: (text: string) => void): Writer {
  const [writing, setWriting] = useState(false);
  const [wildness, setWildness] = useState<Wildness>("lively");
  const promptRef = useRef(prompt);
  promptRef.current = prompt;
  const wildRef = useRef(wildness);
  wildRef.current = wildness;

  useEffect(() => {
    if (!writing || !engine || !tokenizer) return;
    let cancelled = false;
    (async () => {
      let text = promptRef.current;
      const L = engine.config.n_layers;
      for (let step = 0; step < MAX_STEPS && !cancelled; step++) {
        const tokens = tokenize(tokenizer, text);
        if (tokens.cut) break;
        const trace = await engine.run({ tokens, lens: { positions: [tokens.length - 1], topK: TOP_K } });
        if (cancelled) break;
        let ids = trace.lens!.ids.subarray(L * TOP_K, (L + 1) * TOP_K);
        let lps = trace.lens!.logprobs.subarray(L * TOP_K, (L + 1) * TOP_K);
        // After a typed trailing space, only tokens that start with it fit.
        const fits = Array.from(ids, (id) => tokenizer.piece(id).startsWith(tokens.trailing));
        if (tokens.trailing && fits.some(Boolean)) {
          ids = ids.filter((_, i) => fits[i]);
          lps = lps.filter((_, i) => fits[i]);
        }
        const piece = tokenizer.piece(pick(ids, lps, TEMPERATURE[wildRef.current]));
        text += piece.startsWith(tokens.trailing) ? piece.slice(tokens.trailing.length) : piece;
        append(text);
        await new Promise((r) => setTimeout(r, 140));
      }
      if (!cancelled) setWriting(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [writing, engine, tokenizer]);

  return { writing, toggle: () => setWriting((w) => !w), stop: () => setWriting(false), wildness, setWildness };
}

// Sample among the top tokens, sharpened or flattened by the temperature.
function pick(ids: Uint32Array, logprobs: Float32Array, temperature: number) {
  if (temperature === 0) return ids[0];
  const weights = Array.from(logprobs, (lp) => Math.exp((lp - logprobs[0]) / temperature));
  let r = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < weights.length; i++) if ((r -= weights[i]) <= 0) return ids[i];
  return ids[0];
}
