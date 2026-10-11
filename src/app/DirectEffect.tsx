// What one part's output, on its own, pushes the next token toward and away from.

import { useEffect, useState } from "react";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show, word } from "./analysis.ts";
import { GLOSSARY } from "./glossary.ts";
import { directEffect } from "./heads.ts";
import { Term } from "./Tip.tsx";

// Whole words (with their leading space) and numbers. Word fragments like
// "rodu" and GPT-2's glitch tokens like " RandomRedditor" (barely trained,
// so they pile up at the extremes) are real but tell a reader little.
let allowed: Uint8Array | null = null;
function wordTokens(tok: Tokenizer, V: number) {
  if (!allowed) {
    allowed = new Uint8Array(V);
    for (let t = 0; t < V; t++) allowed[t] = /^( \p{L}\p{Ll}+| \p{Lu}{2,}| ?\p{N}+)$/u.test(tok.piece(t)) ? 1 : 0;
  }
  return allowed;
}

export function DirectEffect({ trace, weights, tokenizer, vector, position, off }: {
  trace: Trace; weights: Weights; tokenizer: Tokenizer; vector: () => Float32Array; position: number; off?: boolean;
}) {
  const [effect, setEffect] = useState<ReturnType<typeof directEffect> | null>(null);
  // Wait for the pointer to settle: each calculation scores all 50,257 tokens.
  useEffect(() => {
    setEffect(null);
    if (off) return;
    const t = setTimeout(() => setEffect(directEffect(trace, weights, vector(), position, 5, wordTokens(tokenizer, weights.config.d_vocab))), 120);
    return () => clearTimeout(t);
  }, [trace, position, vector, off]);
  const at = position === 0 ? "the start token" : `“${word(tokenizer.piece(trace.tokens[position]))}”`;
  return (
    <div className="effect">
      <Term tip={GLOSSARY.direct}><span className="card-label">What it pushes the next token toward, at {at}</span></Term>
      {off ? (
        <p className="muted small">Unplugged: it pushes nothing.</p>
      ) : !effect ? (
        <p className="muted small">Working it out…</p>
      ) : (
        <>
          <div className="push-row"><span className="push-label">toward</span>{effect.up.map((e) => <span key={e.id} className="push push-up">{show(tokenizer.piece(e.id))}</span>)}</div>
          <div className="push-row"><span className="push-label">away from</span>{effect.down.map((e) => <span key={e.id} className="push push-down">{show(tokenizer.piece(e.id))}</span>)}</div>
        </>
      )}
    </div>
  );
}
