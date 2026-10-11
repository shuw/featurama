// The prompt, how GPT-2 splits it (instantly, as you type), and what GPT-2
// would say next (a moment later). Clicking a prediction adds it.

import type { Tokenizer } from "../engine/tokenizer.ts";
import { show } from "./analysis.ts";
import { MAX_PROMPT_TOKENS } from "./model.ts";

export interface Prediction {
  piece: string;
  p: number;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  tokens: number[] & { cut?: boolean };
  tokenizer: Tokenizer;
  next: Prediction[] | null; // null hides the predictions (for example, before a guess)
  stale: boolean;
  disabled?: boolean;
  disabledNote?: string;
}

export function PromptBox({ value, onChange, tokens, tokenizer, next, stale, disabled, disabledNote }: Props) {
  const pieces = tokens.slice(1).map((id) => ({ id, text: tokenizer.piece(id) }));
  return (
    <section className="promptbox" aria-label="Your prompt">
      <div className="promptbox-input">
        <input value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled} spellCheck={false} maxLength={400}
          aria-label="Prompt" placeholder="Type anything…" />
        {disabled && disabledNote && <span className="promptbox-note">{disabledNote}</span>}
      </div>
      <div className="promptbox-tokens" aria-label={`${pieces.length} tokens`}>
        <span className="row-label">{pieces.length} token{pieces.length === 1 ? "" : "s"}</span>
        <span className="token-strip">
          {pieces.length === 0 && <span className="muted">Each piece GPT-2 reads will appear here.</span>}
          {pieces.map((p, i) => (
            <span key={i} className={`tk tk${i % 4}`} title={`Token ${p.id}${p.text.includes("�") ? " (part of a character that spans several tokens)" : ""}`}>
              {show(p.text)}
            </span>
          ))}
          {tokens.cut && <span className="tk-cut">cut at {MAX_PROMPT_TOKENS - 1}</span>}
        </span>
      </div>
      {next && (
        <div className={stale ? "promptbox-next stale" : "promptbox-next"}>
          <span className="row-label">Next token</span>
          <span className="next-strip">
            {next.map((n, i) => (
              <button key={i} className="next-chip" onClick={() => onChange(value + n.piece)} title={`Add “${show(n.piece)}” to the prompt`} disabled={disabled}>
                <span className="next-piece">{show(n.piece)}</span>
                <span className="next-p">{pct(n.p)}</span>
                <span className="next-bar" style={{ width: `${Math.max(2, n.p * 100)}%` }} />
              </button>
            ))}
          </span>
        </div>
      )}
    </section>
  );
}

export const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
