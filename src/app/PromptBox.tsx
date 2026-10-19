// The prompt, how GPT-2 splits it (instantly, as you type), and what GPT-2
// would say next (a moment later). The likeliest next token shows in grey
// after the cursor, new lines included; Tab accepts it, and clicking any
// prediction adds it.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show } from "./analysis.ts";
import { GLOSSARY } from "./glossary.ts";
import { MAX_PROMPT_TOKENS } from "./model.ts";
import { Tip } from "./Tip.tsx";
import type { Wildness, Writer } from "./useWriter.ts";

export interface Prediction {
  piece: string;
  p: number;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  tokens: number[] & { cut?: boolean; trailing?: string };
  tokenizer: Tokenizer;
  next: Prediction[] | null; // null shows `nextNote` instead
  nextNote?: string;
  stale: boolean;
  disabled?: boolean;
  disabledNote?: string;
  autoFocus?: boolean;
  onShuffle?: () => void;
  writer?: Writer;
  onTokenHover?: (position: number | null) => void; // position in the prompt, counting the start token as 0
  activeToken?: number | null;
}

const WILDNESS: { value: Wildness; label: string; tip: string }[] = [
  { value: "calm", label: "calm", tip: "Always takes GPT-2's top guess" },
  { value: "lively", label: "lively", tip: "Usually takes a likely token, sometimes a less likely one" },
  { value: "wild", label: "wild", tip: "Often takes unlikely tokens: stranger writing" },
];

// How a suggested piece looks in grey: new lines show as ↵ before breaking.
const ghostText = (piece: string) => piece.replace(/\n/g, "↵\n");

export function PromptBox({ value, onChange, tokens, tokenizer, next, nextNote = "Running GPT-2…", stale, disabled, disabledNote, autoFocus, onShuffle, writer, onTokenHover, activeToken }: Props) {
  const input = useRef<HTMLTextAreaElement>(null);
  const mirror = useRef<HTMLDivElement>(null);
  const [caretAtEnd, setCaretAtEnd] = useState(true);
  const pieces = tokens.slice(1).map((id) => ({ id, text: tokenizer.piece(id) }));
  // With a trailing space typed, only guesses that start with it fit; adding one types the rest.
  const trailing = tokens.trailing ?? "";
  const fitting = next?.filter((n) => n.piece.startsWith(trailing) && n.piece.length > trailing.length).slice(0, 5) ?? null;
  const rest = (piece: string) => piece.slice(trailing.length);
  const top = fitting?.[0];
  const writing = !!writer?.writing;
  // Not over the placeholder: an empty box shows that instead.
  const ghost = value && !disabled && !writing && !stale && caretAtEnd && top && !tokens.cut ? rest(top.piece) : null;

  const trackCaret = () => {
    const el = input.current;
    if (el) setCaretAtEnd(el.selectionStart === el.value.length && el.selectionEnd === el.value.length);
  };
  // Ready to type as soon as the page opens, with the cursor at the end.
  useEffect(() => {
    const el = input.current;
    if (!autoFocus || !el || el.disabled) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
  }, []);

  useEffect(() => {
    const on = () => document.activeElement === input.current && trackCaret();
    document.addEventListener("selectionchange", on);
    return () => document.removeEventListener("selectionchange", on);
  }, []);

  // Grow the box with its text (the grey suggestion included), and keep the
  // suggestion lined up if the box ever scrolls.
  useLayoutEffect(() => {
    const el = input.current;
    const m = mirror.current;
    if (!el || !m) return;
    el.style.height = "auto";
    el.style.height = `${Math.max(el.scrollHeight, m.scrollHeight)}px`;
    m.scrollTop = el.scrollTop;
  });

  const accept = (piece: string) => {
    onChange(value + piece);
    setCaretAtEnd(true);
    requestAnimationFrame(() => {
      const el = input.current;
      if (el) {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      }
    });
  };

  return (
    <section className="promptbox" aria-label="Your prompt">
      <div className="promptbox-input">
        <div className="promptbox-field">
          <div className="promptbox-mirror" ref={mirror} aria-hidden="true">
            <span className="mirror-typed">{value}</span>
            {ghost && <span className="mirror-ghost">{ghostText(ghost)}</span>}
          </div>
          <textarea ref={input} value={value} disabled={disabled} spellCheck={false} maxLength={400} rows={1} aria-label="Prompt" placeholder="Type anything…"
            onChange={(e) => { onChange(e.target.value); trackCaret(); }}
            onSelect={trackCaret} onKeyUp={trackCaret} onClick={trackCaret}
            onScroll={() => mirror.current && input.current && (mirror.current.scrollTop = input.current.scrollTop)}
            onKeyDown={(e) => {
              const el = e.currentTarget;
              const atEnd = el.selectionStart === el.value.length && el.selectionEnd === el.value.length;
              if (e.key === "Tab" && !e.shiftKey && ghost && atEnd) {
                e.preventDefault();
                accept(ghost);
              }
            }} />
        </div>
        {ghost && <kbd className="tab-hint" title={GLOSSARY.tab}>Tab</kbd>}
        {onShuffle && !disabled && (
          <button className="shuffle" onClick={() => { onShuffle(); requestAnimationFrame(() => input.current?.focus()); }} aria-label={GLOSSARY.shuffle} title={GLOSSARY.shuffle}>
            <svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true" className="icon"><path d="M3 6h3.5c4 0 5 8 9 8H17M3 14h3.5c1.4 0 2.4-1 3.3-2.3M13.4 6.6C14 6.2 14.6 6 15.5 6H17M15 3.5 17.5 6 15 8.5M15 11.5l2.5 2.5-2.5 2.5" /></svg>
          </button>
        )}
        {disabled && disabledNote && <span className="promptbox-note">{disabledNote}</span>}
      </div>
      <div className="promptbox-tokens" aria-label={`${pieces.length} tokens`}>
        <span className="row-label"><Tip tip={GLOSSARY.token} align="start">{pieces.length} token{pieces.length === 1 ? "" : "s"}</Tip></span>
        <span className="token-strip">
          {pieces.length === 0 && <span className="muted">The pieces GPT-2 reads will appear here.</span>}
          {pieces.map((p, i) => (
            <span key={i} className={`tk tk${i % 4}${onTokenHover ? " tk-hover" : ""}${activeToken === i + 1 ? " tk-active" : ""}`}
              title={`Token ${p.id}${p.text.includes("�") ? " (part of a character that spans several tokens)" : ""}`}
              tabIndex={onTokenHover ? 0 : undefined}
              onMouseEnter={onTokenHover && (() => onTokenHover(i + 1))} onMouseLeave={onTokenHover && (() => onTokenHover(null))}
              onFocus={onTokenHover && (() => onTokenHover(i + 1))} onBlur={onTokenHover && (() => onTokenHover(null))}>
              {show(p.text)}
            </span>
          ))}
          {trailing && <span className="tk tk-pending" title="A space at the end waits for the next word: GPT-2 reads spaces as the start of the following token.">{show(trailing)}</span>}
          {tokens.cut && <span className="tk-cut">cut at {MAX_PROMPT_TOKENS - 1}</span>}
        </span>
      </div>
      <div className={stale ? "promptbox-next stale" : "promptbox-next"}>
        <span className="row-label"><Tip tip={GLOSSARY.next} align="start">Next token</Tip></span>
        <span className="next-strip">
          {!next && <span className="next-note">{nextNote}</span>}
          {next && fitting?.length === 0 && <span className="next-note">GPT-2's top guesses don't start with a space here. Delete the space to see them.</span>}
          {fitting?.map((n, i) => (
            <button key={i} className="next-chip" onClick={() => accept(rest(n.piece))} title={`Add “${show(n.piece)}” to the prompt`} disabled={disabled}>
              <span className="next-piece">{show(n.piece)}</span>
              <span className="next-p">{pct(n.p)}</span>
              <span className="next-bar" style={{ width: `${Math.max(2, n.p * 100)}%` }} />
            </button>
          ))}
        </span>
        {writer && !disabled && (
          <span className="writer">
            <span className="wildness" role="radiogroup" aria-label="How adventurous GPT-2's writing is">
              {WILDNESS.map((w) => (
                <button key={w.value} role="radio" aria-checked={writer.wildness === w.value} className={writer.wildness === w.value ? "wild-on" : ""}
                  onClick={() => writer.setWildness(w.value)} title={w.tip}>{w.label}</button>
              ))}
            </span>
            <button className={writing ? "write-button writing" : "write-button"} onClick={writer.toggle} disabled={tokens.cut}
              title={writing ? "Stop writing" : "Let GPT-2 keep writing, one token at a time"}>
              {writing ? <><span className="stop-icon" aria-hidden="true" /> Stop</> : <><span className="play-icon" aria-hidden="true" /> Write</>}
            </button>
          </span>
        )}
      </div>
    </section>
  );
}

export const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
