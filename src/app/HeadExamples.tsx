// Real passages where a head does the most work, with what it looked at
// highlighted, and ways to try the head on your own text.

import { useEffect, useState } from "react";
import { show } from "./analysis.ts";
import { done, markDone } from "./progress.ts";
import { href } from "./route.ts";
import { TRY_PROMPTS } from "./research.ts";

interface Example {
  pieces: (string | null)[]; // null marks a gap in the text
  attn: (number | null)[];
  start: number;
  next: string | null;
}

let examples: Promise<Record<string, Example[]>> | null = null;
function loadExamples() {
  examples ??= fetch(new URL("head-examples.json", location.href))
    .then((r) => r.json())
    .then((d) => d.heads);
  return examples;
}

// Open a head in the Attention exhibit with a prompt, skipping its opening question.
export function tryHead(id: string, text: string, setPrompt: (t: string) => void) {
  if (!done("attention.predict")) markDone("attention.predict");
  setPrompt(text);
  location.hash = href({ view: "attention", head: id });
}

export function HeadExamples({ id, setPrompt, limit = 3 }: { id: string; setPrompt: (t: string) => void; limit?: number }) {
  const [list, setList] = useState<Example[] | null>(null);
  const [all, setAll] = useState(false);
  useEffect(() => {
    let live = true;
    setList(null);
    loadExamples().then((d) => live && setList(d[id] ?? []));
    return () => {
      live = false;
    };
  }, [id]);

  const curated = TRY_PROMPTS[id];
  const shown = list ? (all ? list : list.slice(0, limit)) : [];
  return (
    <div className="examples">
      <div className="card-label">Where it works hardest on Wikipedia</div>
      {!list && <p className="muted small">Loading examples…</p>}
      {shown.map((e, i) => (
        <div key={i} className="example">
          <p className="example-text">
            {e.pieces.map((p, j) =>
              p === null ? (
                <span key={j} className="ex-gap"> … </span>
              ) : (
                <span key={j} className={j === e.pieces.length - 1 ? "ex-tok ex-here" : "ex-tok"}
                  style={(e.attn[j] ?? 0) >= 0.05 ? { background: `rgba(var(--attn-rgb), ${(0.15 + 0.85 * (e.attn[j] ?? 0)).toFixed(2)})` } : undefined}
                  title={j === e.pieces.length - 1 ? "The head is working here" : `${Math.round((e.attn[j] ?? 0) * 100)}% of its attention`}>{p}</span>
              ),
            )}
            {e.next && <span className="ex-next"> → next: <span className="mono">{show(e.next)}</span></span>}
          </p>
          <button className="link-small" onClick={() => tryHead(id, windowText(e), setPrompt)}>Try this passage</button>
        </div>
      ))}
      {list && list.length > limit && <button className="link-small" onClick={() => setAll(!all)}>{all ? "Show fewer" : `Show all ${list.length}`}</button>}
      <p className="muted small">Orange shows where the head looks from the outlined word; darker means more attention.</p>
      {curated && <button className="secondary try-curated" onClick={() => tryHead(id, curated, setPrompt)}>Try “{curated}”</button>}
    </div>
  );
}

// The passage as a prompt: the text after any gap, up to and including the head's word.
function windowText(e: Example) {
  const gap = e.pieces.lastIndexOf(null);
  return e.pieces.slice(gap + 1).join("").trimStart();
}
