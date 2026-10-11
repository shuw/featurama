// Exhibit 4: activation patching. Copy one token's running summary at one
// layer from sentence A into sentence B, and see whether B's answer follows.

import { useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show, word } from "./analysis.ts";
import { Beats } from "./Beats.tsx";
import { GLOSSARY } from "./glossary.ts";
import { tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { pct } from "./PromptBox.tsx";
import { Ring } from "./Ring.tsx";
import { Term } from "./Tip.tsx";

const PAIR = {
  a: "When Mary and John went to the store, John gave a drink to",
  b: "When Mary and John went to the store, Mary gave a drink to",
};

interface Side {
  tokens: number[];
  trace: Trace;
  answer: number; // this sentence's own top guess
}

interface Probe {
  pa: number; // probability of A's answer in (patched) B
  pb: number; // probability of B's answer
  flip: number; // 0 = B's answer unchanged, 1 = as sure of A's answer as A itself
}

export function Patching({ engine, tokenizer, guide }: { engine: GpuEngine; tokenizer: Tokenizer; guide: boolean }) {
  const { n_layers: L, d_model: d } = engine.config;
  const [textA, setTextA] = useState(PAIR.a);
  const [textB, setTextB] = useState(PAIR.b);
  const tokA = useMemo(() => tokenize(tokenizer, textA), [tokenizer, textA]);
  const tokB = useMemo(() => tokenize(tokenizer, textB), [tokenizer, textB]);
  const [a, setA] = useState<Side | null>(null);
  const [b, setB] = useState<Side | null>(null);
  const [cells, setCells] = useState<(number | null)[]>([]);
  const [probe, setProbe] = useState<{ l: number; p: number; result: Probe } | null>(null);
  const [pinned, setPinned] = useState<{ l: number; p: number } | null>(null);
  const [sweeping, setSweeping] = useState(false);
  const [predicting, setPredicting] = useState(!done("patching.predict"));
  const [guess, setGuess] = useState<string | null>(null);
  const [, rerender] = useState(0);
  const sweepId = useRef(0);
  const ends = useRef<{ clean: Float32Array; corrupt: Float32Array } | null>(null);

  const T = tokB.length;
  const sameLength = tokA.length === tokB.length;
  const problem = !sameLength
    ? `The sentences must have the same number of tokens (A has ${tokA.length - 1}, B has ${tokB.length - 1}).`
    : a && b && a.answer === b.answer
      ? `Both sentences end with the same guess, “${word(tokenizer.piece(a.answer))}”, so there's nothing to flip. Change one so they disagree.`
      : null;

  // Run both sentences whenever either changes.
  useEffect(() => {
    let live = true;
    sweepId.current++;
    setSweeping(false);
    setCells([]);
    setProbe(null);
    setPinned(null);
    const handle = setTimeout(async () => {
      const run = async (tokens: number[]) => {
        const trace = await engine.run({ tokens, read: { resid: true }, lens: { positions: [tokens.length - 1], topK: 1 } });
        return { tokens, trace, answer: trace.lens!.ids[L] };
      };
      const [ra, rb] = [await run(tokA), await run(tokB)];
      if (live) {
        ends.current = null;
        setA(ra);
        setB(rb);
      }
    }, 200);
    return () => {
      live = false;
      clearTimeout(handle);
    };
  }, [engine, tokA, tokB, L]);

  const ready = a && b && a.tokens === tokA && b.tokens === tokB && !problem;

  // How sure A and B are of the two answers: the ends of the flip scale.
  const answerOdds = async (tokens: number[], interventions?: Parameters<GpuEngine["run"]>[0]["interventions"]) => {
    const t = await engine.run({ tokens, interventions, lens: { positions: [T - 1], topK: 1, targets: [a!.answer, b!.answer] } });
    return t.lens!.targetLogprobs.slice(L * 2, L * 2 + 2);
  };

  // B with one (layer, position) of its residual stream replaced by A's.
  const patch = async (l: number, p: number): Promise<Probe> => {
    ends.current ??= { clean: await answerOdds(tokA), corrupt: await answerOdds(tokB) };
    const value = a!.trace.resid!.slice((l * T + p) * d, (l * T + p + 1) * d);
    const lp = await answerOdds(tokB, [{ kind: "resid", hook: "resid_pre", layer: l, position: p, mode: "set", value }]);
    return { pa: Math.exp(lp[0]), pb: Math.exp(lp[1]), flip: flipOf(lp, ends.current.clean, ends.current.corrupt) };
  };

  const look = async (l: number, p: number) => {
    if (!ready || predicting) return;
    const result = await patch(l, p);
    setProbe({ l, p, result });
  };

  const sweep = async () => {
    if (!ready) return;
    const id = ++sweepId.current;
    setSweeping(true);
    const next: (number | null)[] = Array(L * T).fill(null);
    setCells(next.slice());
    // Column by column, so the map fills in the order the model runs.
    for (let l = 0; l < L; l++) {
      for (let p = 0; p < T; p++) {
        const r = await patch(l, p);
        if (id !== sweepId.current) return;
        next[l * T + p] = r.flip;
      }
      setCells(next.slice());
    }
    setSweeping(false);
    if (!done("patching.sweep")) {
      markDone("patching.sweep");
      rerender((n) => n + 1);
    }
  };

  const reveal = () => {
    markDone("patching.predict");
    setPredicting(false);
    sweep();
  };

  useEffect(() => {
    if (probe && probe.result.flip >= 0.9 && probe.l >= 6 && !done("patching.one")) {
      markDone("patching.one");
      rerender((n) => n + 1);
    }
  }, [probe]);

  const pieces = tokB.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));
  const differs = (p: number) => tokA[p] !== tokB[p];
  const shown = probe ?? null;
  const ansA = a ? word(tokenizer.piece(a.answer)) : "…";
  const ansB = b ? word(tokenizer.piece(b.answer)) : "…";
  const summary = cells.length && !sweeping ? summarize(cells, L, T, differs) : null;

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Exhibit 4 · Patching</div>
        <h1>The Swap-o-matic</h1>
        <p className="dek">Two sentences, one word apart, two different answers. Copy one token's running summary from A into B and see whether B's answer follows. This is how researchers find where a model keeps the information it needs.</p>
      </header>

      <div className="pair">
        <SentenceCard label="A" text={textA} setText={setTextA} tokens={tokA} tokenizer={tokenizer} side={a} other={tokB} locked={predicting} />
        <SentenceCard label="B" text={textB} setText={setTextB} tokens={tokB} tokenizer={tokenizer} side={b} other={tokA} locked={predicting} />
      </div>
      {problem && <p className="error">{problem}</p>}

      <div className="stage patch-stage">
        <div className="patch-grid-wrap">
          <div className="patch-title">
            <span className="card-label">Copy from A into B at…</span>
            <button className="secondary" onClick={sweep} disabled={!ready || sweeping || predicting}>{sweeping ? "Filling in…" : cells.length ? "Fill again" : "Fill every square"}</button>
          </div>
          <div className="patch-grid" style={{ gridTemplateColumns: `112px repeat(${L}, minmax(0, 1fr))` }} onMouseLeave={() => !pinned && setProbe(null)}>
            <div className="patch-corner"><Term tip={GLOSSARY.layer} align="start">layer</Term></div>
            {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
            {pieces.map((piece, p) => (
              <PatchRow key={p} label={show(piece)} highlight={differs(p)}>
                {Array.from({ length: L }, (_, l) => {
                  const v = cells[l * T + p];
                  const isProbe = shown && shown.l === l && shown.p === p;
                  return (
                    <button key={l} className={`patch-cell${isProbe ? " patch-on" : ""}${pinned && pinned.l === l && pinned.p === p ? " patch-pinned" : ""}`}
                      style={v != null ? { background: `rgba(var(--resid-rgb), ${(0.06 + 0.94 * Math.max(0, Math.min(1, v))).toFixed(2)})` } : undefined}
                      onMouseEnter={() => !pinned && look(l, p)} onFocus={() => look(l, p)}
                      onClick={() => { setPinned(pinned && pinned.l === l && pinned.p === p ? null : { l, p }); look(l, p); }}
                      disabled={!ready || predicting} aria-label={`Copy “${show(piece)}” at layer ${l} from A into B${v != null ? `: flips ${Math.round(v * 100)}%` : ""}`}>
                      {v != null && v >= 0.5 && <span className="patch-num">{Math.round(v * 100)}</span>}
                    </button>
                  );
                })}
              </PatchRow>
            ))}
          </div>
          <div className="table-foot">
            <span>Rows: B's tokens (the one that differs is marked). Columns: the layer the copy happens before.</span>
            <span className="scale"><span>B's answer</span><span className="scale-bar scale-resid" /><span>A's answer</span></span>
          </div>
        </div>

        <aside className="patch-side">
          <div className="card">
            <div className="card-label">{shown ? `Copying “${show(pieces[shown.p])}” before layer ${shown.l}` : "Point at a square"}</div>
            {shown ? (
              <>
                <Bar label={ansA} p={shown.result.pa} kind="a" />
                <Bar label={ansB} p={shown.result.pb} kind="b" />
                <p>{verdictOf(shown.result.flip, ansA, ansB)}</p>
              </>
            ) : (
              <>
                <p className="muted">Each square copies one of A's running summaries into B, at one token and one layer. The bars show B's answer afterwards.</p>
              </>
            )}
          </div>
        </aside>
      </div>

      <Beats
        guide={guide && (predicting
          ? "Guess first. Then the map fills in by itself, one layer at a time."
          : "Point at squares to see B's answer change. Click one to hold it. Try your own pair: change one word in each sentence.")}
        side={!predicting && (
          <section className="card">
            <div className="card-label">Challenges</div>
            <div className="challenge"><Ring full={done("patching.sweep")} /><div><strong>Cartographer.</strong> Fill in the whole map.</div></div>
            <div className="challenge"><Ring full={done("patching.one")} /><div><strong>One square.</strong> Find a single square after layer 6 that flips B's answer at least 90% of the way.</div></div>
          </section>
        )}
      >
        {predicting ? (
          <section className="card card-predict">
            <div className="card-label">Predict</div>
            <p className="question">A and B differ only in the second name. To make B answer “{ansA}”, where would you have to copy from A?</p>
            <div className="chips chips-words chips-3" role="group" aria-label="Your guess">
              {GUESSES.map((g) => (
                <button key={g.id} className={guess === g.id ? "chip chip-text chip-on" : "chip chip-text"} onClick={() => setGuess(g.id)} aria-pressed={guess === g.id}>{g.label}</button>
              ))}
            </div>
            <div className="card-actions">
              <span className="hint">{guess ? "Now fill in the map and see." : "Pick one."}</span>
              <button className="primary" disabled={!guess || !ready} onClick={reveal}>Fill in the map</button>
            </div>
          </section>
        ) : (
          <>
            {guess && summary && textA === PAIR.a && textB === PAIR.b && (
              <section className="card card-result">
                <div className="card-label">Your guess</div>
                <p>{guess === summary.kind ? "Spot on. " : ""}{summary.text}</p>
              </section>
            )}
            <section className="card">
              <div className="card-label">What you're seeing</div>
              <p><strong>Activation patching</strong> swaps one piece of the model's working from one run into another. If copying the <Term tip={GLOSSARY.residual}>running summary</Term> at one token and one layer makes B give A's answer, that summary carried the information that decides the answer.</p>
              {textA === PAIR.a && textB === PAIR.b && <p>For this pair, the information sits on the second name, where the sentences differ, until about layer 9, then moves to the last token, “to”. The name movers you met in Attention live in layers 9 and 10: they are what carries it across.</p>}
              <p className="caveat">A square that does nothing doesn't prove that spot is unimportant: the information can be spread out, or kept in more than one place.</p>
            </section>
          </>
        )}
      </Beats>
    </article>
  );
}

const GUESSES = [
  { id: "name", label: "At the second name" },
  { id: "end", label: "At the last token, “to”" },
  { id: "moves", label: "At the name early on, then at “to” later" },
];

function flipOf(lp: ArrayLike<number>, clean: ArrayLike<number>, corrupt: ArrayLike<number>) {
  const diff = (x: ArrayLike<number>) => x[0] - x[1];
  const span = diff(clean) - diff(corrupt);
  return span === 0 ? 0 : (diff(lp) - diff(corrupt)) / span;
}

function verdictOf(flip: number, ansA: string, ansB: string) {
  if (flip >= 0.9) return `B now answers “${ansA}”, as A does: this square carries the deciding information.`;
  if (flip >= 0.5) return `B leans to “${ansA}” now: this square carries much of the information.`;
  if (flip >= 0.15) return `B wavers a little toward “${ansA}”.`;
  return `B still answers “${ansB}”: this square makes little difference.`;
}

// Where the copies that flip the answer sit, early and late.
function summarize(cells: (number | null)[], L: number, T: number, differs: (p: number) => boolean) {
  const best = (l: number) => {
    let bp = 0;
    for (let p = 0; p < T; p++) if ((cells[l * T + p] ?? 0) > (cells[l * T + bp] ?? 0)) bp = p;
    return bp;
  };
  const early = best(1);
  const late = best(L - 1);
  const atName = differs(early);
  const atEnd = late === T - 1;
  let switchLayer = -1;
  for (let l = 0; l < L; l++) if (best(l) === T - 1) { switchLayer = l; break; }
  const kind = atName && atEnd ? "moves" : atName ? "name" : atEnd ? "end" : "other";
  const text =
    kind === "moves"
      ? `In early layers, only copying the second name flips the answer. From about layer ${switchLayer}, copying “to” does it instead: the information has moved to the last token.`
      : kind === "name"
        ? "Copying the second name flips the answer at every layer."
        : kind === "end"
          ? "Copying the last token flips the answer at every layer."
          : "The flips are spread out on this pair.";
  return { kind, text };
}

function SentenceCard({ label, text, setText, tokens, tokenizer, side, other, locked }: {
  label: string; text: string; setText: (s: string) => void; tokens: number[]; tokenizer: Tokenizer; side: Side | null; other: number[]; locked: boolean;
}) {
  const answer = side && side.tokens === tokens ? side : null;
  // The lens's last row is the model's real output.
  const p = answer ? Math.exp(answer.trace.lens!.logprobs[answer.trace.lens!.logprobs.length - 1]) : 0;
  return (
    <section className={`card sentence sentence-${label.toLowerCase()}`}>
      <div className="sentence-head"><span className="sentence-label">{label}</span>
        {answer && <span className="sentence-answer">answers <strong>{show(tokenizer.piece(answer.answer))}</strong> {pct(p)}</span>}
      </div>
      <input className="sentence-input" value={text} onChange={(e) => setText(e.target.value)} disabled={locked} spellCheck={false} aria-label={`Sentence ${label}`} />
      <div className="token-strip small-strip">
        {tokens.slice(1).map((id, i) => (
          <span key={i} className={other[i + 1] !== id ? "tk tk-diff" : "tk tk3"}>{show(tokenizer.piece(id))}</span>
        ))}
      </div>
    </section>
  );
}

function PatchRow({ label, highlight, children }: { label: string; highlight: boolean; children: React.ReactNode }) {
  return (
    <>
      <div className={highlight ? "patch-label patch-label-diff" : "patch-label"}>{label}</div>
      {children}
    </>
  );
}

function Bar({ label, p, kind }: { label: string; p: number | null; kind: "a" | "b" }) {
  return (
    <div className={`answer-bar answer-${kind}`}>
      <span className="answer-word">{label}</span>
      <span className="answer-track"><span style={{ width: `${(p ?? 0) * 100}%` }} /></span>
      <span className="answer-p">{p == null ? "" : pct(p)}</span>
    </div>
  );
}
