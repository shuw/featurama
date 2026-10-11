// Exhibit 2: attention. Pick a head and a word, and see where it looks.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { Beats } from "./Beats.tsx";
import { attentionRow, combinedRow, headActivity, inductionScore, IOI_ROLES, IOI_SOURCE, JOB_TEXT, jobsOf, previousScore, type HeadJobs } from "./heads.ts";
import { tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { href, promptFromUrl } from "./route.ts";
import { HighlightChips, highlighted } from "./highlights.tsx";
import { PromptBox } from "./PromptBox.tsx";
import type { Writer } from "./useWriter.ts";
import { Ring } from "./Ring.tsx";
import { GLOSSARY } from "./glossary.ts";
import { Term, Tip } from "./Tip.tsx";
import { ATLAS_SOURCE, habits } from "./atlas.ts";
import { HeadExamples } from "./HeadExamples.tsx";
import { label } from "./research.ts";
import { ResearchNotes } from "./ResearchNotes.tsx";
import { useRun } from "./useRun.ts";

// The predict beat asks about this sentence; afterwards the page uses the shared prompt.
const PREDICT_PROMPT = "When Mary and John went to the store, John gave a drink to";
const PREDICT_HEAD = "9.9";
const CANDIDATES = [" Mary", " John", " store", " drink"];

const CHALLENGES = [
  {
    id: "attention.copycat",
    title: "Copycat",
    text: "Find a head where every token looks at the one just before it.",
    check: (t: Trace, H: number, l: number, h: number) => t.tokens.length >= 6 && previousScore(t, H, l, h) >= 0.8,
  },
  {
    id: "attention.dejavu",
    title: "Déjà vu",
    text: "Write something that repeats itself, then find a head that, the second time round, looks at what came next the first time.",
    check: (t: Trace, H: number, l: number, h: number) => inductionScore(t, H, l, h) >= 0.5,
  },
];

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  jobs: HeadJobs;
  head?: string;
  guide: boolean;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function Attention({ engine, tokenizer, weights, jobs, head: routeHead, guide, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, n_heads: H } = engine.config;
  const [override, setOverride] = useState<string | null>(() => (done("attention.predict") || promptFromUrl() !== null ? null : PREDICT_PROMPT));
  const text = override ?? prompt;
  const tokens = useMemo(() => tokenize(tokenizer, text), [tokenizer, text]);
  const [query, setQuery] = useState<number | null>(null); // null follows the last token
  const [predicting, setPredicting] = useState(() => !done("attention.predict") && promptFromUrl() === null); // a shared link skips the opening question
  // The predict beat is about head 9.9, so it shows that head until answered.
  const [head, setHead] = useState(predicting ? PREDICT_HEAD : routeHead ?? PREDICT_HEAD);
  const [guess, setGuess] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [, rerender] = useState(0);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { pattern: true, z: true }, lens: { positions: [t.length - 1], topK: 10 } }), [engine]);
  const { result: trace } = useRun(tokens, run);

  useEffect(() => {
    if (routeHead && !predicting) setHead(routeHead);
  }, [routeHead]);

  const [l, h] = head.split(".").map(Number);
  const [all, setAll] = useState(false);
  const [hoverHead, setHoverHead] = useState<string | null>(null);
  const [lit, setLit] = useState<string | null>(null);
  const litHeads = useMemo(() => highlighted(lit, jobs, L, H), [lit, jobs, L, H]);
  // Step to a neighbouring head, staying inside the 12 x 12 grid.
  const move = (dl: number, dh: number) => {
    setAll(false);
    const id = `${Math.min(L - 1, Math.max(0, l + dl))}.${Math.min(H - 1, Math.max(0, h + dh))}`;
    setHead(id);
    return id;
  };
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const q = query === null ? shown.length - 1 : Math.min(query, shown.length - 1);
  const pieces = shown.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));
  // How much each head adds at each token: shades the picker and weights the all-heads view.
  const activityAt = useMemo(() => {
    const cache = new Map<number, Float32Array>();
    return (j: number) => {
      if (!trace?.z || j >= trace.tokens.length) return null;
      if (!cache.has(j)) cache.set(j, headActivity(trace, weights, j));
      return cache.get(j)!;
    };
  }, [trace, weights]);
  const rowAt = trace ? (j: number) => (all ? combinedRow(trace, H, L, j, activityAt(j)!) : attentionRow(trace, H, l, h, j)) : null;
  const row = rowAt ? rowAt(q) : null;
  const activity = activityAt(q);
  const activityMax = activity ? Math.max(...activity) : 1;

  useEffect(() => {
    if (!trace || !current || predicting) return;
    for (const c of CHALLENGES) if (c.check(trace, H, l, h) && !done(c.id)) {
      markDone(c.id);
      rerender((n) => n + 1);
    }
  });

  const edit = (value: string) => {
    setOverride(null);
    setPrompt(value);
    setQuery(null);
  };

  const reveal = () => {
    if (!trace || !guess) return;
    const r = attentionRow(trace, H, 9, 9, trace.tokens.length - 1);
    const byWord = new Map<string, number>();
    r.forEach((a, j) => byWord.set(pieces[j], (byWord.get(pieces[j]) ?? 0) + a));
    const ranked = [...byWord.entries()].sort((a, b) => b[1] - a[1]);
    const best = ranked.filter(([p]) => CANDIDATES.includes(p))[0][0];
    const top = ranked.slice(0, 3).map(([p, a]) => `${Math.round(a * 100)}% on ${p === "⟨start⟩" ? "the start token" : `“${word(p)}”`}`);
    setVerdict(`${guess === best ? "Spot on. " : `You said “${word(guess)}”. `}Head 9.9 put ${top.join(", ")}.`);
    markDone("attention.predict");
    setPredicting(false);
  };

  const measured = jobsOf(jobs, l, h, H);
  const ioi = IOI_ROLES[head];
  const prevHere = trace ? previousScore(trace, H, l, h) : 0;
  const indHere = trace ? inductionScore(trace, H, l, h) : 0;

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Exhibit 2 · Attention</div>
        <h1>Who's looking at whom</h1>
        <p className="dek">Every token can look back at the tokens before it. Each of GPT-2's 144 <Term tip={GLOSSARY.head}>attention heads</Term> has its own idea of where to look.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={() => { setOverride(null); shuffle(); }} value={text} onChange={edit} tokens={tokens} tokenizer={tokenizer} next={predicting || !trace ? null : predictions(trace, L, tokenizer)} nextNote={predicting ? "Appears after your guess" : undefined}
        stale={!current} disabled={predicting} disabledNote="Make your guess below first" />

      <div className={!current && trace ? "stage attn-layout stale" : "stage attn-layout"}>
        <div className="picker">
          <div className="picker-head">
            <span className="card-label">Pick a head</span>
            <strong>{head}</strong>
          </div>
          <div className="stepper" role="group" aria-label="Step through heads">
            <span className="stepper-label">layer</span>
            <button className="step" onClick={() => move(-1, 0)} disabled={predicting || l === 0} aria-label="Previous layer">‹</button>
            <span className="stepper-value">{l}</span>
            <button className="step" onClick={() => move(1, 0)} disabled={predicting || l === L - 1} aria-label="Next layer">›</button>
            <span className="stepper-label">head</span>
            <button className="step" onClick={() => move(0, -1)} disabled={predicting || h === 0} aria-label="Previous head">‹</button>
            <span className="stepper-value">{h}</span>
            <button className="step" onClick={() => move(0, 1)} disabled={predicting || h === H - 1} aria-label="Next head">›</button>
          </div>
          <div className={predicting ? "picker-grid locked" : "picker-grid"} role="grid" aria-label="Choose a head: columns are layers, rows are heads"
            onKeyDown={(e) => {
              const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
              if (!d || predicting) return;
              e.preventDefault();
              const grid = e.currentTarget;
              const id = move(d[0], d[1]);
              requestAnimationFrame(() => grid.querySelector<HTMLButtonElement>(`[data-head="${id}"]`)?.focus());
            }}>
            {Array.from({ length: H }, (_, hh) =>
              Array.from({ length: L }, (_, ll) => {
                const id = `${ll}.${hh}`;
                return (
                  <button key={id} data-head={id} className={`pick${id === head && !all ? " pick-on" : ""}${lit ? (litHeads.has(id) ? " pick-lit" : " pick-dim") : ""}`} onClick={() => { setHead(id); setAll(false); }} disabled={predicting}
                    onMouseEnter={() => setHoverHead(id)} onMouseLeave={() => setHoverHead(null)} onFocus={() => setHoverHead(id)} onBlur={() => setHoverHead(null)}
                    aria-label={`Head ${id}: ${label(id) ?? habits(id)[0]}`} aria-pressed={id === head && !all} tabIndex={id === head ? 0 : -1}
                    style={{ gridColumn: ll + 1, gridRow: hh + 1, ...(activity && !predicting ? { background: `rgba(var(--attn-rgb), ${(0.05 + 0.95 * Math.sqrt(activity[ll * H + hh] / activityMax)).toFixed(2)})` } : {}) }}>
                  </button>
                );
              }),
            )}
            {hoverHead && !predicting && (() => {
            const [hl, hh] = hoverHead.split(".").map(Number);
            return (
              <div className="pick-tip" role="status" style={{ left: hl * 21 + 9, top: hh * 21 }}>
                <strong>{hoverHead}</strong> {label(hoverHead) ?? habits(hoverHead)[0]}
              </div>
            );
          })()}
          </div>
          <div className="picker-axis"><span>layer 0 →</span><span>11</span></div>
          <button className={all ? "secondary all-heads all-on" : "secondary all-heads"} aria-pressed={all} disabled={predicting} onClick={() => setAll(!all)}>
            {all ? "Showing all heads" : "All heads at once"}
          </button>
          {predicting ? (
            <p className="picker-note">Unlocks after your guess.</p>
          ) : (
            <p className="picker-note">Darker heads add more at “{word(pieces[q] ?? "")}”. Click a square or use the arrow keys; columns are layers, rows are heads.</p>
          )}
          <HighlightChips value={lit} onChange={setLit} compact />
        </div>
        <div className="attn-main">
          <Arcs pieces={pieces} row={predicting ? null : row} rowAt={predicting ? null : rowAt} all={all} query={q} onQuery={setQuery} locked={predicting} />
          {!predicting && rowAt && <Grid rowAt={rowAt} label={all ? "all heads" : `head ${head}`} pieces={pieces} query={q} onQuery={setQuery} />}
        </div>
      </div>

      <Beats
        guide={guide && (predicting
          ? "Guess first. All four words are in the sentence, but only one finishes “gave a drink to…”."
          : "Click a token to see where it looks. Pick heads in the small grid; the chips under it outline heads known to do a job.")}
        side={!predicting && (
          <section className="card">
            <div className="card-label">Challenges</div>
            {CHALLENGES.map((c) => (
              <div key={c.id} className="challenge">
                <Ring full={done(c.id)} />
                <div><strong>{c.title}.</strong> {c.text}</div>
              </div>
            ))}
          </section>
        )}
      >
        {predicting ? (
          <section className="card card-predict">
            <div className="card-label">Predict</div>
            <p className="question">Head 9.9 sits in layer 9. When GPT-2 reaches the final “to”, which earlier word will this head look at most?</p>
            <div className="chips chips-words" role="group" aria-label="Your guess">
              {CANDIDATES.map((c) => (
                <button key={c} className={guess === c ? "chip chip-on" : "chip"} onClick={() => setGuess(c)} aria-pressed={guess === c}>{word(c)}</button>
              ))}
            </div>
            <div className="card-actions">
              <span className="hint">{guess ? `Your guess: “${word(guess)}”.` : "Pick a word."}</span>
              <button className="primary" disabled={!guess || !trace} onClick={reveal}>Show me</button>
            </div>
          </section>
        ) : (
          <>
            {verdict && text === PREDICT_PROMPT && (
              <section className="card card-result">
                <div className="card-label">Your guess</div>
                <p>{verdict}</p>
                {head === PREDICT_HEAD && <p className="muted">Why not the second “John”? According to the researchers who mapped this sentence, earlier heads such as 8.6 notice the repeated name and tell the name movers to skip it. Pick head 8.6 and click “to” to see where it looks.</p>}
              </section>
            )}
            {all ? (
            <section className="card">
              <div className="card-label">All heads at once</div>
              <p>The arrows show where “{word(pieces[q] ?? "")}” pulls information from across all 144 heads. Each head's attention counts in proportion to how much that head adds at this token, and attention resting on the start token is left out.</p>
              <p className="caveat">This shows where heads looked, not whether it changed the prediction. To test that, swap tokens on the Patching page.</p>
            </section>
            ) : (<>
            <section className="card">
              <div className="card-label">What you're seeing</div>
              <p><strong>Attention</strong> is how a token gathers information from earlier tokens. The arcs show where one head looks from one token. <Term tip={GLOSSARY.attentionRow}>The grid</Term> shows every token at once.</p>
              <ResearchNotes id={head} />
              {measured.map((j) => <p key={j}><span className="badge">{JOB_TEXT[j].letter}</span> <strong>{JOB_TEXT[j].name}.</strong> {JOB_TEXT[j].text} <Term tip={GLOSSARY.measured}>Measured in your browser</Term>: <Term tip={GLOSSARY.score}>score {jobs.scores[j][l * H + h].toFixed(2)}</Term>.</p>)}
              <div className="habits">
                <Term tip={ATLAS_SOURCE}><span className="card-label">Head {head}, measured on Wikipedia text</span></Term>
                <ul>{habits(head).map((x) => <li key={x}>{x}</li>)}</ul>
              </div>
              {trace && <p className="muted">On your prompt, {Math.round(prevHere * 100)}% of its attention goes to the token just before{indHere > 0 ? `, and ${Math.round(indHere * 100)}% to whatever followed an earlier copy` : ""}.</p>}
              <p className="caveat">Where a head looks is only half its story. What it copies from there, and what later layers do with it, matter too.</p>
              <a className="link-small" href={href({ view: "zoom", part: "head", arg: head })}>Look inside head {head}: what it asks, fetches and writes →</a>
            </section>
            <section className="card">
              <HeadExamples id={head} setPrompt={edit} />
            </section>
            </>)}
          </>
        )}
      </Beats>
    </article>
  );
}

const CHAR_W = 7.4;

function Arcs({ pieces, row, rowAt, all = false, query, onQuery, locked }: {
  pieces: string[]; row: number[] | null; rowAt: ((q: number) => number[]) | null; all?: boolean; query: number; onQuery: (q: number) => void; locked: boolean;
}) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [peek, setPeek] = useState<number | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const widths = pieces.map((p) => Math.max(26, show(p).length * CHAR_W + 14));
  const xs: number[] = [];
  widths.reduce((x, w) => (xs.push(x + w / 2), x + w + 4), 4);
  const width = xs[xs.length - 1] + widths[widths.length - 1] / 2 + 8;
  const base = 150;
  const height = base + 40;
  const q = word(pieces[query]);
  const name = (j: number) => (j === 0 ? "the start token" : `“${word(pieces[j])}”`);
  const arcTip = (j: number, a: number) =>
    (all ? `Across all heads, ${Math.round(a * 100)}% of what “${q}” pulls in comes from ${name(j)}.` : `“${q}” gives ${Math.round(a * 100)}% of this head's attention to ${name(j)}.`) +
    (j === 0 ? " Attention on the start token usually means the head is resting: it has nothing useful to fetch here." : " Thicker arcs mean more attention.");
  const tokenTip = (j: number) =>
    (j === 0 ? "⟨start⟩ is GPT-2's start token, put in front of every prompt. " : `Token ${j}: “${show(pieces[j])}”. `) +
    (j === query ? "The arrows show where this token looks." : "Its faint arcs preview where it looks; click to select it.");
  // The tooltip sits outside the scrolling box so it is never clipped.
  const show_ = (x: number, y: number, text: string) => {
    const el = scroller.current;
    setTip({ x: x - (el?.scrollLeft ?? 0) + (el?.offsetLeft ?? 0), y: y + (el?.offsetTop ?? 0), text });
  };
  return (
    <div className="arcs" onMouseLeave={() => setTip(null)}>
      {row && (
        <p className="arcs-caption">
          {all ? <>Arrows point from “{q}” to where it pulls information from, across all heads; thicker means more. Attention resting on the start token is left out.</> : <>Arrows point from “{q}” to the tokens it looks at; thicker means more attention. The grey dashed arrow is attention resting on the <Term tip={GLOSSARY.start}>start token</Term>, where a head parks it when it has nothing useful to fetch. Point at another token to preview where it looks.</>}
        </p>
      )}
      <div className="arcs-scroll" ref={scroller}>
        <svg width={width} height={height} role="img" data-arcs aria-label={row ? `Attention from “${show(pieces[query])}”: ` + row.map((a, j) => (a >= 0.05 ? `${show(pieces[j])} ${Math.round(a * 100)}%` : "")).filter(Boolean).join(", ") : "Not shown yet"}>
          <defs>
            <marker id="looks-at" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M0 0 L10 5 L0 10 Z" className="arrow-head" />
            </marker>
            <marker id="rests-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M0 0 L10 5 L0 10 Z" className="arrow-head-rest" />
            </marker>
          </defs>
          {peek !== null && rowAt && rowAt(peek).map((a, j) => {
            if (j === peek || a < 0.05) return null;
            const lift = Math.min(base - 18, 24 + Math.abs(xs[peek] - xs[j]) * 0.32);
            const d = `M${xs[peek]} ${base} C ${xs[peek]} ${base - lift}, ${xs[j]} ${base - lift}, ${xs[j]} ${base}`;
            return <path key={`p${j}`} d={d} className={j === 0 ? "arc arc-start arc-peek" : "arc arc-peek"} strokeWidth={j === 0 ? 1.2 : 1 + 5 * a} />;
          })}
          <defs>
          </defs>
          {row?.map((a, j) => {
            if (j === query || a < (all ? 0.05 : 0.02)) return null;
            const lift = Math.min(base - 18, 24 + Math.abs(xs[query] - xs[j]) * 0.32);
            const d = `M${xs[query]} ${base} C ${xs[query]} ${base - lift}, ${xs[j]} ${base - lift}, ${xs[j]} ${base}`;
            const enter = () => show_((xs[query] + xs[j]) / 2, base - lift * 0.75, arcTip(j, a));
            return (
              <g key={j} className="arc-group" onMouseEnter={enter} onMouseLeave={() => setTip(null)}>
                {/* Resting attention is drawn thin whatever its share, so idle heads don't dominate. */}
                {j === 0
                  ? <path d={d} className="arc arc-start" strokeWidth={1.5} markerEnd="url(#rests-on)" />
                  : <path d={d} className="arc" strokeWidth={1 + 7 * a} opacity={0.25 + 0.75 * a} markerEnd="url(#looks-at)" />}
                <path d={d} className="arc-hit" />
                {a >= 0.08 && <text x={xs[j] + 9} y={base - 10} className={j === 0 ? "arc-pct arc-pct-rest" : "arc-pct"}>{j === 0 ? `resting ${Math.round(a * 100)}%` : `${Math.round(a * 100)}%`}</text>}
              </g>
            );
          })}
          {pieces.map((p, j) => (
            <g key={j} className="arc-token" role="button" tabIndex={locked ? -1 : 0} aria-label={tokenTip(j)}
              onMouseEnter={() => { if (j === query || j === 0) show_(xs[j], base + 2, tokenTip(j)); setPeek(j === query ? null : j); }} onMouseLeave={() => { setTip(null); setPeek(null); }}
              onFocus={() => { show_(xs[j], base + 2, tokenTip(j)); setPeek(j === query ? null : j); }} onBlur={() => { setTip(null); setPeek(null); }}
              onClick={() => { if (!locked) { onQuery(j); setPeek(null); } }} onKeyDown={(e) => !locked && (e.key === "Enter" || e.key === " ") && onQuery(j)}>
              <rect x={xs[j] - widths[j] / 2} y={base + 4} width={widths[j]} height={26} rx={4} className={j === query ? "tok tok-on" : "tok"} />
              <text x={xs[j]} y={base + 21} textAnchor="middle" className={j === query ? "tok-text tok-text-on" : "tok-text"}>{show(p)}</text>
            </g>
          ))}
        </svg>
      </div>
      {tip && <div className="svg-tip" role="status" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  );
}

function Grid({ rowAt, label, pieces, query, onQuery }: { rowAt: (q: number) => number[]; label: string; pieces: string[]; query: number; onQuery: (q: number) => void }) {
  const T = pieces.length;
  const cell = Math.max(10, Math.min(24, 420 / T));
  const labelW = 96;
  const top = 72;
  const width = labelW + T * cell + 4;
  const height = top + T * cell + 4;
  return (
    <div className="grid-wrap">
      <svg width={width} height={height} role="img" aria-label={`Attention grid for ${label}`}>
        <defs>
          <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className="hatch" />
          </pattern>
        </defs>
        {pieces.map((p, j) => (
          <text key={j} transform={`translate(${labelW + j * cell + cell / 2 + 3} ${top - 6}) rotate(-60)`} className="grid-label">{show(p).slice(0, 10)}</text>
        ))}
        {pieces.map((p, i) => { const r = rowAt(i); return (
          <g key={i} onClick={() => onQuery(i)} className="grid-row">
            <text x={labelW - 6} y={top + i * cell + cell / 2 + 4} textAnchor="end" className={i === query ? "grid-label grid-label-on" : "grid-label"}>{show(p).slice(0, 12)}</text>
            {pieces.map((_, j) => {
              const a = j <= i ? r[j] : 0;
              return j <= i ? (
                <rect key={j} x={labelW + j * cell} y={top + i * cell} width={cell - 1} height={cell - 1} className="grid-cell" style={{ fillOpacity: 0.04 + 0.96 * a }}>
                  <title>{`${show(pieces[i])} → ${show(pieces[j])}: ${Math.round(a * 100)}%`}</title>
                </rect>
              ) : (
                <rect key={j} x={labelW + j * cell} y={top + i * cell} width={cell - 1} height={cell - 1} fill="url(#hatch)" />
              );
            })}
          </g>
        ); })}
        <rect x={labelW - 1} y={top + query * cell - 1} width={T * cell + 1} height={cell + 1} className="grid-sel" />
      </svg>
    </div>
  );
}
