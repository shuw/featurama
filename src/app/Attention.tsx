// Exhibit 2: attention. Pick a head and a word, and see where it looks.

import { useCallback, useEffect, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { Beats } from "./Beats.tsx";
import { attentionRow, inductionScore, IOI_ROLES, IOI_SOURCE, JOB_TEXT, jobsOf, previousScore, type HeadJobs } from "./heads.ts";
import { tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { PromptBox } from "./PromptBox.tsx";
import { Ring } from "./Ring.tsx";
import { useRun } from "./useRun.ts";

const DEFAULT = "When Mary and John went to the store, John gave a drink to";
const PREDICT_HEAD = "9.9";
const CANDIDATES = [" Mary", " John", " store", " drink"];

const CHALLENGES = [
  {
    id: "attention.copycat",
    title: "Copycat",
    text: "Find a head where every word looks at the word just before it.",
    check: (t: Trace, H: number, l: number, h: number) => t.tokens.length >= 6 && previousScore(t, H, l, h) >= 0.8,
  },
  {
    id: "attention.dejavu",
    title: "Déjà vu",
    text: "Write something that repeats itself, then find a head that looks at what came next the first time.",
    check: (t: Trace, H: number, l: number, h: number) => inductionScore(t, H, l, h) >= 0.5,
  },
];

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  jobs: HeadJobs;
  head?: string;
  guide: boolean;
}

export function Attention({ engine, tokenizer, jobs, head: routeHead, guide }: Props) {
  const { n_layers: L, n_heads: H } = engine.config;
  const [text, setText] = useState(DEFAULT);
  const [tokens, setTokens] = useState(() => tokenize(tokenizer, DEFAULT));
  const [query, setQuery] = useState<number | null>(null); // null follows the last token
  const [predicting, setPredicting] = useState(!done("attention.predict"));
  // The predict beat is about head 9.9, so it shows that head until answered.
  const [head, setHead] = useState(predicting ? PREDICT_HEAD : routeHead ?? PREDICT_HEAD);
  const [guess, setGuess] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [, rerender] = useState(0);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { pattern: true }, lens: { positions: [t.length - 1], topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run);

  useEffect(() => {
    if (routeHead && !predicting) setHead(routeHead);
  }, [routeHead]);

  const [l, h] = head.split(".").map(Number);
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const q = query === null ? shown.length - 1 : Math.min(query, shown.length - 1);
  const pieces = shown.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));
  const row = trace ? attentionRow(trace, H, l, h, q) : null;

  useEffect(() => {
    if (!trace || !current || predicting) return;
    for (const c of CHALLENGES) if (c.check(trace, H, l, h) && !done(c.id)) {
      markDone(c.id);
      rerender((n) => n + 1);
    }
  });

  const edit = (value: string) => {
    setText(value);
    setTokens(tokenize(tokenizer, value));
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
        <p className="dek">Every word can look back at the words before it. Each of GPT-2's 144 attention heads has its own idea of where to look.</p>
      </header>

      <PromptBox value={text} onChange={edit} tokens={tokens} tokenizer={tokenizer} next={predicting || !trace ? null : predictions(trace, L, tokenizer)}
        stale={!current} disabled={predicting} disabledNote="Make your guess below first" />

      <div className={!current && trace ? "stage attn-layout stale" : "stage attn-layout"}>
        <div className="picker">
          <div className="picker-head"><span className="card-label">Head</span> <strong>{head}</strong></div>
          <div className="picker-grid" role="grid" aria-label="Choose a head">
            {Array.from({ length: H }, (_, hh) =>
              Array.from({ length: L }, (_, ll) => {
                const id = `${ll}.${hh}`;
                const marked = jobsOf(jobs, ll, hh, H).length > 0;
                return (
                  <button key={id} className={id === head ? "pick pick-on" : "pick"} onClick={() => setHead(id)} disabled={predicting}
                    aria-label={`Head ${id}`} aria-pressed={id === head} style={{ gridColumn: ll + 1, gridRow: hh + 1 }}>
                    {marked && <span className="mark-job" />}
                    {IOI_ROLES[id] && <span className="mark-ioi" />}
                  </button>
                );
              }),
            )}
          </div>
          <div className="picker-axis"><span>layer 0</span><span>11</span></div>
          <div className="legend small">
            <div><span className="mark-job static" /> job measured here</div>
            <div><span className="mark-ioi static" /> role from research</div>
          </div>
        </div>
        <div className="attn-main">
          <Arcs pieces={pieces} row={predicting ? null : row} query={q} onQuery={setQuery} locked={predicting} />
          {!predicting && trace && <Grid trace={trace} H={H} l={l} h={h} pieces={pieces} query={q} onQuery={setQuery} />}
        </div>
      </div>

      <Beats
        guide={guide && (predicting
          ? "Guess first. The candidates are all in the sentence; only one is the right answer to “gave a drink to…”."
          : "Click a word to see where it looks. Pick heads in the small grid: dotted ones have jobs measured just now in your browser; ringed ones were found in published research.")}
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
            <p className="question">Head 9.9 is in layer 9. When GPT-2 reaches the last word, “to”, which earlier word will this head look at most?</p>
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
            {verdict && text === DEFAULT && (
              <section className="card card-result">
                <div className="card-label">Your guess</div>
                <p>{verdict}</p>
                {head === PREDICT_HEAD && <p className="muted">Why not the second “John”? In the IOI paper's account, earlier heads such as 8.6 look at the repeated name and tell the name movers to skip it. Pick head 8.6 and click “to” to see where it looks.</p>}
              </section>
            )}
            <section className="card">
              <div className="card-label">What you're seeing</div>
              <p><strong>Attention</strong> is how a word gathers information from earlier words. The arcs show one head's choice for one word; the grid shows it for every word at once, each row adding up to 100%.</p>
              {ioi && <p><strong>Head {head}</strong> has a known role in sentences like the default one. {ioi} <span className="muted">({IOI_SOURCE})</span></p>}
              {measured.map((j) => <p key={j}><strong>{JOB_TEXT[j].name}.</strong> {JOB_TEXT[j].text} <span className="muted">Measured in your browser on a repeated random sequence: score {jobs.scores[j][l * H + h].toFixed(2)}.</span></p>)}
              {!ioi && !measured.length && <p>Nobody has named head {head}'s job yet. Try different prompts and see if you can tell what it's for.</p>}
              {trace && <p className="muted">On this prompt it looks at the previous word {Math.round(prevHere * 100)}% of the time{indHere > 0 ? `, and at what followed an earlier copy ${Math.round(indHere * 100)}% of the time` : ""}.</p>}
              <p className="caveat">Where a head looks is only half its story: what it copies from there, and what later layers do with it, matter too.</p>
            </section>
          </>
        )}
      </Beats>
    </article>
  );
}

const CHAR_W = 7.4;

function Arcs({ pieces, row, query, onQuery, locked }: { pieces: string[]; row: number[] | null; query: number; onQuery: (q: number) => void; locked: boolean }) {
  const widths = pieces.map((p) => Math.max(26, show(p).length * CHAR_W + 14));
  const xs: number[] = [];
  widths.reduce((x, w) => (xs.push(x + w / 2), x + w + 4), 4);
  const width = xs[xs.length - 1] + widths[widths.length - 1] / 2 + 8;
  const base = 150;
  const height = base + 40;
  return (
    <div className="arcs">
      <svg width={width} height={height} role="img" aria-label={row ? `Attention from “${show(pieces[query])}”: ` + row.map((a, j) => (a >= 0.05 ? `${show(pieces[j])} ${Math.round(a * 100)}%` : "")).filter(Boolean).join(", ") : "Not shown yet"}>
        {row?.map((a, j) => {
          if (j === query || a < 0.02) return null;
          const lift = Math.min(base - 18, 24 + Math.abs(xs[query] - xs[j]) * 0.32);
          return (
            <g key={j}>
              <path d={`M${xs[query]} ${base} C ${xs[query]} ${base - lift}, ${xs[j]} ${base - lift}, ${xs[j]} ${base}`}
                className={j === 0 ? "arc arc-start" : "arc"} strokeWidth={1 + 7 * a} opacity={j === 0 ? 0.6 : 0.25 + 0.75 * a} />
              {a >= 0.08 && <text x={xs[j]} y={base - 6} textAnchor="middle" className="arc-pct">{Math.round(a * 100)}%</text>}
            </g>
          );
        })}
        {pieces.map((p, j) => (
          <g key={j} className="arc-token" role="button" tabIndex={locked ? -1 : 0} aria-label={`Look from ${show(p)}`}
            onClick={() => !locked && onQuery(j)} onKeyDown={(e) => !locked && (e.key === "Enter" || e.key === " ") && onQuery(j)}>
            <rect x={xs[j] - widths[j] / 2} y={base + 4} width={widths[j]} height={26} rx={4} className={j === query ? "tok tok-on" : "tok"} />
            <text x={xs[j]} y={base + 21} textAnchor="middle" className={j === query ? "tok-text tok-text-on" : "tok-text"}>{show(p)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function Grid({ trace, H, l, h, pieces, query, onQuery }: { trace: Trace; H: number; l: number; h: number; pieces: string[]; query: number; onQuery: (q: number) => void }) {
  const T = pieces.length;
  const cell = Math.max(10, Math.min(24, 420 / T));
  const labelW = 96;
  const top = 72;
  const width = labelW + T * cell + 4;
  const height = top + T * cell + 4;
  return (
    <div className="grid-wrap">
      <svg width={width} height={height} role="img" aria-label={`Attention grid for head ${l}.${h}`}>
        <defs>
          <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className="hatch" />
          </pattern>
        </defs>
        {pieces.map((p, j) => (
          <text key={j} transform={`translate(${labelW + j * cell + cell / 2 + 3} ${top - 6}) rotate(-60)`} className="grid-label">{show(p).slice(0, 10)}</text>
        ))}
        {pieces.map((p, i) => (
          <g key={i} onClick={() => onQuery(i)} className="grid-row">
            <text x={labelW - 6} y={top + i * cell + cell / 2 + 4} textAnchor="end" className={i === query ? "grid-label grid-label-on" : "grid-label"}>{show(p).slice(0, 12)}</text>
            {pieces.map((_, j) => {
              const a = j <= i ? trace.pattern![((l * H + h) * T + i) * T + j] : 0;
              return j <= i ? (
                <rect key={j} x={labelW + j * cell} y={top + i * cell} width={cell - 1} height={cell - 1} className="grid-cell" style={{ fillOpacity: 0.04 + 0.96 * a }}>
                  <title>{`${show(pieces[i])} → ${show(pieces[j])}: ${Math.round(a * 100)}%`}</title>
                </rect>
              ) : (
                <rect key={j} x={labelW + j * cell} y={top + i * cell} width={cell - 1} height={cell - 1} fill="url(#hatch)" />
              );
            })}
          </g>
        ))}
        <rect x={labelW - 1} y={top + query * cell - 1} width={T * cell + 1} height={cell + 1} className="grid-sel" />
      </svg>
    </div>
  );
}
