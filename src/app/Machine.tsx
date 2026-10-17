// The home screen: every part of GPT-2 laid out like a periodic table,
// shaded by how much it writes at a token of your prompt, or by how it votes
// for GPT-2's answer. Click a part to open it.

import { useCallback, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { GLOSSARY } from "./glossary.ts";
import { headActivity, headOutput, IOI_ROLE, IOI_ROLES, IOI_SOURCE, JOB_TEXT, jobsOf, mlpActivity, ROLES, type HeadJobs } from "./heads.ts";
import { DirectEffect } from "./DirectEffect.tsx";
import { HighlightChips, highlighted } from "./highlights.tsx";
import { MlpCard } from "./MlpCard.tsx";
import { tokenize } from "./model.ts";
import { pct, PromptBox } from "./PromptBox.tsx";
import type { Writer } from "./useWriter.ts";
import { href, navigate, setUrlParam } from "./route.ts";
import { IOI_PROMPT, NAME_MOVERS } from "./LabUnplug.tsx";
import { votes } from "./inside.ts";
import { cssRgb, diverging, useThemeVersion } from "./inside-ui.tsx";
import { Term, Tip } from "./Tip.tsx";
import { ATLAS_SOURCE, habits } from "./atlas.ts";
import { HeadExamples } from "./HeadExamples.tsx";
import { label, research } from "./research.ts";
import { ResearchNotes } from "./ResearchNotes.tsx";
import { useRun } from "./useRun.ts";

const PLACES = [
  { name: "Heads", text: "Where each head looks, and how it decides", route: href({ view: "head", head: "9.9" }) },
  { name: "MLPs", text: "3,072 neurons a layer, and what they fire on", route: href({ view: "mlp", arg: "7" }) },
  { name: "Layers", text: "Watch the guess take shape, number by number", route: href({ view: "layers" }) },
  { name: "Features", text: "Concepts inside the numbers", route: href({ view: "features" }) },
  { name: "Weights", text: "Every weight in 3D, down to a single one", route: href({ view: "weights" }) },
  { name: "Lab", text: "Unplug, swap, steer and trace", route: href({ view: "lab" }) },
];

type Shade = "activity" | "votes";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  jobs: HeadJobs;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function Machine({ engine, tokenizer, weights, jobs, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, n_heads: H, d_model: d } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [shadeBy, setShadeBy] = useState<Shade>("activity");
  const [rival, setRival] = useState(1);
  const [lit, setLit] = useState<string | null>(null);
  const litHeads = useMemo(() => highlighted(lit, jobs, L, H), [lit, jobs, L, H]);

  // The lens at every position, so pointing at any token can show what GPT-2 expected next there.
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { z: true, resid: true, residMid: true }, lens: { positions: t.map((_, i) => i), topK: 10 } }).then((trace) => ({ trace })), [engine]);
  const { result, version } = useRun(tokens, run, 150);

  const current = !!result && result.trace.tokens === tokens;
  const shown = result?.trace.tokens ?? tokens;
  // Activity is shown at the token you point at in the prompt, or else the last one.
  const [pointed, setPointed] = useState<number | null>(null);
  const pos = pointed !== null && pointed < shown.length ? pointed : shown.length - 1;
  const activity = useMemo(
    () => (result ? { heads: headActivity(result.trace, weights, pos), mlps: mlpActivity(result.trace, d, L, pos) } : null),
    [result, pos, weights, d, L],
  );
  const lastWord = pos === 0 ? "⟨start⟩" : word(tokenizer.piece(shown[pos]));
  const expected = result && pointed !== null && pos < shown.length - 1 ? expectedAt(result.trace, L, pos, tokenizer) : null;
  const headMax = activity ? Math.max(...activity.heads) : 1;
  const mlpMax = activity ? Math.max(...activity.mlps) : 1;
  // Square root, so the busiest head doesn't wash out the rest.
  const shade = (v: number, max: number) => (0.05 + 0.95 * Math.sqrt(v / max)).toFixed(2);
  // Votes: each part's push toward GPT-2's answer over another of its guesses.
  const last = shown.length - 1;
  const guesses = result ? Array.from({ length: 5 }, (_, k) => result.trace.lens!.ids[(L * shown.length + last) * 10 + k]) : [];
  const voted = useMemo(() => (result && shadeBy === "votes" ? votes(result.trace, weights, last, guesses[0], guesses[rival]) : null), [result, shadeBy, rival, weights, last]);
  const voteOf = useMemo(() => new Map(voted?.parts.map((p) => [p.id, p.value]) ?? []), [voted]);
  const voteMax = voted ? Math.max(...voted.parts.map((p) => Math.abs(p.value))) : 1;
  const theme = useThemeVersion();
  const rgb = useMemo(() => ({ pos: cssRgb("--pos-rgb"), neg: cssRgb("--neg-rgb") }), [theme]);
  const headBg = (l: number, h: number) =>
    voted ? diverging(voteOf.get(`${l}.${h}`) ?? 0, voteMax, rgb.pos, rgb.neg) : `rgba(var(--attn-rgb), ${activity ? shade(activity.heads[l * H + h], headMax) : "0.05"})`;
  const mlpBg = (l: number) => (voted ? diverging(voteOf.get(`m${l}`) ?? 0, voteMax, rgb.pos, rgb.neg) : `rgba(var(--mlp-rgb), ${activity ? shade(activity.mlps[l], mlpMax) : "0.05"})`);

  const focus = hover ?? selected;
  const focusMlp = focus?.startsWith("m") ? Number(focus.slice(1)) : null;
  const focused = focus && focusMlp === null ? describeHead(focus, jobs, H, activity?.heads ?? null, headMax) : null;
  const headVector = useCallback(() => {
    const [hl, hh] = (focus ?? "0.0").split(".").map(Number);
    return headOutput(result!.trace, weights, hl, hh, pos);
  }, [focus, result, weights, pos]);
  const toLab = (off: string[]) => {
    setUrlParam("off", off.join(","));
    navigate(href({ view: "lab", tool: "unplug" }));
  };
  const tryNameMovers = () => {
    setPrompt(IOI_PROMPT);
    toLab(NAME_MOVERS);
  };

  return (
    <article className="page">
      <header className="page-head">
        <h1>Every part of the model, working on your prompt</h1>
        <p className="dek">GPT-2 small has 12 <Term tip={GLOSSARY.layer}>layers</Term>. Each has 12 <Term tip={GLOSSARY.head}>attention heads</Term>, which move information between tokens, and one <Term tip={GLOSSARY.mlp}>MLP</Term>, which works on each token alone. Brightness shows how much each part adds at {pointed === null ? "the last token" : "the token you're pointing at"}, “{lastWord}”{expected ? <>, where GPT-2 expected “{word(expected.piece)}” next ({pct(expected.p)})</> : null}. Point at a token in the prompt to see another; click any part to open it.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        onTokenHover={setPointed} activeToken={pointed}
        next={result ? predictions(result.trace, L, tokenizer) : null} stale={!current} />

      <div className="machine-body">
        <div className={!current && result ? "stage table-wrap stale" : "stage table-wrap"}>
          <div className="machine-tools">
            <div className="seg" role="radiogroup" aria-label="Shade by">
              <button role="radio" aria-checked={shadeBy === "activity"} className={shadeBy === "activity" ? "seg-on" : ""} onClick={() => setShadeBy("activity")}>How much it adds</button>
              <button role="radio" aria-checked={shadeBy === "votes"} className={shadeBy === "votes" ? "seg-on" : ""} onClick={() => setShadeBy("votes")}>How it votes</button>
            </div>
            {shadeBy === "activity" ? <HighlightChips value={lit} onChange={setLit} /> : result && (
              <p className="small vote-legend">
                <Term tip="A part's vote is what its output, on its own, adds to the gap between two tokens' scores at the last token. All the votes add up exactly to the gap.">Votes</Term> for <span className="vote-for">“{word(tokenizer.piece(guesses[0]))}”</span> over{" "}
                <select value={rival} onChange={(e) => setRival(Number(e.target.value))} aria-label="The token to compare with">
                  {guesses.slice(1).map((g, i) => <option key={i} value={i + 1}>{show(tokenizer.piece(g))}</option>)}
                </select>{" "}
                <span className="vote-against">against</span>. <a href={href({ view: "lab", tool: "trace" })}>Trace it back in the Lab →</a>
              </p>
            )}
          </div>
          <div className="table" role="grid" aria-label="Attention heads by layer" onMouseLeave={() => setHover(null)}>
            <div className="table-corner"><Tip tip={GLOSSARY.head} align="start">head</Tip></div>
            {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
            {Array.from({ length: H }, (_, h) => (
              <Row key={h} label={`${h}`}>
                {Array.from({ length: L }, (_, l) => {
                  const id = `${l}.${h}`;
                  const measured = jobsOf(jobs, l, h, H);
                  return (
                    <button key={id} className={`head${focus === id ? " head-on" : ""}${selected === id ? " head-picked" : ""}${lit && !voted ? (litHeads.has(id) ? " head-lit" : " head-dim") : ""}`}
                      style={{
                        background: headBg(l, h),
                        // Alternate two identical animations so each new run replays the ripple.
                        animationName: version % 2 ? "ripple-a" : "ripple-b",
                        animationDelay: `${l * 35}ms`,
                      }}
                      onMouseEnter={() => setHover(id)} onFocus={() => setHover(id)} onBlur={() => setHover(null)}
                      onClick={() => setSelected(id)}
                      aria-pressed={selected === id}
                      aria-label={`Head ${id}${measured.length ? `, ${measured.map((j) => JOB_TEXT[j].name).join(", ")}` : ""}${IOI_ROLES[id] ? ", research role" : ""}`}>
                    </button>
                  );
                })}
              </Row>
            ))}
            <div className="table-row-label table-mlp-label"><Tip tip={GLOSSARY.mlp} align="start">MLP</Tip></div>
            {Array.from({ length: L }, (_, l) => (
              <button key={l} className={`mlp-cell${focus === `m${l}` ? " head-on" : ""}${selected === `m${l}` ? " head-picked" : ""}`} aria-label={`MLP of layer ${l}`}
                onMouseEnter={() => setHover(`m${l}`)} onFocus={() => setHover(`m${l}`)} onBlur={() => setHover(null)} onClick={() => setSelected(`m${l}`)} aria-pressed={selected === `m${l}`}
                style={{ background: mlpBg(l), animationName: version % 2 ? "ripple-a" : "ripple-b", animationDelay: `${l * 35 + 20}ms` }} />
            ))}
          </div>
          {hover && <div className="grid-hint" aria-hidden="true"><strong>{hover.startsWith("m") ? `MLP ${hover.slice(1)}` : hover}</strong> {hover.startsWith("m") ? "Works on each token by itself. Click for what it pushes toward." : label(hover) ?? habits(hover)[0]}</div>}
          <div className="table-foot">
            <Tip tip={GLOSSARY.layer} align="start">layer →</Tip>
            {voted ? (
              <span className="scale"><span>against</span><span className="scale-bar scale-votes" /><span>for</span></span>
            ) : (
              <Tip tip={GLOSSARY.activity} align="end"><span className="scale"><span>less</span><span className="scale-bar" /><span>more added at “{lastWord}”</span></span></Tip>
            )}
          </div>
        </div>

        <aside className="machine-side">
          {focusMlp !== null ? (
            <MlpCard layer={focusMlp} trace={result?.trace ?? null} weights={weights} tokenizer={tokenizer} position={pos}
              activity={activity ? `${Math.round((activity.mlps[focusMlp] / mlpMax) * 100)}%` : "…"} />
          ) : focused ? (
            <div className="card head-card">
              <div className="head-name">Head {focus}</div>
              <p className="head-gist">{label(focus!) ?? habits(focus!)[0]}</p>
              {research(focus!).circuits.length > 0 && (
                <div className="role-tags">{research(focus!).circuits.map((c) => <span key={c.role} className="role-tag" title={c.text}>{c.role}</span>)}</div>
              )}
              <p className="muted small">At “{lastWord}” it adds {focused.activity} as much as the busiest head.{voted ? ` Its vote: ${(voteOf.get(focus!) ?? 0) > 0 ? "+" : ""}${(voteOf.get(focus!) ?? 0).toFixed(2)}.` : ""}</p>
              {result && current && <DirectEffect trace={result.trace} weights={weights} tokenizer={tokenizer} vector={headVector} position={pos} />}
              <div className="card-buttons">
                <a className="primary" href={href({ view: "head", head: focus! })}>Open head {focus} →</a>
                <button className="secondary" onClick={() => toLab([focus!])}>Unplug it in the Lab</button>
              </div>
              <details className="more">
                <summary>More about head {focus}</summary>
                <ResearchNotes id={focus!} />
                {focused.jobs.map((j) => <p key={j.name}><strong>{j.name}.</strong> {j.text} <Term tip={GLOSSARY.score}>Tested just now: {j.score}</Term>.</p>)}
                <div className="habits">
                  <Term tip={ATLAS_SOURCE}><span className="card-label">Measured on Wikipedia text</span></Term>
                  <ul>{habits(focus!).map((h) => <li key={h}>{h}</li>)}</ul>
                </div>
                <HeadExamples id={focus!} setPrompt={setPrompt} limit={2} />
              </details>
            </div>
          ) : (
            <div className="card">
              <div className="card-label">How to read this</div>
              <p>Columns are the 12 layers, left to right. Each square is an attention head (9.9 is layer 9, head 9), and the teal row is each layer's MLP. Brighter squares add more at the last token.</p>
              <p className="muted">Click any square to learn what it does, then open it to look inside.</p>
              <button className="secondary" onClick={tryNameMovers}>Try it: unplug the name movers</button>
            </div>
          )}
          <nav className="exhibit-list" aria-label="Places to go">
            <div className="card-label">Go deeper</div>
            {PLACES.map((e) => (
              <a key={e.name} href={e.route} className="exhibit-link"><span><strong>{e.name}</strong><span className="muted"> · {e.text}</span></span></a>
            ))}
          </nav>
        </aside>
      </div>
    </article>
  );
}

// GPT-2's top guess for the token after position `pos`.
function expectedAt(trace: Trace, L: number, pos: number, tok: Tokenizer) {
  const T = trace.tokens.length;
  const row = L * T + pos;
  const K = trace.lens!.topK;
  return { piece: tok.piece(trace.lens!.ids[row * K]), p: Math.exp(trace.lens!.logprobs[row * K]) };
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <div className="table-row-label">{label}</div>
      {children}
    </>
  );
}

function describeHead(id: string, jobs: HeadJobs, H: number, heads: Float32Array | null, max: number) {
  const [l, h] = id.split(".").map(Number);
  return {
    activity: heads ? `${Math.round((heads[l * H + h] / max) * 100)}%` : "…",
    jobs: jobsOf(jobs, l, h, H).map((j) => ({ ...JOB_TEXT[j], score: jobs.scores[j][l * H + h].toFixed(2) })),
    role: IOI_ROLE[id] ? ROLES[IOI_ROLE[id]] : null,
  };
}
