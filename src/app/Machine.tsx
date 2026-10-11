// The home screen: every part of GPT-2 laid out like a periodic table,
// shaded by how much it writes at the last token of your prompt. Heads can
// be unplugged to see what the model does without them.

import { useCallback, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention, Trace } from "../engine/hooks.ts";
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
import { href } from "./route.ts";
import { Term, Tip } from "./Tip.tsx";
import { ATLAS_SOURCE, habits } from "./atlas.ts";
import { HeadExamples } from "./HeadExamples.tsx";
import { label, research } from "./research.ts";
import { ResearchNotes } from "./ResearchNotes.tsx";
import { useRun } from "./useRun.ts";

const EXHIBITS = [
  { n: 1, name: "Tokens", text: "How text is cut into pieces" },
  { n: 2, name: "Attention", text: "Who looks at whom", route: href({ view: "attention" }) },
  { n: 3, name: "Layers", text: "Watch the guess take shape", route: href({ view: "layers" }) },
  { n: 4, name: "Patching", text: "Swap activations until the answer flips", route: href({ view: "patching" }) },
  { n: 5, name: "Features", text: "Concepts inside the numbers, and steering with them", route: href({ view: "features" }) },
  { n: 6, name: "Zoom", text: "From the answer down to single numbers", route: href({ view: "zoom" }) },
];

// The classic experiment: switch off the heads that copy the name.
const IOI_PROMPT = "When Mary and John went to the store, John gave a drink to";
const NAME_MOVERS = ["9.6", "9.9", "10.0"];
const BACKUPS = ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"];

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
  const [unplugged, setUnplugged] = useState<string[]>([]);
  const [lit, setLit] = useState<string | null>(null);
  const litHeads = useMemo(() => highlighted(lit, jobs, L, H), [lit, jobs, L, H]);
  const unKey = unplugged.join(",");

  const run = useCallback(async (t: number[]) => {
    const last = t.length - 1;
    const interventions: Intervention[] = unplugged.map((id) => {
      const [layer, head] = id.split(".").map(Number);
      return { kind: "head", layer, head, scale: 0 };
    });
    // With heads unplugged, also run the intact model, to compare its top
    // guesses with what the changed model now gives them.
    const intact = interventions.length ? await engine.run({ tokens: t, lens: { positions: [last], topK: 5 } }) : null;
    const targets = intact ? Array.from(intact.lens!.ids.subarray(L * 5, L * 5 + 5)) : [];
    // The lens at every position, so pointing at any token can show what GPT-2 expected next there.
    const positions = t.map((_, i) => i);
    const trace = await engine.run({ tokens: t, interventions, read: { z: true, resid: true, residMid: true }, lens: { positions, topK: 10, targets } });
    const row = L * t.length + last; // final layer, last token
    const compare = intact
      ? {
          items: targets.map((id, k) => ({ piece: tokenizer.piece(id), before: Math.exp(intact.lens!.logprobs[L * 5 + k]), after: Math.exp(trace.lens!.targetLogprobs[row * targets.length + k]) })),
          now: tokenizer.piece(trace.lens!.ids[row * 10]),
        }
      : null;
    return { trace, compare };
  }, [engine, weights, tokenizer, d, L, unKey]);
  const { result, version } = useRun(tokens, run, 150, unKey);

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

  const focus = hover ?? selected;
  const focusMlp = focus?.startsWith("m") ? Number(focus.slice(1)) : null;
  const focused = focus && focusMlp === null ? describeHead(focus, jobs, H, activity?.heads ?? null, headMax) : null;
  const headVector = useCallback(() => {
    const [hl, hh] = (focus ?? "0.0").split(".").map(Number);
    return headOutput(result!.trace, weights, hl, hh, pos);
  }, [focus, result, weights, pos]);
  const isOff = (id: string) => unplugged.includes(id);
  const toggle = (id: string) => setUnplugged((u) => (u.includes(id) ? u.filter((x) => x !== id) : [...u, id]));
  const tryNameMovers = () => {
    setPrompt(IOI_PROMPT);
    setUnplugged(NAME_MOVERS);
    setSelected("9.9");
  };

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">The machine · GPT-2 small</div>
        <h1>Every part of the model, working on your prompt</h1>
        <p className="dek">GPT-2 small has 12 <Term tip={GLOSSARY.layer}>layers</Term>. Each has 12 <Term tip={GLOSSARY.head}>attention heads</Term>, which move information between tokens, and one <Term tip={GLOSSARY.mlp}>MLP</Term>, which works on each token alone. Brightness shows how much each part adds at {pointed === null ? "the last token" : "the token you're pointing at"}, “{lastWord}”{expected ? <>, where GPT-2 expected “{word(expected.piece)}” next ({pct(expected.p)})</> : null}. Point at a token in the prompt to see another; click a head to learn about it or unplug it.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        onTokenHover={setPointed} activeToken={pointed}
        next={result ? predictions(result.trace, L, tokenizer) : null} stale={!current} />

      <div className="machine-body">
        <div className={!current && result ? "stage table-wrap stale" : "stage table-wrap"}>
          <HighlightChips value={lit} onChange={setLit} />
          <div className="table" role="grid" aria-label="Attention heads by layer" onMouseLeave={() => setHover(null)}>
            <div className="table-corner"><Tip tip={GLOSSARY.head} align="start">head</Tip></div>
            {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
            {Array.from({ length: H }, (_, h) => (
              <Row key={h} label={`${h}`}>
                {Array.from({ length: L }, (_, l) => {
                  const id = `${l}.${h}`;
                  const measured = jobsOf(jobs, l, h, H);
                  const off = isOff(id);
                  return (
                    <button key={id} className={`head${focus === id ? " head-on" : ""}${selected === id ? " head-picked" : ""}${off ? " head-off" : ""}${lit ? (litHeads.has(id) ? " head-lit" : " head-dim") : ""}`}
                      style={{
                        ...(off ? {} : { background: `rgba(var(--attn-rgb), ${activity ? shade(activity.heads[l * H + h], headMax) : "0.05"})` }),
                        // Alternate two identical animations so each new run replays the ripple.
                        animationName: version % 2 ? "ripple-a" : "ripple-b",
                        animationDelay: `${l * 35}ms`,
                      }}
                      onMouseEnter={() => setHover(id)} onFocus={() => setHover(id)} onBlur={() => setHover(null)}
                      onClick={() => setSelected(id)} onKeyDown={(e) => (e.key === "u" || e.key === "U") && toggle(id)}
                      aria-pressed={selected === id}
                      aria-label={`Head ${id}${off ? ", unplugged" : ""}${measured.length ? `, ${measured.map((j) => JOB_TEXT[j].name).join(", ")}` : ""}${IOI_ROLES[id] ? ", research role" : ""}`}>
                    </button>
                  );
                })}
              </Row>
            ))}
            <div className="table-row-label table-mlp-label"><Tip tip={GLOSSARY.mlp} align="start">MLP</Tip></div>
            {Array.from({ length: L }, (_, l) => (
              <button key={l} className={`mlp-cell${focus === `m${l}` ? " head-on" : ""}${selected === `m${l}` ? " head-picked" : ""}`} aria-label={`MLP of layer ${l}`}
                onMouseEnter={() => setHover(`m${l}`)} onFocus={() => setHover(`m${l}`)} onBlur={() => setHover(null)} onClick={() => setSelected(`m${l}`)} aria-pressed={selected === `m${l}`}
                style={{ background: `rgba(var(--mlp-rgb), ${activity ? shade(activity.mlps[l], mlpMax) : "0.05"})`, animationName: version % 2 ? "ripple-a" : "ripple-b", animationDelay: `${l * 35 + 20}ms` }} />
            ))}
          </div>
          {hover && <div className="grid-hint" aria-hidden="true"><strong>{hover.startsWith("m") ? `MLP ${hover.slice(1)}` : hover}</strong> {hover.startsWith("m") ? "Works on each token by itself. Click for what it pushes toward." : label(hover) ?? habits(hover)[0]}</div>}
          <div className="table-foot">
            <Tip tip={GLOSSARY.layer} align="start">layer →</Tip>
            <Tip tip={GLOSSARY.activity} align="end"><span className="scale"><span>less</span><span className="scale-bar" /><span>more added at “{lastWord}”</span></span></Tip>
          </div>
        </div>

        <aside className="machine-side">
          {unplugged.length > 0 && (
            <section className="card card-unplugged" aria-live="polite">
              <div className="card-label">Unplugged</div>
              <div className="plug-list">
                {unplugged.map((id) => (
                  <button key={id} className="plug-chip" onClick={() => toggle(id)} title={`Plug head ${id} back in`}>{id} <span aria-hidden="true">×</span></button>
                ))}
                <button className="plug-reset" onClick={() => setUnplugged([])}>Plug all back in</button>
              </div>
              {result?.compare && current && (
                <>
                  <p className="effect-headline">{headline(result.compare)}</p>
                  <ul className="effect-list">
                    {result.compare.items.map((c, i) => (
                      <li key={i}>
                        <span className="effect-piece">{show(c.piece)}</span>
                        <span className="effect-bar"><span className="effect-before" style={{ width: `${c.before * 100}%` }} /><span className="effect-after" style={{ width: `${c.after * 100}%` }} /></span>
                        <span className="effect-nums">{pct(c.before)} → <strong className={c.after < c.before ? "down" : "up"}>{pct(c.after)}</strong></span>
                      </li>
                    ))}
                  </ul>
                  <p className="muted small">Small changes are normal: when a head is switched off, others often step in. Researchers call this self-repair.</p>
                  {unKey === NAME_MOVERS.join(",") && prompt === IOI_PROMPT && (
                    <button className="secondary" onClick={() => setUnplugged([...NAME_MOVERS, ...BACKUPS])}>Now unplug the backup name movers too</button>
                  )}
                </>
              )}
            </section>
          )}
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
              <p className="muted small">{isOff(focus!) ? "Unplugged: it adds nothing." : `At “${lastWord}” it adds ${focused.activity} as much as the busiest head.`}</p>
              {result && current && <DirectEffect trace={result.trace} weights={weights} tokenizer={tokenizer} vector={headVector} position={pos} off={isOff(focus!)} />}
              <div className="card-buttons">
                <button className={isOff(focus!) ? "secondary" : "primary"} onClick={() => toggle(focus!)}>{isOff(focus!) ? "Plug back in" : "Unplug"}</button>
                <a className="secondary" href={href({ view: "attention", head: focus! })}>Open in Attention →</a>
                <a className="secondary" href={href({ view: "zoom", part: "head", arg: focus! })}>Look inside →</a>
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
              <p className="muted">Click any square to learn what it does.</p>
              <button className="secondary" onClick={tryNameMovers}>Try it: unplug the name movers</button>
            </div>
          )}
          <nav className="exhibit-list" aria-label="Exhibits">
            <div className="card-label">Exhibits</div>
            {EXHIBITS.map((e) =>
              e.route ? (
                <a key={e.n} href={e.route} className="exhibit-link"><span className="n">{e.n}</span><span><strong>{e.name}</strong><span className="muted"> · {e.text}</span></span></a>
              ) : (
                <div key={e.n} className="exhibit-link soon"><span className="n">{e.n}</span><span><strong>{e.name}</strong> · {e.text}</span><span className="soon-tag">soon</span></div>
              ),
            )}
          </nav>
        </aside>
      </div>
    </article>
  );
}

function headline({ items, now }: { items: { piece: string; before: number; after: number }[]; now: string }) {
  const compare = items;
  const was = compare[0].piece;
  return now === was
    ? `Top guess is still “${word(was)}”, at ${pct(compare[0].after)} instead of ${pct(compare[0].before)}.`
    : `Top guess changed from “${word(was)}” to “${word(now)}”.`;
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
