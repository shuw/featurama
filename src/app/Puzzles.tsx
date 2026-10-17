// Puzzle levels, the list of them, and the way into You vs GPT-2.

import { useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention } from "../engine/hooks.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { readingLabel, show, word } from "./analysis.ts";
import { featureHint, featureName, searchFeatures, steering, useSae } from "./features.ts";
import type { HeadJobs } from "./heads.ts";
import { HighlightChips, highlighted } from "./highlights.tsx";
import { tokenize } from "./model.ts";
import { pct } from "./PromptBox.tsx";
import { check, LEVELS, loadProgress, saveProgress, shareText, tokenOf, type Level, type Progress } from "./puzzles.ts";
import { label } from "./research.ts";
import { href } from "./route.ts";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  jobs: HeadJobs;
}

// The Play page: puzzles first, then the guessing game.
export function PlayHub() {
  const progress = loadProgress();
  return (
    <article className="page">
      <header className="page-head">
        <h1>Puzzles for the machine</h1>
        <p className="dek">Each puzzle gives you a goal and one tool: unplug heads, steer with a feature, find a head, or read the logit lens. Every one has been solved on this same model.</p>
      </header>
      <PuzzleList progress={progress} />
      <section className="card play-guess">
        <div>
          <div className="card-label">Also</div>
          <strong>You vs GPT-2</strong>
          <p className="muted">Guess the next word of ten everyday sentences, and see how GPT-2 does on the same ones.</p>
        </div>
        <a className="primary" href={href({ view: "play", level: "guess" })}>Play →</a>
      </section>
    </article>
  );
}

export function PuzzleList({ progress }: { progress: Progress }) {
  const solved = LEVELS.filter((l) => progress[l.id]?.solved).length;
  return (
    <section className="puzzles">
      <div className="puzzle-head">
        <h2>Puzzles</h2>
        <span className="muted">{solved} of {LEVELS.length} solved</span>
        <ShareButton progress={progress} />
      </div>
      <ol className="puzzle-grid">
        {LEVELS.map((l, i) => (
          <li key={l.id}>
            <a className={progress[l.id]?.solved ? "puzzle-card puzzle-done" : "puzzle-card"} href={href({ view: "play", level: l.id })}>
              <span className="puzzle-n">{i + 1}</span>
              <span className="puzzle-kind">{KIND[l.kind]}</span>
              <strong>{l.title}</strong>
              {progress[l.id]?.solved && <span className="puzzle-tick" aria-label="solved">✓</span>}
            </a>
          </li>
        ))}
      </ol>
    </section>
  );
}

const KIND: Record<Level["kind"], string> = { unplug: "Unplug", steer: "Steer", find: "Find a head", lens: "Read the lens" };

function ShareButton({ progress }: { progress: Progress }) {
  const [copied, setCopied] = useState(false);
  const text = shareText(progress, new URL(href({ view: "play" }), document.baseURI).href);
  return (
    <button className="secondary share" onClick={async () => {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {
        prompt("Copy your result:", text);
      }
    }}>{copied ? "Copied!" : "Share your progress"}</button>
  );
}

export function PuzzleLevel({ engine, tokenizer, jobs, id }: Props & { id: string }) {
  const index = LEVELS.findIndex((l) => l.id === id);
  const level = LEVELS[index];
  const next = LEVELS[index + 1];
  const [progress, setProgress] = useState(loadProgress);
  const [showHint, setShowHint] = useState(false);
  const solved = !!progress[level?.id]?.solved;
  const win = (best?: number) => {
    setProgress((p) => {
      const prev = p[level.id];
      const n = { ...p, [level.id]: { solved: true, best: best === undefined ? prev?.best : Math.min(best, prev?.best ?? Infinity) } };
      saveProgress(n);
      return n;
    });
  };
  if (!level) {
    return (
      <article className="page">
        <header className="page-head">
          <div className="eyebrow"><a href={href({ view: "play" })}>Puzzles</a></div>
          <h1>No such puzzle</h1>
          <p className="dek">There's no puzzle called “{id}”. <a href={href({ view: "play" })}>See all puzzles</a>.</p>
        </header>
      </article>
    );
  }
  return (
    <article className="page puzzle">
      <header className="page-head">
        <div className="eyebrow"><a href={href({ view: "play" })}>Puzzles</a> · {index + 1} of {LEVELS.length} · {KIND[level.kind]}</div>
        <h1>{level.title}</h1>
        <p className="dek">{level.story}</p>
      </header>
      <p className="puzzle-prompt mono">{level.prompt}</p>
      {level.kind === "unplug" && <Unplug key={level.id} level={level} engine={engine} tokenizer={tokenizer} jobs={jobs} onWin={win} />}
      {level.kind === "steer" && <Steer key={level.id} level={level} engine={engine} tokenizer={tokenizer} onWin={() => win()} />}
      {level.kind === "find" && <Find key={level.id} level={level} engine={engine} tokenizer={tokenizer} jobs={jobs} onWin={() => win()} />}
      {level.kind === "lens" && <Lens key={level.id} level={level} engine={engine} tokenizer={tokenizer} onWin={() => win()} />}
      <div className="puzzle-foot">
        {solved ? (
          <section className="card card-result" aria-live="polite">
            <div className="card-label">Solved{level.kind === "unplug" && progress[level.id]?.best !== undefined ? ` with ${progress[level.id].best} head${progress[level.id].best === 1 ? "" : "s"} (par ${level.par})` : ""}</div>
            <p>{level.after}</p>
            <div className="card-buttons">
              {next ? <a className="primary" href={href({ view: "play", level: next.id })}>Next: {next.title} →</a> : <a className="primary" href={href({ view: "play" })}>All puzzles</a>}
              <ShareButton progress={progress} />
            </div>
          </section>
        ) : (
          <button className="link-small" onClick={() => setShowHint(!showHint)}>{showHint ? "Hide the hint" : "Stuck? Show a hint"}</button>
        )}
        {!solved && showHint && <p className="muted">{level.hint}</p>}
      </div>
    </article>
  );
}

type Outcome = Awaited<ReturnType<typeof check>>;

function Outcome({ level, outcome, tokenizer }: { level: Extract<Level, { goal: unknown }>; outcome: Outcome | null; tokenizer: Tokenizer }) {
  const g = level.goal;
  const want = g.is === "top" ? `“${word(g.token)}” is the top guess` : g.is === "not-top" ? `“${word(g.token)}” is not the top guess` : g.is === "above" ? `“${word(g.token)}” at ${pct(g.p!)} or more` : `“${word(g.token)}” under ${pct(g.p!)}`;
  return (
    <div className={outcome?.met ? "outcome outcome-met" : "outcome"}>
      <div className="outcome-goal"><span className="card-label">Goal</span> {want}</div>
      {!outcome ? <p className="muted">Running…</p> : (
        <>
          <div className="outcome-now"><span className="card-label">Now</span> “{word(g.token)}” at <strong>{pct(outcome.p)}</strong>{outcome.met && <span className="outcome-tick"> ✓ goal met</span>}</div>
          <div className="outcome-top">{outcome.top.map((t, i) => <span key={i} className={tokenizer.piece(t.id) === g.token ? "push push-up" : "push"}>{show(tokenizer.piece(t.id))} {pct(t.p)}</span>)}</div>
        </>
      )}
    </div>
  );
}

function HeadGrid({ engine, jobs, selected, onToggle, allowed, note }: {
  engine: GpuEngine; jobs: HeadJobs; selected: string[]; onToggle: (id: string) => void; allowed?: [number, number]; note?: (id: string) => string;
}) {
  const { n_layers: L, n_heads: H } = engine.config;
  const [lit, setLit] = useState<string | null>(null);
  const litHeads = useMemo(() => highlighted(lit, jobs, L, H), [lit, jobs, L, H]);
  return (
    <div className="picker puzzle-grid-wrap">
      <div className="picker-grid" role="group" aria-label="Heads: columns are layers, rows are heads">
        {Array.from({ length: H }, (_, h) => Array.from({ length: L }, (_, l) => {
          const id = `${l}.${h}`;
          const ok = !allowed || (l >= allowed[0] && l <= allowed[1]);
          return (
            <button key={id} data-head={id} disabled={!ok} aria-pressed={selected.includes(id)} aria-label={`Head ${id}`} title={`Head ${id}${label(id) ? `: ${label(id)}` : ""}${note ? ` ${note(id)}` : ""}`}
              className={`pick${selected.includes(id) ? " pick-chosen" : ""}${!ok ? " pick-locked" : ""}${lit ? (litHeads.has(id) ? " pick-lit" : " pick-dim") : ""}`}
              style={{ gridColumn: l + 1, gridRow: h + 1 }} onClick={() => onToggle(id)} />
          );
        }))}
      </div>
      <div className="picker-axis"><span>layer 0 →</span><span>11</span></div>
      <HighlightChips value={lit} onChange={setLit} compact />
    </div>
  );
}

function Unplug({ level, engine, tokenizer, jobs, onWin }: { level: Extract<Level, { kind: "unplug" }>; engine: GpuEngine; tokenizer: Tokenizer; jobs: HeadJobs; onWin: (heads: number) => void }) {
  const [off, setOff] = useState<string[]>([]);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  useEffect(() => {
    let live = true;
    const ivs: Intervention[] = off.map((id) => {
      const [layer, head] = id.split(".").map(Number);
      return { kind: "head", layer, head, scale: 0 };
    });
    check(engine, tokenizer, level.prompt, level.goal, ivs).then((o) => {
      if (!live) return;
      setOutcome(o);
      if (o.met && off.length > 0) onWin(off.length);
    });
    return () => {
      live = false;
    };
  }, [off.join(",")]);
  const toggle = (id: string) => setOff((o) => (o.includes(id) ? o.filter((x) => x !== id) : o.length >= level.budget ? o : [...o, id]));
  return (
    <div className="puzzle-play">
      <div>
        <HeadGrid engine={engine} jobs={jobs} selected={off} onToggle={toggle} allowed={level.layers} />
        <p className="muted small">Click heads to unplug them: {off.length} of {level.budget} used{level.layers ? `, layers ${level.layers[0]}–${level.layers[1]} only` : ""}. {off.length > 0 && <button className="link-small" onClick={() => setOff([])}>Plug all back in</button>}</p>
      </div>
      <Outcome level={level} outcome={outcome} tokenizer={tokenizer} />
    </div>
  );
}

function Steer({ level, engine, tokenizer, onWin }: { level: Extract<Level, { kind: "steer" }>; engine: GpuEngine; tokenizer: Tokenizer; onWin: () => void }) {
  const sae = useSae(engine);
  const ready = sae.status === "ready" ? sae : null;
  const [search, setSearch] = useState("");
  const [feature, setFeature] = useState<number | null>(null);
  const [strength, setStrength] = useState(0);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const results = useMemo(() => (ready ? searchFeatures(ready.index, tokenizer, search, 8) : []), [ready, search, tokenizer]);
  useEffect(() => {
    let live = true;
    const t = setTimeout(async () => {
      const T = tokenize(tokenizer, level.prompt).length;
      const ivs = ready && feature !== null && strength !== 0 ? [steering(await ready.sae.direction(feature), strength, T)] : [];
      const o = await check(engine, tokenizer, level.prompt, level.goal, ivs);
      if (!live) return;
      setOutcome(o);
      if (o.met && ivs.length) onWin();
    }, 80);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [feature, strength, ready]);
  const name = (id: number) => (ready ? featureName(id, ready.index[id], tokenizer) : `#${id}`);
  return (
    <div className="puzzle-play">
      <div className="card">
        {!ready ? <p className="muted">{sae.status === "error" ? `Couldn't load the features: ${sae.message}` : "Fetching the feature dictionary (38 MB, only the first time)…"}</p> : (
          <>
            <div className="card-label">1. Find a feature</div>
            <input className="feat-search" placeholder="A word, like “ocean”" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a feature" />
            <ul className="feat-list">
              {results.map((id) => (
                <li key={id}><button className={feature === id ? "feat-row feat-row-on" : "feat-row"} onClick={() => setFeature(id)}>
                  <span className="feat-name">{name(id)} <span className="feat-hint">{featureHint(ready.index[id], tokenizer)}</span></span><span className="feat-id">#{id}</span>
                </button></li>
              ))}
            </ul>
            <div className="card-label">2. Set the strength{feature !== null ? ` of #${feature}` : ""}</div>
            <label className="steer-row">
              <input type="range" min={-100} max={150} step={5} value={strength} disabled={feature === null} onChange={(e) => setStrength(Number(e.target.value))} aria-label="Steering strength" />
              <span className="mono">{strength > 0 ? "+" : ""}{strength}</span>
            </label>
          </>
        )}
      </div>
      <Outcome level={level} outcome={outcome} tokenizer={tokenizer} />
    </div>
  );
}

function Find({ level, engine, tokenizer, jobs, onWin }: { level: Extract<Level, { kind: "find" }>; engine: GpuEngine; tokenizer: Tokenizer; jobs: HeadJobs; onWin: () => void }) {
  const { n_heads: H } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, level.prompt), [tokenizer, level.prompt]);
  const T = tokens.length;
  const [pattern, setPattern] = useState<Float32Array | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => {
    engine.run({ tokens, read: { pattern: true } }).then((t) => setPattern(t.pattern!));
  }, [tokens]);
  const q = T - 1;
  const k = level.target === "previous" ? q - 1 : tokens.findIndex((t, i) => i > 0 && t === tokenOf(tokenizer, (level.target as { token: string }).token));
  const share = (id: string) => {
    const [l, h] = id.split(".").map(Number);
    return pattern ? pattern[((l * H + h) * T + q) * T + k] : 0;
  };
  const a = picked ? share(picked) : null;
  const met = a !== null && a >= level.min;
  useEffect(() => {
    if (met) onWin();
  }, [met]);
  const where = `“${word(tokenizer.piece(tokens[k]))}”`;
  return (
    <div className="puzzle-play">
      <HeadGrid engine={engine} jobs={jobs} selected={picked ? [picked] : []} onToggle={setPicked} />
      <div className={met ? "outcome outcome-met" : "outcome"}>
        <div className="outcome-goal"><span className="card-label">Goal</span> at least {pct(level.min)} on {where}, from the last “{word(tokenizer.piece(tokens[q]))}”</div>
        {!picked ? <p className="muted">Click a head to test it.</p> : (
          <div className="outcome-now"><span className="card-label">Head {picked}</span> puts <strong>{pct(a!)}</strong> on {where}{met && <span className="outcome-tick"> ✓ found one</span>}</div>
        )}
      </div>
    </div>
  );
}

function Lens({ level, engine, tokenizer, onWin }: { level: Extract<Level, { kind: "lens" }>; engine: GpuEngine; tokenizer: Tokenizer; onWin: () => void }) {
  const L = engine.config.n_layers;
  const [rows, setRows] = useState<string[] | null>(null);
  const [guess, setGuess] = useState<number | null>(null);
  useEffect(() => {
    const tokens = tokenize(tokenizer, level.prompt);
    engine.run({ tokens, lens: { positions: [tokens.length - 1], topK: 1 } }).then((t) => setRows(Array.from({ length: L + 1 }, (_, r) => tokenizer.piece(t.lens!.ids[r]))));
  }, []);
  const answer = rows ? rows.findIndex((p) => p === level.answer) : -1;
  const right = guess !== null && guess === answer;
  useEffect(() => {
    if (right) onWin();
  }, [right]);
  return (
    <div className="puzzle-play">
      <div className="card">
        <div className="card-label">Your answer</div>
        <div className="lens-choices">
          {Array.from({ length: L + 1 }, (_, r) => (
            <button key={r} className={guess === r ? (right ? "hl-chip hl-on" : "hl-chip lens-wrong") : "hl-chip"} onClick={() => setGuess(r)} disabled={!rows}>{readingLabel(r)}</button>
          ))}
        </div>
        {guess !== null && !right && <p className="muted">Not quite. {guess < answer ? `“${word(level.answer)}” isn't on top yet there.` : "It's already on top before that."}</p>}
      </div>
      {guess !== null && rows && (
        <div className="card">
          <div className="card-label">GPT-2's best guess after each layer</div>
          <ol className="lens-rows">{rows.map((p, r) => <li key={r} className={r === answer ? "lens-hit" : ""}><span className="muted">{readingLabel(r)}</span> <span className="mono">{show(p)}</span></li>)}</ol>
        </div>
      )}
    </div>
  );
}
