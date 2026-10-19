// The home page: the whole path from a prompt to GPT-2's next token as one
// diagram, run on your prompt, with six short steps that light up one part
// at a time.

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { GLOSSARY } from "./glossary.ts";
import { attentionRow, mlpActivity } from "./heads.ts";
import { votes } from "./inside.ts";
import { tokenize } from "./model.ts";
import { pct, PromptBox } from "./PromptBox.tsx";
import { label } from "./research.ts";
import { href, navigate, setUrlParam } from "./route.ts";
import { Term } from "./Tip.tsx";
import type { Writer } from "./useWriter.ts";
import { useRun } from "./useRun.ts";

// The steps, in the order a token meets the parts. Attention and MLPs repeat
// in every layer, so the stepper groups them.
const WHOLE = 0, TOKENS = 1, NUMBERS = 2, ATTN = 3, MLP = 4, STREAM = 5, GUESS = 6;
const STEPS = ["The whole path", "Tokens", "Numbers", "Attention", "MLPs", "Adding up", "The guess"];
// Long prompts show the start token and the last few, so lanes stay readable.
const MAX_LANES = 20;
// The "watch it run" film: when each part appears, in ms.
const FILM = { numbers: 350, layers: 800, layer: 210, guess: 800 + 12 * 210 + 150 };

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function Overview({ engine, tokenizer, weights, prompt, setPrompt, shuffle, writer }: Props) {
  const L = engine.config.n_layers;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const [step, setStep] = useState(WHOLE);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { qkv: true, z: true, pattern: true, resid: true, residMid: true }, lens: { positions: [t.length - 1], topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run, 150);
  const current = !!trace && trace.tokens === tokens;
  const facts = useMemo(() => (trace ? readOut(trace, weights) : null), [trace, weights]);
  const guesses = useMemo(() => (trace ? predictions(trace, L, tokenizer) : null), [trace, L, tokenizer]);
  // Counts showings of the film; it plays once when the first run lands.
  const [film, setFilm] = useState(0);
  useEffect(() => {
    if (trace && film === 0) setFilm(1);
  }, [trace]);
  const watch = () => {
    setStep(WHOLE);
    setFilm((f) => f + 1);
  };

  // ← and → step through, except while typing.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.altKey || e.metaKey || e.ctrlKey || (e.target as Element | null)?.closest?.("textarea, input, select")) return;
      if (e.key === "ArrowRight") setStep((s) => Math.min(GUESS, s + 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(WHOLE, s - 1));
    };
    addEventListener("keydown", key);
    return () => removeEventListener("keydown", key);
  }, []);

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Start here</div>
        <h1>How GPT-2 picks the next token</h1>
        <p className="dek">Every page on this site opens up one part of this picture. Type anything, then follow it through the model, one step at a time.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={guesses} stale={!current} />

      <Stepper step={step} setStep={setStep} />

      <div className="ov-body">
        <div className={!current && trace ? "ov-stage stale" : "ov-stage"}>
          <div className="ov-legend">
            <button onClick={() => setStep(STREAM)}><i className="ov-key ov-key-lane" />each token's 768 numbers</button>
            <button onClick={() => setStep(ATTN)}><i className="ov-key ov-key-attn" />attention: moves information between tokens</button>
            <button onClick={() => setStep(MLP)}><i className="ov-key ov-key-mlp" />MLP: works on one token</button>
          </div>
          <Path tokens={trace?.tokens ?? tokens} tokenizer={tokenizer} facts={facts} guesses={guesses} step={step} setStep={setStep} layers={L} film={film} />
        </div>
        <aside className="ov-side">
          <div className="card ov-card" aria-live="polite">
            <div className="card-label">{step === WHOLE ? "The big picture" : `Step ${step} of ${GUESS} · ${STEPS[step]}`}</div>
            <StepText step={step} trace={trace} facts={facts} guesses={guesses} tokenizer={tokenizer} writer={writer} watch={watch} />
            <div className="ov-nav">
              {step > WHOLE && <button className="secondary" onClick={() => setStep(step - 1)}>Back</button>}
              {step < GUESS ? (
                <button className="primary" onClick={() => setStep(step + 1)}>{step === WHOLE ? "Follow it step by step →" : `Next: ${STEPS[step + 1].toLowerCase()} →`}</button>
              ) : (
                <button className="secondary" onClick={() => setStep(WHOLE)}>Back to the whole path</button>
              )}
            </div>
          </div>
        </aside>
      </div>
    </article>
  );
}

// The outline, which doubles as the mental model: tokens, numbers, then
// attention and an MLP in each of 12 layers, adding up to a guess.
function Stepper({ step, setStep }: { step: number; setStep: (s: number) => void }) {
  const nav = useRef<HTMLElement>(null);
  // Keep the current step in view when the row scrolls (phones).
  useEffect(() => {
    const el = nav.current!;
    const on = el.querySelector<HTMLElement>(".ov-step-on");
    if (on && el.scrollWidth > el.clientWidth) el.scrollTo({ left: on.offsetLeft - el.offsetLeft - (el.clientWidth - on.offsetWidth) / 2, behavior: "smooth" });
  }, [step]);
  const item = (s: number) => (
    <button key={s} className={step === s ? "ov-step ov-step-on" : "ov-step"} aria-current={step === s ? "step" : undefined} onClick={() => setStep(s)}>
      {s > WHOLE && <span className="ov-step-n">{s}</span>}{STEPS[s]}
    </button>
  );
  const arrow = (k: string) => <span key={k} className="ov-step-arrow" aria-hidden="true">→</span>;
  return (
    <nav ref={nav} className="ov-stepper" aria-label="Steps">
      {item(WHOLE)}
      <span className="ov-step-sep" aria-hidden="true" />
      {item(TOKENS)}{arrow("a")}{item(NUMBERS)}{arrow("b")}
      <span className="ov-step-group"><span className="ov-step-times">× 12 layers</span>{item(ATTN)}{arrow("c")}{item(MLP)}</span>
      {arrow("d")}{item(STREAM)}{arrow("e")}{item(GUESS)}
    </nav>
  );
}

type Facts = ReturnType<typeof readOut>;
type Guesses = ReturnType<typeof predictions>;

// Everything the steps quote, measured on this run at the last token.
function readOut(trace: Trace, w: Weights) {
  const { n_layers: L, n_heads: H, d_model: d } = w.config;
  const T = trace.tokens.length;
  const last = T - 1;
  const lanes = T <= MAX_LANES ? trace.tokens.map((_, i) => i) : [0, ...Array.from({ length: MAX_LANES - 1 }, (_, i) => T - MAX_LANES + 1 + i)];
  // In each layer, the head that brings the most from one earlier token into
  // the last one: its attention to that token times the size of what that
  // token's value writes, against the size of the stream there. Tokens it
  // barely looks at are skipped.
  const arcs: { layer: number; head: number; from: number; share: number; size: number }[] = [];
  const dh = d / H;
  for (let l = 0; l < L; l++) {
    const WO = w.f32(`h.${l}.attn.c_proj.weight`);
    const stream = norm(trace.resid!.subarray((l * T + last) * d, (l * T + last + 1) * d));
    let best: (typeof arcs)[number] | null = null;
    for (let h = 0; h < H; h++) {
      const row = attentionRow(trace, H, l, h, last);
      for (const j of lanes) {
        if (j === 0 || j === last || row[j] < 0.05) continue;
        const size = (row[j] * valueSize(trace, WO, l, h, j, d, dh)) / stream;
        if (!best || size > best.size) best = { layer: l, head: h, from: j, share: row[j], size };
      }
    }
    if (best) arcs.push(best);
  }
  // The head and the MLP that push hardest for GPT-2's answer over its second guess.
  const K = trace.lens!.topK;
  const answer = trace.lens!.ids[L * K];
  const rival = trace.lens!.ids[L * K + 1];
  const vote = new Map(votes(trace, w, last, answer, rival).parts.map((p) => [p.id, p.value]));
  const top = (ids: string[]) => ids.reduce((a, b) => ((vote.get(b) ?? 0) > (vote.get(a) ?? 0) ? b : a));
  const voter = top(Array.from({ length: L * H }, (_, i) => `${Math.floor(i / H)}.${i % H}`));
  const [vl, vh] = voter.split(".").map(Number);
  const row = attentionRow(trace, H, vl, vh, last);
  const earlier = lanes.filter((j) => j > 0 && j < last);
  const from = earlier.length ? earlier.reduce((a, b) => (row[b] > row[a] ? b : a)) : -1;
  const head = from > 0 && row[from] >= 0.05 ? { layer: vl, head: vh, from, share: row[from] } : null;
  if (head && !arcs.some((a) => a.layer === vl && a.head === vh)) arcs.push({ ...head, size: Math.max(...arcs.map((a) => a.size)) });
  const mlp = Number(top(Array.from({ length: L }, (_, l) => `m${l}`)).slice(1));
  // How much each MLP adds at every lane.
  const mlps = lanes.map((p) => mlpActivity(trace, d, L, p));
  // The lens at the last token: reading 0 is the embedding alone, reading k is after layer k - 1.
  const lens = Array.from({ length: L + 1 }, (_, k) => trace.lens!.ids[k * K]);
  let settle = L;
  while (settle > 0 && lens[settle - 1] === lens[L]) settle--;
  const embed = (p: number) => trace.resid!.subarray(p * d, p * d + d);
  return { lanes, last, arcs, head, mlps, mlp, lens, settle, embed, rival };
}

const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

function valueSize(trace: Trace, WO: Float32Array, l: number, h: number, j: number, d: number, dh: number) {
  const T = trace.tokens.length;
  const vo = (l * T + j) * 3 * d + 2 * d + h * dh;
  const acc = new Float64Array(d);
  for (let k = 0; k < dh; k++) {
    const v = trace.qkv![vo + k];
    const row = (h * dh + k) * d;
    for (let o = 0; o < d; o++) acc[o] += v * WO[row + o];
  }
  let s = 0;
  for (let o = 0; o < d; o++) s += acc[o] * acc[o];
  return Math.sqrt(s);
}

// The diagram. Each token is a lane running down through the 12 layers; in
// every layer, attention moves information between lanes and the MLP works on
// each lane alone. The last lane is scored at the bottom.
function Path({ tokens, tokenizer, facts, guesses, step, setStep, layers: L, film }: {
  tokens: number[]; tokenizer: Tokenizer; facts: Facts | null; guesses: Guesses | null; step: number; setStep: (s: number) => void; layers: number; film: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(760);
  useEffect(() => {
    const el = box.current!;
    const ro = new ResizeObserver(() => setWidth(Math.max(320, Math.round(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // The film runs for a few seconds; after that, parts that appear while typing just appear.
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!film) return;
    setPlaying(true);
    const t = setTimeout(() => setPlaying(false), FILM.guess + 800);
    return () => clearTimeout(t);
  }, [film]);
  const narrow = width < 560;
  const T = tokens.length;
  const lanes = facts && facts.lanes[facts.lanes.length - 1] === T - 1 ? facts.lanes : tokens.map((_, i) => i).slice(T > MAX_LANES ? T - MAX_LANES : 0);
  const n = lanes.length;
  const LX = narrow ? 30 : 92;
  const RW = narrow ? 74 : 124;
  const x0 = LX + 8;
  const x1 = width - RW - 8;
  const gap = (x1 - x0) / n;
  const lx = (i: number) => x0 + gap * (i + 0.5);
  // Tight lanes get their token names at an angle above them.
  const slant = gap < 56;
  const TOK_Y = slant ? 70 : 22;
  const EMB_Y = TOK_Y + 34;
  const LAY_Y = EMB_Y + 40;
  const LH = 34;
  const by = (l: number) => LAY_Y + l * LH;
  const LAY_END = by(L);
  const OUT_Y = LAY_END + 22;
  const OUT_H = 92;
  const height = OUT_Y + OUT_H + 8;
  const lastX = lx(n - 1);
  // Each part is lit on its own steps, half lit where it's context, dim otherwise.
  const cls = (lit: number[], half: number[] = []) => `ov-part${step === WHOLE || lit.includes(step) ? "" : half.includes(step) ? " ov-half" : " ov-dim"}`;
  const go = (s: number) => ({ onClick: () => setStep(s), style: { cursor: "pointer" } });
  const at = (ms: number) => ({ "--d": `${Math.round(ms)}ms` }) as CSSProperties;
  const chars = Math.max(2, Math.floor((gap - 8) / 6.8));
  const fit = (s: string, k: number) => (s.length > k ? s.slice(0, Math.max(1, k - 1)) + "…" : s);
  const pieceText = (p: number) => (p === 0 ? "⟨start⟩" : word(tokenizer.piece(tokens[p])));
  const lensText = (k: number) => fit(word(tokenizer.piece(facts!.lens[k])), narrow ? 8 : 14);
  const maxArc = facts?.arcs.length ? Math.max(...facts.arcs.map((a) => a.size)) : 1;
  const mlpMax = facts ? Math.max(...facts.mlps.slice(1).flatMap((m) => [...m]), 1e-6) : 1;
  const embMax = facts ? Math.max(...lanes.slice(1).flatMap((p) => [...facts.embed(p).subarray(0, 16)].map(Math.abs)), 1e-6) : 1;
  const laneOf = (p: number) => lanes.indexOf(p);
  const layerAt = (l: number) => FILM.layers + l * FILM.layer;

  return (
    <div className="ov-path" ref={box}>
      <svg key={film} className={playing ? "ov-film" : undefined} viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img"
        aria-label={`GPT-2 reading ${T} tokens through ${L} layers${guesses ? `, then guessing “${word(guesses[0].piece)}”` : ""}`}>
        <defs>
          <marker id="ov-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0 0L10 5L0 10z" className="ov-arrowhead" />
          </marker>
        </defs>

        {/* Lanes: the residual stream. */}
        <g className={cls([TOKENS, STREAM], [NUMBERS, ATTN, MLP, GUESS])} {...go(STREAM)}>
          <title>Each token's 768 numbers, flowing down: the residual stream</title>
          {lanes.map((p, i) => (
            <line key={p} x1={lx(i)} x2={lx(i)} y1={TOK_Y + 12} y2={i === n - 1 ? OUT_Y : LAY_END - 6} pathLength={1}
              className={`ov-flow ${i === n - 1 ? "ov-lane ov-lane-last" : "ov-lane"}`} style={at(FILM.numbers)} />
          ))}
        </g>

        {/* Tokens. */}
        <g className={cls([TOKENS], [NUMBERS, ATTN, MLP, STREAM, GUESS])} {...go(TOKENS)}>
          <title>Your prompt, cut into tokens</title>
          {!narrow && <text x={LX} y={TOK_Y + 4} textAnchor="end" className="ov-label">tokens</text>}
          {lanes.length < T && <text x={(lx(0) + lx(1)) / 2} y={TOK_Y + 4} textAnchor="middle" className="ov-label">…</text>}
          {lanes.map((p, i) => {
            const full = p === 0 ? "⟨start⟩" : show(tokenizer.piece(tokens[p]));
            if (slant) {
              return (
                <g key={p} className="ov-seq" style={at(i * 25)}>
                  <rect x={lx(i) - 7} y={TOK_Y - 6} width={14} height={12} rx={3} className={`ov-tok tk${p % 4}`} />
                  <text x={lx(i) - 2} y={TOK_Y - 12} transform={`rotate(-40 ${lx(i) - 2} ${TOK_Y - 12})`} className="ov-tok-text">{fit(pieceText(p), 10)}<title>{full}</title></text>
                </g>
              );
            }
            const t = fit(pieceText(p), chars);
            const w = Math.min(gap - 4, t.length * 6.8 + 10);
            return (
              <g key={p} className="ov-seq" style={at(i * 25)}>
                <rect x={lx(i) - w / 2} y={TOK_Y - 10} width={w} height={20} rx={4} className={`ov-tok tk${p % 4}`} />
                <text x={lx(i)} y={TOK_Y + 4} textAnchor="middle" className="ov-tok-text">{t}<title>{full}</title></text>
              </g>
            );
          })}
        </g>

        {/* Each token's first 16 of its 768 numbers. */}
        <g className={cls([NUMBERS])} {...go(NUMBERS)}>
          <title>Each token becomes 768 numbers (the first 16 shown)</title>
          {!narrow && <text x={LX} y={EMB_Y + 4} textAnchor="end" className="ov-label">768 numbers</text>}
          {facts && lanes.map((p, i) => {
            const v = facts.embed(p);
            const w = Math.min(64, gap * 0.8);
            return (
              <g key={p} className="ov-seq" style={at(FILM.numbers + i * 25)}>
                <rect x={lx(i) - w / 2 - 1} y={EMB_Y - 6} width={w + 2} height={12} rx={2} className="ov-emb-frame" />
                {Array.from({ length: 16 }, (_, k) => (
                  <rect key={k} x={lx(i) - w / 2 + (k * w) / 16} y={EMB_Y - 5} width={w / 16 - 0.4} height={10}
                    style={{ fill: `rgba(var(${v[k] >= 0 ? "--pos-rgb" : "--neg-rgb"}), ${(0.08 + 0.92 * Math.min(1, Math.abs(v[k]) / embMax)).toFixed(2)})` }} />
                ))}
              </g>
            );
          })}
        </g>

        {/* Layers: attention across lanes, then the MLP on each lane. */}
        {Array.from({ length: L }, (_, l) => (
          <g key={l}>
            <g className={cls([STREAM], [ATTN, MLP])}>
              <text x={LX} y={by(l) + 16} textAnchor="end" className="ov-label ov-seq" style={at(layerAt(l))}>{narrow ? l : `layer ${l}`}</text>
            </g>
            <g className={cls([ATTN], [STREAM])} {...go(ATTN)}>
              <rect x={x0 - 4} y={by(l) + 11} width={x1 - x0 + 8} height={2} rx={1} className="ov-attn-bar ov-seq" style={at(layerAt(l))} />
            </g>
            <g className={cls([MLP], [STREAM])} {...go(MLP)}>
              <title>Layer {l}'s MLP, on each token</title>
              {lanes.map((p, i) => {
                const a = facts ? Math.min(1, facts.mlps[i][l] / mlpMax) : 0.3;
                const picked = facts && step === MLP && i === n - 1 && l === facts.mlp;
                return <rect key={p} x={lx(i) - 6} y={by(l) + 18} width={12} height={9} rx={2}
                  className={picked ? "ov-mlp ov-mlp-pick ov-seq" : "ov-mlp ov-seq"} style={{ ...at(layerAt(l) + 110), fillOpacity: p === 0 ? 0.12 : 0.15 + 0.85 * Math.sqrt(a) }} />;
              })}
              {facts && step === MLP && <text x={lastX - 12} y={by(facts.mlp) + 26} textAnchor="end" className="ov-note ov-note-mlp">MLP {facts.mlp}</text>}
            </g>
          </g>
        ))}
        {/* Each layer's strongest move into the last token. */}
        <g className={cls([ATTN], [STREAM])} {...go(ATTN)}>
          <title>Attention: information moving from earlier tokens into the last one</title>
          {facts?.arcs.map((a) => {
            const xs = lx(laneOf(a.from));
            const y = by(a.layer) + 12;
            const rise = Math.min(14, 4 + (lastX - xs) * 0.04);
            const picked = step === ATTN && a.layer === facts.head?.layer && a.head === facts.head.head;
            return (
              <g key={`${a.layer}.${a.head}`}>
                <path d={`M${xs} ${y} Q${(xs + lastX) / 2} ${y - 2 * rise} ${lastX - 1} ${y - 1}`} pathLength={1} className={`ov-arc ov-draw${picked ? " ov-arc-pick" : ""}`}
                  style={{ ...at(layerAt(a.layer)), strokeWidth: picked ? 3 : 0.8 + 2 * (a.size / maxArc) }} markerEnd="url(#ov-arrow)" />
                {picked && <text x={(xs + lastX) / 2} y={y - rise - 4} textAnchor="middle" className="ov-note">head {a.layer}.{a.head}</text>}
              </g>
            );
          })}
        </g>

        {/* The lens down the right: the guess so far, at the last token. */}
        <g className={cls([STREAM, GUESS])} {...go(STREAM)}>
          <title>The guess if the model stopped here (the logit lens)</title>
          <text x={x1 + 14} y={EMB_Y - 18} className="ov-label">{narrow ? "guess" : "guess so far"}</text>
          {facts && Array.from({ length: L + 1 }, (_, k) => (
            <text key={k} x={x1 + 14} y={k === 0 ? EMB_Y + 4 : by(k - 1) + 16} style={at(k === 0 ? FILM.numbers : layerAt(k - 1) + 170)}
              className={`ov-seq ${k === L ? "ov-lens ov-lens-final" : k >= facts.settle ? "ov-lens ov-lens-settled" : "ov-lens"}`}>
              {lensText(k)}<title>{k === 0 ? "Before layer 0" : `After layer ${k - 1}`}: {show(tokenizer.piece(facts.lens[k]))}</title>
            </text>
          ))}
        </g>

        {/* The guess. */}
        <g className={cls([GUESS])} {...go(GUESS)}>
          <title>The last token's numbers, scored against every token</title>
          {!narrow && <text x={LX} y={OUT_Y + 22} textAnchor="end" className="ov-label">the guess</text>}
          <g className="ov-seq" style={at(FILM.guess)}>
            <rect x={x0 - 4} y={OUT_Y} width={x1 - x0 + 8} height={OUT_H} rx={10} className="ov-out" />
            <text x={x0 + 10} y={OUT_Y + 20} className="ov-label">scored against all 50,257 tokens</text>
            {guesses?.slice(0, 3).map((g, k) => {
              const bw = Math.max(2, (x1 - x0 - 170) * g.p);
              return (
                <g key={k} transform={`translate(${x0 + 10} ${OUT_Y + 38 + k * 18})`}>
                  <text y={4} className={k === 0 ? "ov-guess ov-guess-top" : "ov-guess"}>{fit(show(g.piece), 14)}</text>
                  <rect x={110} y={-5} width={bw} height={9} rx={2} className="ov-guess-bar ov-grow" style={{ ...at(FILM.guess + 150 + k * 80), opacity: k === 0 ? 0.9 : 0.4 }} />
                  <text x={116 + bw} y={4} className="ov-label">{pct(g.p)}</text>
                </g>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
}

function StepText({ step, trace, facts, guesses, tokenizer, writer, watch }: {
  step: number; trace: Trace | null; facts: Facts | null; guesses: Guesses | null; tokenizer: Tokenizer; writer: Writer; watch: () => void;
}) {
  if (!trace || !facts || !guesses) return <p className="muted">Running GPT-2 on your prompt…</p>;
  const T = trace.tokens.length;
  const tok = (p: number) => `“${p === 0 ? "⟨start⟩" : word(tokenizer.piece(trace.tokens[p]))}”`;
  const at = tok(facts.last);
  const answer = `“${word(guesses[0].piece)}”`;
  const rival = `“${word(tokenizer.piece(facts.rival))}”`;
  const head = facts.head;
  const id = head ? `${head.layer}.${head.head}` : null;
  const found = id ? label(id) : null;
  const why = (text: ReactNode) => <div className="ov-why"><span className="card-label">Why it matters</span><p>{text}</p></div>;
  const links = (...l: [string, string][]) => <p className="ov-links">{l.map(([text, to]) => <a key={to} href={to}>{text} →</a>)}</p>;
  const unplug = () => {
    setUrlParam("off", id ?? `m${facts.mlp}`);
    navigate(href({ view: "lab", tool: "unplug" }));
  };
  const settleText = facts.settle === 0
    ? `It's already ${answer} before layer 0, from the token's own numbers.`
    : facts.settle === facts.lens.length - 1
      ? `It only lands on ${answer} at the very last layer.`
      : `It settles on ${answer} after layer ${facts.settle - 1} and stays there.`;
  const text: Record<number, ReactNode> = {
    [WHOLE]: <>
      <ol className="ov-summary">
        <li>Your prompt is cut into <Term tip={GLOSSARY.token}>tokens</Term>, and each token becomes 768 numbers.</li>
        <li>The numbers pass through 12 <Term tip={GLOSSARY.layer}>layers</Term>. In each, <span className="ov-c-attn">attention</span> moves information between tokens, then an <span className="ov-c-mlp">MLP</span> works on each token alone.</li>
        <li>Every part adds to the numbers instead of replacing them.</li>
        <li>The last token's numbers are scored against every token GPT-2 knows. Its guess here: {answer}.</li>
      </ol>
      <button className="secondary ov-watch" onClick={watch}>▶ Watch it run again</button>
      <div className="ov-deeper">
        <span className="card-label">Or go straight to a part</span>
        <div>
          <a href={href({ view: "head" })}>Heads</a>
          <a href={href({ view: "mlp" })}>MLPs</a>
          <a href={href({ view: "layers" })}>Layers</a>
          <a href={href({ view: "features" })}>Features</a>
          <a href={href({ view: "weights" })}>Weights</a>
          <a href={href({ view: "lab" })}>Lab</a>
        </div>
      </div>
    </>,
    [TOKENS]: <>
      <p>GPT-2 doesn't read letters or sentences. It reads tokens: common words whole, rarer ones in pieces. “featurama” is fe · atur · ama.</p>
      {T === 1 ? (
        <p>Your prompt is empty, so GPT-2 sees only the <Term tip={GLOSSARY.start}>start token</Term> it always gets first. Its guesses are tokens that often begin a text. Type something to give it more lanes.</p>
      ) : (
        <p>Your prompt is {T - 1} token{T === 2 ? "" : "s"}, after a <Term tip={GLOSSARY.start}>start token</Term> GPT-2 always gets first. Each token gets its own lane and keeps it all the way down.</p>
      )}
      {why("GPT-2 can only ever answer with one of its 50,257 tokens, one at a time. And it never sees letters, which is one reason language models are bad at spelling games.")}
      <p className="ov-try"><strong>Try it:</strong> type a made-up word and watch it split in the box above.</p>
    </>,
    [NUMBERS]: <>
      <p>Each token is looked up in a table with a row for each of the 50,257 tokens GPT-2 knows, and becomes 768 numbers. A second table adds where it sits in the prompt.</p>
      <p>{at} starts as {[...facts.embed(facts.last).subarray(0, 3)].map((v) => v.toFixed(2).replace("-", "−")).join(", ")}, … The strips show the first 16, blue for positive and red for negative.</p>
      {why("These numbers are all the later parts ever see. Tokens with related meanings start out with similar numbers: “ Monday” and “ Tuesday” are far closer than two random tokens.")}
      {links(["See which tokens have similar numbers", href({ view: "layers", mode: "map" })], ["See the table, weight by weight", href({ view: "weights" })])}
    </>,
    [ATTN]: <>
      <p>Each layer starts with 12 <span className="ov-c-attn">attention heads</span>, 144 in all. A head at one token looks back at earlier tokens and copies some of what it finds.</p>
      {head ? (
        <p>The arrows show the biggest move into {at} in each layer. The bold one is head {id}: of all 144, it pushes hardest toward {answer} over {rival}. It puts {pct(head.share)} of its attention on {tok(head.from)}.{found ? ` Researchers sum it up as “${found.replace(/\.$/, "")}.”` : ""}</p>
      ) : (
        <p>With only one token there's nothing earlier to look at. Type a few more words and arrows will appear.</p>
      )}
      {why(<>Heads are the only parts that move information between tokens. Without them, {at} would be read alone, with no idea what came before it.</>)}
      {links([id ? `Open head ${id}` : "Open a head", href({ view: "head", head: id ?? "9.9" })])}
    </>,
    [MLP]: <>
      <p>After its heads, each layer runs an <span className="ov-c-mlp">MLP</span>, a small network of 3,072 neurons. It works on each token by itself and never looks at the others.</p>
      <p>Brighter squares add more. Of the 12 MLPs, layer {facts.mlp}'s pushes hardest toward {answer} over {rival} at {at}.</p>
      {why("Attention brings the context in; the MLPs then work with it, one token at a time. Much of what GPT-2 knows, such as facts and word associations, seems to be stored in them.")}
      {links([`Open layer ${facts.mlp}'s MLP`, href({ view: "mlp", arg: String(facts.mlp) })])}
    </>,
    [STREAM]: <>
      <p>No part replaces a token's numbers. Every head and MLP adds its output on top, so each lane is a running total, called the <Term tip={GLOSSARY.residual}>residual stream</Term>.</p>
      <p>So you can read a guess out of the last lane after any layer, as if the model stopped there. That's the right-hand column. {settleText}</p>
      {why("Because the parts add up, you can take one out and see what changes. That's a big part of how researchers work out what each part does, and it's what the Lab is for.")}
      <div className="card-buttons">
        <button className="secondary" onClick={unplug}>Try it: unplug {id ? `head ${id}` : `MLP ${facts.mlp}`}</button>
      </div>
      {links(["Watch the guess take shape", href({ view: "layers" })], ["Find the concepts hidden in these numbers", href({ view: "features" })])}
    </>,
    [GUESS]: <>
      <p>After layer 11, the last lane's 768 numbers are scored against all 50,257 tokens, and the scores become chances: {guesses.slice(0, 3).map((g) => `“${word(g.piece)}” ${pct(g.p)}`).join(", ")}.</p>
      {why("That's all writing is: pick a token, add it to the prompt, and run the whole path again. Every reply from a chatbot is built this way, one token at a time.")}
      <div className="card-buttons">
        <button className="secondary" onClick={writer.toggle}>{writer.writing ? "■ Stop writing" : "▶ Watch it write"}</button>
      </div>
      <div className="ov-next">
        <span className="card-label">Where next</span>
        <a href={href({ view: "machine" })}><strong>Machine</strong> · every part at once, and how each one voted</a>
        <a href={href({ view: "lab" })}><strong>Lab</strong> · unplug, swap and steer parts</a>
        <a href={href({ view: "play" })}><strong>Puzzles</strong> · use what you've learned to bend GPT-2</a>
      </div>
    </>,
  };
  return <div className="ov-text">{text[step]}</div>;
}
