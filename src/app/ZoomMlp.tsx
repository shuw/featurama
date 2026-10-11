// Inside one MLP: its 3,072 neurons at a token, what each one fires on, and
// one neuron worked out down to single numbers.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { DirectEffect } from "./DirectEffect.tsx";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { href } from "./route.ts";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { gelu, neuronData, neuronDirection, neuronInput, neuronsAt, type NeuronInfo } from "./zoom.ts";
import { cssRgb, fmt, LayerStepper, Numbers, Section, TokenStrip, useThemeVersion } from "./zoom-ui.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  layer: number;
  neuron: number | null;
  onPick: (layer: number, neuron: number | null) => void;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

const COLS = 64; // 3,072 neurons as 48 rows of 64

export function ZoomMlp({ engine, tokenizer, weights, layer, neuron, onPick, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, d_mlp: N } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { neurons: true, resid: true, residMid: true }, lens: { positions: [t.length - 1], topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run);
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const [query, setQuery] = useState<number | null>(null);
  const q = query === null || query >= shown.length ? shown.length - 1 : query;
  const [info, setInfo] = useState<NeuronInfo[] | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    setInfo(null);
    neuronData(layer).then((d) => live && setInfo(d)).catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [layer]);

  const acts = useMemo(() => (trace ? neuronsAt(trace, engine.config, layer, q) : null), [trace, layer, q, engine]);
  const top = useMemo(() => {
    if (!acts) return [];
    return Array.from(acts, (v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 10);
  }, [acts]);
  const firing = acts ? Array.from(acts).filter((v) => v > 0.5).length : 0;
  const at = q === 0 ? "⟨start⟩" : word(tokenizer.piece(shown[q]));
  const picked = neuron ?? top[0]?.[1] ?? null;
  const along = picked !== null && trace ? shown.map((_, t) => neuronsAt(trace, engine.config, layer, t)[picked]) : null;
  const alongMax = along ? Math.max(1e-6, ...along.slice(1)) : 1;
  // Byte pieces of longer characters print as �; leave them out.
  const hint = (n: number) => (info ? info[n].firing.map((t) => tokenizer.piece(t)).filter((p) => !p.includes("\uFFFD")).slice(0, 3).map(show).join(" ") : "");

  return (
    <>
      <PromptBox writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current && trace?.lens ? predictions(trace, L, tokenizer) : null} stale={!current} />

      <div className="zoom-controls">
        <LayerStepper layer={layer} onChange={(l) => onPick(l, null)} L={L} />
        <p className="muted small zoom-gist"><strong>MLP {layer}</strong> · 768 numbers in, 3,072 <Term tip="Each neuron reads the token's 768 numbers through its own 768 weights, adds them up, and passes the total through a curve called GELU that keeps positives and flattens negatives to about zero.">neurons</Term> inside, 768 out</p>
      </div>
      <div className="zoom-field">
        <span className="card-label">At token</span>
        <TokenStrip tokens={shown} tokenizer={tokenizer} value={q} onChange={setQuery} allowStart label="Choose a token"
          tint={(t) => (along && t > 0 && along[t] > 0 ? `rgba(var(--mlp-rgb), ${(0.12 + 0.88 * Math.min(1, along[t] / alongMax)).toFixed(2)})` : undefined)} />
        {along && picked !== null && <p className="muted small">Tinted by neuron {layer}.{picked} on each token.</p>}
      </div>

      <div className="features">
        <div className="features-main">
          <Section title={<>All 3,072 neurons at “{at}”</>}>
            {acts ? (
              <>
                <NeuronGrid acts={acts} picked={picked} onPick={(n) => onPick(layer, n)} />
                <p className="muted small">Each square is one neuron; brighter means it fires harder. {firing} of {N.toLocaleString()} fire noticeably here ({((firing / N) * 100).toFixed(1)}%). Click one to look inside it.</p>
              </>
            ) : <p className="muted">Running…</p>}
          </Section>
          <Section title={<>Strongest at “{at}”</>}>
            <ul className="feat-list">
              {top.map(([v, n]) => (
                <li key={n}>
                  <button className={picked === n ? "feat-row feat-row-on mlp-row" : "feat-row mlp-row"} onClick={() => onPick(layer, n)}>
                    <span className="feat-bar"><span style={{ width: `${Math.min(100, (100 * v) / Math.max(info?.[n].top ?? v, v))}%` }} /></span>
                    <span className="feat-name">{layer}.{n} <span className="feat-hint">{info ? `fires on ${hint(n)}` : ""}</span></span>
                    <span className="feat-id">{fmt(v, 1)}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="muted small">Bars compare each neuron with its strongest firing on Wikipedia.</p>
          </Section>
          <Section title="Neurons are messy">
            <p>You might hope each neuron means one thing. Some do, but many fire on several unrelated things: open a few and compare their passages. GPT-2 stores more ideas than it has neurons by sharing them, which researchers call <Term tip="Packing more concepts than there are neurons by giving each concept a direction made of many neurons at once. Sparse autoencoders try to undo it.">superposition</Term>. That's why the <a href={href({ view: "features" })}>Features page</a> uses a sparse autoencoder: it splits the numbers into directions that are much more often about one thing.</p>
          </Section>
        </div>
        <aside className="features-side">
          {picked !== null && trace && acts ? (
            <NeuronCard key={`${layer}.${picked}`} layer={layer} n={picked} act={acts[picked]} info={info?.[picked] ?? null} failed={failed} trace={trace} weights={weights} tokenizer={tokenizer} position={q} at={at} />
          ) : (
            <div className="card"><p className="muted">Pick a neuron.</p></div>
          )}
        </aside>
      </div>
    </>
  );
}

function NeuronGrid({ acts, picked, onPick }: { acts: Float32Array; picked: number | null; onPick: (n: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const rows = Math.ceil(acts.length / COLS);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth;
    const cell = w / COLS;
    const h = cell * rows;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.height = `${h}px`;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const mlp = cssRgb("--mlp-rgb");
    const max = Math.max(1e-6, ...acts);
    for (let i = 0; i < acts.length; i++) {
      const x = (i % COLS) * cell;
      const y = Math.floor(i / COLS) * cell;
      const v = Math.max(0, acts[i]) / max;
      g.fillStyle = `rgba(${mlp}, ${(0.05 + 0.95 * Math.sqrt(v)).toFixed(3)})`;
      g.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
    }
    for (const [n, width] of [[picked, 2], [hover, 1.5]] as const) {
      if (n === null) continue;
      g.strokeStyle = cssRgb("--ink");
      g.lineWidth = width;
      g.strokeRect((n % COLS) * cell, Math.floor(n / COLS) * cell, cell, cell);
    }
  }, [acts, picked, hover, theme, rows]);
  const at = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const cell = r.width / COLS;
    const n = Math.floor((e.clientY - r.top) / cell) * COLS + Math.floor((e.clientX - r.left) / cell);
    return n >= 0 && n < acts.length ? n : null;
  };
  return (
    <div className="neuron-grid-wrap">
      <canvas ref={ref} className="neuron-grid" role="img" aria-label="Every neuron's activation" onMouseMove={(e) => setHover(at(e))} onMouseLeave={() => setHover(null)}
        onClick={(e) => { const n = at(e); if (n !== null) onPick(n); }} />
      <div className="neuron-hover small">{hover !== null ? `neuron ${hover}: ${fmt(acts[hover])}` : " "}</div>
    </div>
  );
}

function NeuronCard({ layer, n, act, info, failed, trace, weights, tokenizer, position, at }: {
  layer: number; n: number; act: number; info: NeuronInfo | null; failed: boolean; trace: Trace; weights: Weights; tokenizer: Tokenizer; position: number; at: string;
}) {
  const input = useMemo(() => neuronInput(trace, weights, layer, position, n), [trace, weights, layer, position, n]);
  const dir = useMemo(() => neuronDirection(weights, layer, n), [weights, layer, n]);
  const written = useMemo(() => dir.map((x) => x * act), [dir, act]);
  const vector = useCallback(() => written, [written]);
  const biggest = useMemo(() => Array.from(input.products, (v, i) => [v, i] as const).sort((a, b) => Math.abs(b[0]) - Math.abs(a[0])).slice(0, 3), [input]);
  return (
    <section className="card head-card">
      <div className="head-name">Neuron {layer}.{n}</div>
      {info && <p className="muted small">Fires on {info.perMillion >= 10000 ? `${(info.perMillion / 10000).toFixed(1)}% of` : `${info.perMillion.toLocaleString()} in a million`} Wikipedia tokens; strongest {fmt(info.top, 1)}.</p>}
      <div className="card-label zoom-sub">Worked out at “{at}”</div>
      <div className="one-grid">
        <span className="card-label">768 × weight</span><Numbers values={input.products} label="Each input number times its weight" height={18} />
      </div>
      <p className="one-sum small">
        {biggest.map(([v], i) => <span key={i}>{i ? (v < 0 ? " − " : " + ") : ""}{fmt(i ? Math.abs(v) : v)}</span>)} + … (768 in all) {input.bias < 0 ? "−" : "+"} bias {fmt(Math.abs(input.bias))} = <strong>{fmt(input.pre)}</strong>
      </p>
      <p className="one-sum small">GELU({fmt(input.pre)}) = <strong>{fmt(gelu(input.pre))}</strong> <span className="muted">{gelu(input.pre) < 0.05 ? "(flattened: it stays quiet)" : "(passed through)"}</span></p>
      <p className="muted small">The inputs are the token's 768 numbers after LayerNorm, each times this neuron's weight for it. GELU keeps positives and turns negatives to almost zero.</p>
      {act > 0.05 ? (
        <DirectEffect trace={trace} weights={weights} tokenizer={tokenizer} vector={vector} position={position} />
      ) : (
        <p className="muted small">Quiet here, so it adds almost nothing at “{at}”.</p>
      )}
      <div className="examples">
        <div className="card-label">Where it fires hardest on Wikipedia</div>
        {failed && <p className="muted small">Couldn't load the passages.</p>}
        {!info && !failed && <p className="muted small">Loading…</p>}
        {info?.passages.map((p, i) => (
          <p key={i} className="example-text">
            {p.ids.map((t, j) => (
              <span key={j} className={j === p.at ? "ex-tok ex-here" : "ex-tok"} style={p.acts[j] > 12 ? { background: `rgba(var(--mlp-rgb), ${(0.1 + 0.8 * (p.acts[j] / 255)).toFixed(2)})` } : undefined}>{tokenizer.piece(t)}</span>
            ))}
          </p>
        ))}
      </div>
    </section>
  );
}
