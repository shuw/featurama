// The residual stream under a microscope: every one of a token's 768
// numbers, before the first layer and after each one.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, readingLabel, show, word } from "./analysis.ts";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { cssRgb, diverging, fmt, Section, TokenStrip, useThemeVersion } from "./inside-ui.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

type Mode = "layers" | "tokens";

export function Stream({ engine, tokenizer, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, d_model: d } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { resid: true }, lens: { positions: [t.length - 1], topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run);
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const T = shown.length;
  const [query, setQuery] = useState<number | null>(null);
  const q = query === null || query >= T ? T - 1 : query;
  const [mode, setMode] = useState<Mode>("layers");
  const [layer, setLayer] = useState(Math.ceil(L / 2));
  const [dim, setDim] = useState<number | null>(null);
  const [hover, setHover] = useState<{ row: number; dim: number } | null>(null);
  const pieces = shown.map((t, i) => (i === 0 ? "⟨start⟩" : show(tokenizer.piece(t))));

  const at = (r: number, t: number) => trace!.resid!.subarray((r * T + t) * d, (r * T + t + 1) * d);
  const rows = useMemo(() => {
    if (!trace) return [];
    return mode === "layers" ? Array.from({ length: L + 1 }, (_, r) => at(r, q)) : Array.from({ length: T }, (_, t) => at(layer, t));
  }, [trace, mode, q, layer, L, T]);
  const rowLabel = (r: number) => (mode === "layers" ? readingLabel(r) : pieces[r]);
  const norms = rows.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));

  // The few numbers far bigger than the rest, at this token after the last layer.
  const giants = useMemo(() => {
    if (!trace) return [];
    const v = at(L, q);
    const sorted = Array.from(v, (x, i) => [Math.abs(x), i] as const).sort((a, b) => b[0] - a[0]);
    const typical = sorted[Math.floor(sorted.length / 2)][0];
    return sorted.slice(0, 3).map(([a, i]) => ({ dim: i, value: v[i], times: a / Math.max(typical, 1e-6) }));
  }, [trace, q, L]);
  const size = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const startTimes = trace && T > 1 ? size(at(layer, 0)) / (Array.from({ length: T - 1 }, (_, t) => size(at(layer, t + 1))).reduce((a, b) => a + b, 0) / (T - 1)) : 0;

  return (
    <>
      <PromptBox writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current && trace?.lens ? predictions(trace, L, tokenizer) : null} stale={!current} />

      <div className="part-controls">
        <div className="seg" role="radiogroup" aria-label="What to compare">
          <button role="radio" aria-checked={mode === "layers"} className={mode === "layers" ? "seg-on" : ""} onClick={() => setMode("layers")}>One token, every layer</button>
          <button role="radio" aria-checked={mode === "tokens"} className={mode === "tokens" ? "seg-on" : ""} onClick={() => setMode("tokens")}>Every token, one layer</button>
        </div>
      </div>
      {mode === "layers" ? (
        <div className="part-field">
          <span className="card-label">Token</span>
          <TokenStrip tokens={shown} tokenizer={tokenizer} value={q} onChange={setQuery} allowStart label="Choose a token" />
        </div>
      ) : (
        <div className="part-field">
          <span className="card-label">Reading</span>
          <div className="seg seg-wrap" role="radiogroup" aria-label="Which layer">
            {Array.from({ length: L + 1 }, (_, r) => (
              <button key={r} role="radio" aria-checked={layer === r} className={layer === r ? "seg-on" : ""} onClick={() => setLayer(r)}>{r === 0 ? "start" : r - 1}</button>
            ))}
          </div>
          <p className="muted small">{readingLabel(layer)}</p>
        </div>
      )}

      {trace && (
        <Section title={mode === "layers" ? <>“{word(pieces[q])}”: 768 numbers, layer by layer</> : <>Every token {readingLabel(layer)}</>}>
          <p className="muted">Each row is the token's <Term tip="Each token's running summary: 768 numbers that every layer reads and adds to.">running summary</Term>; each thin column is one of its 768 numbers. Blue is positive, red negative. Colours are scaled per row, so a few huge numbers show as solid stripes. The bar on the right is the row's overall size.</p>
          <StreamCanvas rows={rows} labels={rows.map((_, r) => rowLabel(r))} norms={norms} dim={dim} onDim={setDim} onHover={setHover} />
          <p className="stream-hover small">{hover ? `${rowLabel(hover.row)} · number ${hover.dim}: ${fmt(rows[hover.row][hover.dim])}` : " "}</p>
          <div className="stream-notes">
            {mode === "layers" && giants.length > 0 && (
              <p>At the end, number{" "}
                {giants.map((g, i) => <span key={g.dim}>{i ? (i === giants.length - 1 ? " and " : ", ") : ""}<button className="link-small inline" onClick={() => setDim(g.dim)}>{g.dim}</button></span>)}{" "}
                are up to {Math.round(giants[0].times)}× bigger than a typical one. Researchers call these massive activations: the same few dimensions are huge for almost every token, so they work more like a fixed setting than like information about your text.</p>
            )}
            {mode === "tokens" && T > 1 && startTimes > 1.5 && (
              <p>⟨start⟩'s numbers are about {Math.round(startTimes)}× bigger than the others'. Researchers have linked this to its role as a parking spot for attention (an attention sink).</p>
            )}
            {mode === "layers" && norms.length > 1 && q > 0 && <p>The summary grows from size {fmt(norms[0], 0)} to {fmt(norms[norms.length - 1], 0)} as layers add to it. Layers only ever add; nothing is overwritten.</p>}
          </div>
        </Section>
      )}

      {trace && dim !== null && <DimCard trace={trace} dim={dim} L={L} d={d} pieces={pieces} q={q} onClose={() => setDim(null)} />}
    </>
  );
}

function StreamCanvas({ rows, labels, norms, dim, onDim, onHover }: {
  rows: Float32Array[]; labels: string[]; norms: number[]; dim: number | null; onDim: (d: number) => void; onHover: (h: { row: number; dim: number } | null) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const ROW = 20;
  const LABEL = 116;
  const BAR = 70;
  useEffect(() => {
    const c = ref.current;
    if (!c || !rows.length) return;
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = rows.length * ROW;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.height = `${h}px`;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const pos = cssRgb("--pos-rgb");
    const neg = cssRgb("--neg-rgb");
    const ink = cssRgb("--ink");
    const muted = cssRgb("--muted");
    const plot = w - LABEL - BAR - 8;
    const n = rows[0].length;
    const cw = plot / n;
    const maxNorm = Math.max(...norms);
    g.font = '11px "IBM Plex Mono", monospace';
    g.textBaseline = "middle";
    rows.forEach((v, r) => {
      const y = r * ROW;
      // Scale by the row's 99th-percentile size, so ordinary numbers stay visible next to giants.
      const sorted = Array.from(v, Math.abs).sort((a, b) => a - b);
      const scale = sorted[Math.floor(sorted.length * 0.99)] || 1;
      for (let i = 0; i < n; i++) {
        g.fillStyle = diverging(v[i], scale, pos, neg);
        g.fillRect(LABEL + i * cw, y + 2, Math.max(cw, 0.6), ROW - 4);
      }
      g.fillStyle = muted;
      g.textAlign = "right";
      g.fillText(labels[r].length > 15 ? labels[r].slice(0, 14) + "…" : labels[r], LABEL - 8, y + ROW / 2);
      g.fillStyle = ink;
      g.globalAlpha = 0.25;
      g.fillRect(w - BAR, y + 6, (BAR * norms[r]) / maxNorm, ROW - 12);
      g.globalAlpha = 1;
    });
    if (dim !== null) {
      g.strokeStyle = ink;
      g.lineWidth = 1.5;
      g.strokeRect(LABEL + dim * cw - 1, 0.75, cw + 2, h - 1.5);
    }
  }, [rows, labels, norms, dim, theme]);
  const at = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const plot = r.width - LABEL - BAR - 8;
    const x = e.clientX - r.left - LABEL;
    const row = Math.floor((e.clientY - r.top) / ROW);
    if (x < 0 || x > plot || row < 0 || row >= rows.length) return null;
    return { row, dim: Math.min(rows[0].length - 1, Math.floor((x / plot) * rows[0].length)) };
  };
  return <canvas ref={ref} className="stream-canvas" role="img" aria-label="The residual stream's numbers" onMouseMove={(e) => onHover(at(e))} onMouseLeave={() => onHover(null)} onClick={(e) => { const a = at(e); if (a) onDim(a.dim); }} />;
}

// One of the 768 numbers, followed through every layer for every token.
function DimCard({ trace, dim, L, d, pieces, q, onClose }: { trace: Trace; dim: number; L: number; d: number; pieces: string[]; q: number; onClose: () => void }) {
  const T = trace.tokens.length;
  const series = pieces.map((_, t) => Array.from({ length: L + 1 }, (_, r) => trace.resid![(r * T + t) * d + dim]));
  const all = series.flat();
  const lo = Math.min(0, ...all);
  const hi = Math.max(0, ...all);
  const W = 640;
  const Hh = 200;
  const x = (r: number) => 40 + (r / L) * (W - 60);
  const y = (v: number) => 10 + (1 - (v - lo) / (hi - lo || 1)) * (Hh - 30);
  return (
    <Section title={<>Number {dim}, through the layers</>}>
      <button className="link-small" onClick={onClose}>Close</button>
      <svg viewBox={`0 0 ${W} ${Hh}`} className="dim-chart" role="img" aria-label={`Number ${dim} at every layer for every token`}>
        <line x1={40} x2={W - 20} y1={y(0)} y2={y(0)} className="dim-zero" />
        {series.map((s, t) => (
          <polyline key={t} className={t === q ? "dim-line dim-line-on" : t === 0 ? "dim-line dim-line-start" : "dim-line"} points={s.map((v, r) => `${x(r)},${y(v)}`).join(" ")} />
        ))}
        {Array.from({ length: L + 1 }, (_, r) => <text key={r} x={x(r)} y={Hh - 4} className="dim-axis" textAnchor="middle">{r === 0 ? "in" : r - 1}</text>)}
        <text x={36} y={y(hi) + 4} className="dim-axis" textAnchor="end">{fmt(hi, 0)}</text>
        <text x={36} y={y(lo) + 4} className="dim-axis" textAnchor="end">{fmt(lo, 0)}</text>
      </svg>
      <p className="muted small">One line per token; “{word(pieces[q])}” is darker{T > 1 ? ", ⟨start⟩ dashed" : ""}. The axis runs from the embedding through each layer.</p>
    </Section>
  );
}
