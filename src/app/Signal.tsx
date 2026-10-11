// The Signal view: one line per token, attention arcs into the chosen token,
// MLP writes as dots, and the logit lens along the bottom.

import { useState } from "react";
import { readingLabel, show, type Analysis } from "./analysis.ts";

const LABEL_W = 132;
const ATTN_W = 52;
const MLP_W = 26;
const COL_W = ATTN_W + MLP_W;
const OUT_W = 150;
const TOP = 30;
const LENS_H = 52;

interface Props {
  layers: number;
  pieces: string[];
  analysis: Analysis | null; // null while the learner is still predicting
  onSelect: (position: number) => void;
}

export function Signal({ layers, pieces, analysis, onSelect }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const T = pieces.length;
  // A fixed line height up to 16 tokens, so typing adds lines instead of rescaling the drawing.
  const laneH = T <= 16 ? 26 : Math.max(14, 416 / T);
  const width = LABEL_W + layers * COL_W + 16 + OUT_W;
  const lanesBottom = TOP + T * laneH;
  const lensTop = lanesBottom + 26;
  const height = lensTop + LENS_H + 22;
  const y = (t: number) => TOP + laneH * (t + 0.5);
  const colX = (l: number) => LABEL_W + l * COL_W;
  const pos = analysis?.position ?? T - 1;
  const lensX = (k: number) => (k === 0 ? 8 : colX(k - 1) + 3);
  const lensW = (k: number) => (k === 0 ? LABEL_W - 16 : COL_W - 6);
  const hovered = hover !== null && analysis ? analysis.lens[hover] : null;

  return (
    <div className="signal">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe(pieces, analysis)}>
        {Array.from({ length: layers }, (_, l) => (
          <g key={l}>
            <text x={colX(l) + 2} y={16} className="axis">{l}</text>
            <line x1={colX(l)} x2={colX(l) + COL_W - 6} y1={22} y2={22} className="rule" />
          </g>
        ))}
        <text x={colX(layers) + 18} y={16} className="axis">next token</text>

        {pieces.map((_, t) => (
          <line key={t} x1={LABEL_W - 4} x2={colX(layers) + 6} y1={y(t)} y2={y(t)} className={t === pos ? "lane lane-on" : "lane"} />
        ))}
        <line x1={colX(layers) + 6} x2={colX(layers) + 6} y1={TOP} y2={lanesBottom} className="final-norm" />

        {analysis?.arcs.map((a, i) => {
          const x0 = colX(a.layer) + 2;
          const x1 = x0 + ATTN_W - 4;
          return (
            <path key={i} d={`M${x0} ${y(a.from)} C ${x0 + 30} ${y(a.from)}, ${x1 - 30} ${y(pos)}, ${x1} ${y(pos)}`}
              className={a.from === 0 ? "arc arc-start" : "arc"} strokeWidth={1 + 5 * a.weight} opacity={a.from === 0 ? 0.5 : 0.3 + 0.7 * a.weight} />
          );
        })}

        {analysis?.mlp.map((row, l) =>
          row.map((v, t) => (
            <circle key={`${l}-${t}`} cx={colX(l) + ATTN_W + MLP_W / 2} cy={y(t)} r={1.5 + (laneH / 2.6) * Math.sqrt(v)} className={t === pos ? "mlp mlp-on" : "mlp"} />
          )),
        )}

        {pieces.map((p, t) => (
          <g key={t} className="token" onClick={() => analysis && onSelect(t)} role="button" tabIndex={analysis ? 0 : -1}
            aria-label={`Look at token ${t}: ${show(p)}`} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && analysis && onSelect(t)}>
            <rect x={0} y={y(t) - laneH / 2} width={LABEL_W - 8} height={laneH} className="hit" />
            <text x={LABEL_W - 10} y={y(t) + 4} textAnchor="end" className={t === pos ? "piece piece-on" : t === 0 ? "piece piece-start" : "piece"}>{show(p)}</text>
          </g>
        ))}

        {analysis?.finals.map((f, t) => (
          <text key={t} x={colX(layers) + 18} y={y(t) + 4} className={t === pos ? "out out-on" : "out"}>
            {t === pos ? `${show(f)} ${pct(analysis.lens[layers].p)}` : show(f)}
          </text>
        ))}

        <text x={8} y={lensTop - 8} className="axis">logit lens on “{show(pieces[pos])}”</text>
        {Array.from({ length: layers + 1 }, (_, k) => {
          const r = analysis?.lens[k];
          const long = r ? show(r.piece).length > 8 : false;
          return (
            <g key={k} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
              <rect x={lensX(k)} y={lensTop} width={lensW(k)} height={LENS_H} className={r && r.piece === analysis!.final ? "cell cell-final" : "cell"} />
              {r && <rect x={lensX(k)} y={lensTop + LENS_H * (1 - r.p)} width={lensW(k)} height={LENS_H * r.p} className="cell-fill" />}
              {r && r.finalP > 0.001 && <rect x={lensX(k)} y={lensTop + LENS_H - 4} width={lensW(k) * Math.min(1, r.finalP)} height={4} className="cell-bar" />}
              <text x={lensX(k) + 5} y={lensTop + 19} className={long ? "cell-text cell-text-long" : "cell-text"}>{r ? clip(show(r.piece)) : "?"}</text>
              {r && <text x={lensX(k) + 5} y={lensTop + 36} className="cell-pct">{pct(r.p)}</text>}
              <text x={lensX(k) + lensW(k) / 2} y={lensTop + LENS_H + 14} textAnchor="middle" className="axis">{k === 0 ? "start" : k - 1}</text>
            </g>
          );
        })}
      </svg>
      {hovered && (
        <div className="lens-pop" role="status" style={{ left: `${(Math.min(lensX(hover!), width - 230) / width) * 100}%`, top: `${(lensTop / height) * 100}%` }}>
          <div className="lens-pop-title">{readingLabel(hover!)}</div>
          {hovered.top.map((c, i) => (
            <div key={i} className="lens-pop-row"><span>{show(c.piece)}</span><span>{pct(c.p)}</span></div>
          ))}
          <div className="lens-pop-final">“{show(analysis!.final)}” here: {pct(hovered.finalP)}</div>
        </div>
      )}
    </div>
  );
}

const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
const clip = (s: string) => (s.length > 11 ? s.slice(0, 10) + "…" : s);

function describe(pieces: string[], a: Analysis | null) {
  if (!a) return `GPT-2's ${pieces.length} token lines, not yet run.`;
  return `Logit lens on “${a.pieces[a.position]}”: ` + a.lens.map((r, k) => `${readingLabel(k)} ${r.piece} ${pct(r.p)}`).join("; ");
}
