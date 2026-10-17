// Pictures of the weights themselves, what GPT-2 learned once and uses for
// every prompt: the position embeddings, what a head copies, and the wiring
// between heads.

import { useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show } from "./analysis.ts";
import { Term } from "./Tip.tsx";
import { copyingTable, loadWiring, positionPca, strongestSources, type Wiring } from "./inside.ts";
import { diverging, cssRgb, partGist, Section, useThemeVersion } from "./inside-ui.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
}

export function Positions({ weights }: { weights: Weights }) {
  const [pca, setPca] = useState<ReturnType<typeof positionPca> | null>(null);
  useEffect(() => {
    // A second or so of arithmetic: let the page draw first.
    const t = setTimeout(() => setPca(positionPca(weights)), 50);
    return () => clearTimeout(t);
  }, [weights]);
  const W = 420;
  const Hh = 300;
  const view = useMemo(() => {
    if (!pca) return null;
    const [a, b] = pca.coords;
    const pad = 16;
    // Position 0 sits far from the rest; fit the scale to the others.
    const xs = Array.from(a).slice(1);
    const ys = Array.from(b).slice(1);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const s = Math.min((W - 2 * pad) / (x1 - x0), (Hh - 2 * pad) / (y1 - y0));
    const X = (v: number) => pad + (v - x0) * s;
    const Y = (v: number) => Hh - pad - (v - y0) * s;
    const segs: { d: string; t: number }[] = [];
    for (let p0 = 1; p0 < a.length - 1; p0 += 32) {
      const pts: string[] = [];
      for (let p = p0; p <= Math.min(p0 + 32, a.length - 1); p++) pts.push(`${X(a[p]).toFixed(1)},${Y(b[p]).toFixed(1)}`);
      segs.push({ d: pts.join(" "), t: p0 / a.length });
    }
    const marks = [64, 256, 512, 1023].map((p) => ({ p, x: X(a[p]), y: Y(b[p]) }));
    return { segs, marks };
  }, [pca]);
  return (
    <Section title="Where am I? The position embeddings">
      <p>GPT-2 can't see word order directly, so before layer 0 it adds a position vector to each token: one of 1,024 learned lists of 768 numbers. Nobody told it what they should look like. Squashed down to the two directions they vary along most, they trace a smooth curve, so nearby positions get similar vectors.</p>
      {!pca || !view ? <p className="muted">Working it out…</p> : (
        <div className="pos-wrap">
          <svg viewBox={`0 0 ${W} ${Hh}`} className="pos-chart" role="img" aria-label="Position embeddings on their top two directions">
            {view.segs.map((s, i) => <polyline key={i} points={s.d} className="pos-line" style={{ stroke: `hsl(${220 - 200 * s.t} 80% 50%)` }} />)}
            {view.marks.map((m) => (
              <g key={m.p}>
                <circle cx={m.x} cy={m.y} r={3.5} className="pos-dot" />
                <text x={m.x + 6} y={m.y - 6} className="pos-label">{m.p}</text>
              </g>
            ))}
          </svg>
          <div className="pos-side">
            {pca.coords.map((c, k) => <Spark key={k} values={c} label={`Direction ${k + 1}`} share={pca.explained[k]} />)}
            <p className="muted small">Each direction plotted against position 0 to 1,023. They look like slow waves, a bit like the sine waves the original Transformer used on purpose. Position 0 is left off the curve: it's far from all the others.</p>
          </div>
        </div>
      )}
    </Section>
  );
}

function Spark({ values, label, share }: { values: Float32Array; label: string; share: number }) {
  const W = 260;
  const Hh = 44;
  const xs = Array.from(values).slice(1);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pts = xs.filter((_, i) => i % 4 === 0).map((v, i) => `${((i * 4) / xs.length) * W},${Hh - 4 - ((v - lo) / (hi - lo || 1)) * (Hh - 8)}`).join(" ");
  return (
    <div className="spark">
      <span className="card-label">{label} · {Math.round(share * 100)}% of the variation</span>
      <svg viewBox={`0 0 ${W} ${Hh}`} className="spark-svg" aria-hidden="true"><polyline points={pts} className="spark-line" /></svg>
    </div>
  );
}

const NAMES = [" Mary", " John", " Alice", " Bob", " Sarah", " Tom", " Paris", " London", " dog", " cat"];
const PRESETS = [
  { id: "9.9", why: "a name mover: copies the name it looks at" },
  { id: "10.7", why: "suppresses what it looks at" },
  { id: "4.11", why: "a previous-token head: moves information, doesn't copy words" },
];

// For one head; the chips compare with heads known to copy, suppress, or neither.
export function Copying({ engine, tokenizer, weights, head: start }: Props & { head: string }) {
  const [head, setHead] = useState(start);
  useEffect(() => setHead(start), [start]);
  const ids = useMemo(() => NAMES.map((n) => tokenizer.encode(n)).filter((t) => t.length === 1).map((t) => t[0]), [tokenizer]);
  const [table, setTable] = useState<{ head: string; rows: number[][] } | null>(null);
  useEffect(() => {
    let live = true;
    const [l, h] = head.split(".").map(Number);
    copyingTable(engine, weights, l, h, ids).then((rows) => live && setTable({ head, rows }));
    return () => {
      live = false;
    };
  }, [head, ids, engine, weights]);
  const diag = table ? table.rows.reduce((s, r, i) => s + r[i], 0) / table.rows.length : 0;
  const theme = useThemeVersion();
  const pos = useMemo(() => cssRgb("--pos-rgb"), [theme]);
  const neg = useMemo(() => cssRgb("--neg-rgb"), [theme]);
  return (
    <Section title="What a head copies">
      <p>A head's <Term tip="The value and output weights together: what the head writes for whatever token it looks at.">value and output weights</Term> decide what it writes about the token it looks at. Here, each row is a token the head could look at, and each column is a token it could make more likely next. Blue means more likely, red less. A blue diagonal means the head copies: look at “Mary”, push “Mary”.</p>
      <div className="hl-chips compact">
        {PRESETS.map((p) => <button key={p.id} className={head === p.id ? "hl-chip hl-on" : "hl-chip"} onClick={() => setHead(p.id)} title={p.why}>{p.id}</button>)}
        {!PRESETS.some((p) => p.id === start) && <button className={head === start ? "hl-chip hl-on" : "hl-chip"} onClick={() => setHead(start)}>{start}</button>}
      </div>
      <p className="muted small">Head {head} · {partGist(head)}</p>
      {!table || table.head !== head ? <p className="muted">Working it out…</p> : (
        <>
          <div className="copy-table" style={{ gridTemplateColumns: `72px repeat(${ids.length}, minmax(0, 1fr))` }}>
            <span />
            {ids.map((id) => <span key={id} className="copy-col">{show(tokenizer.piece(id))}</span>)}
            {table.rows.map((r, i) => (
              <div key={i} className="copy-row">
                <span className="copy-label">{show(tokenizer.piece(ids[i]))}</span>
                {r.map((v, j) => <span key={j} className={i === j ? "copy-cell copy-diag" : "copy-cell"} style={{ background: diverging(v, 0.8, pos, neg) }} title={v.toFixed(2)} />)}
              </div>
            ))}
          </div>
          <p className="muted small">{diag > 0.3 ? `Head ${head} copies: its diagonal is clearly blue.` : diag < -0.3 ? `Head ${head} does the opposite: it pushes down the token it looks at.` : `Head ${head} doesn't copy these tokens.`} Computed from the weights, with each token as GPT-2 sees it after layer 0.</p>
        </>
      )}
    </Section>
  );
}

const KIND_TEXT = { Q: "its question (query)", K: "its labels (keys)", V: "what it fetches (values)" } as const;

export function WiringMap({ sel, onSelect: setSel }: { sel: string; onSelect: (id: string) => void }) {
  const L = 12;
  const H = 12;
  const [w, setW] = useState<Wiring | null>(null);
  useEffect(() => {
    loadWiring().then(setW).catch(() => {});
  }, []);
  const C = 34;
  const R = 22;
  const pad = 26;
  const x = (l: number) => pad + l * C;
  const y = (h: number) => pad + h * R;
  const edges = useMemo(() => {
    if (!w) return [];
    const out: { from: string; to: string; kind: "Q" | "K" | "V"; strength: number }[] = [];
    for (const kind of ["Q", "K", "V"] as const) {
      const k = w.kinds[kind];
      const strength = (s: number) => Math.max(0, Math.min(1, (s / k.median - 2) / 3));
      // In: its strongest sources. Out: later heads that count it among theirs.
      for (const [src, s] of strongestSources(w, kind, sel)) out.push({ from: src, to: sel, kind, strength: strength(s) });
      for (const to of Object.keys(k.heads)) for (const [src, s] of strongestSources(w, kind, to)) if (src === sel) out.push({ from: sel, to, kind, strength: strength(s) });
    }
    return out;
  }, [w, sel]);
  const pt = (id: string) => {
    const [l, h] = id.split(".").map(Number);
    return [x(l), y(h)] as const;
  };
  return (
    <Section title="The wiring between heads">
      <p>A head can only use what earlier layers wrote. From the weights alone you can measure which earlier heads write things a later head is built to read, in {KIND_TEXT.Q}, {KIND_TEXT.K} or {KIND_TEXT.V}. Click any head to follow its wires.</p>
      {!w ? <p className="muted">Loading…</p> : (
        <div className="wire-wrap">
          <svg viewBox={`0 0 ${pad * 2 + (L - 1) * C} ${pad * 2 + (H - 1) * R}`} className="wire-svg" role="img" aria-label="Heads and the strongest wiring between them">
            {Array.from({ length: L }, (_, l) => <text key={l} x={x(l)} y={12} className="wire-axis" textAnchor="middle">{l}</text>)}
            {edges.map((e, i) => {
              const [x1, y1] = pt(e.from);
              const [x2, y2] = pt(e.to);
              const mx = (x1 + x2) / 2;
              return <path key={i} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} className={`wire wire-${e.kind}`} style={{ strokeWidth: 1 + 3 * e.strength }} />;
            })}
            {Array.from({ length: L }, (_, l) => Array.from({ length: H }, (_, h) => {
              const id = `${l}.${h}`;
              const linked = id === sel || edges.some((e) => e.from === id || e.to === id);
              return (
                <circle key={id} cx={x(l)} cy={y(h)} r={id === sel ? 7 : linked ? 5.5 : 4} className={id === sel ? "wire-node wire-sel" : linked ? "wire-node wire-linked" : "wire-node"}
                  onClick={() => setSel(id)} role="button" aria-label={`Head ${id}`} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && setSel(id)}>
                  <title>{`Head ${id}: ${partGist(id)}`}</title>
                </circle>
              );
            }))}
          </svg>
          <div className="wire-side">
            <div className="head-name">Head {sel}</div>
            <p className="muted small">{partGist(sel)}</p>
            <ul className="wire-legend">
              <li><span className="wire-key wire-Q" /> feeds {KIND_TEXT.Q}</li>
              <li><span className="wire-key wire-K" /> feeds {KIND_TEXT.K}</li>
              <li><span className="wire-key wire-V" /> feeds {KIND_TEXT.V}</li>
            </ul>
            {sel === "5.5" && <p className="small">5.5 is an induction head. Its strongest key wire comes from 4.11, a previous-token head: 4.11 tags each token with the one before it, and 5.5 searches those tags to find what followed last time.</p>}
          </div>
        </div>
      )}
      <p className="muted small">Each head's three strongest wires of each kind are drawn, if at least twice the typical strength; thicker is stronger. This is what the weights allow, not what happens on any one prompt.</p>
    </Section>
  );
}
