// Small pieces shared by the pages that look inside parts.

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show } from "./analysis.ts";
import { label, research } from "./research.ts";
import { habits } from "./atlas.ts";

// Canvas can't read CSS variables by itself; redraw when the theme flips.
export function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const o = new MutationObserver(() => setV((x) => x + 1));
    o.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => o.disconnect();
  }, []);
  return v;
}

export function cssRgb(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "0, 0, 0";
}

// Positive numbers blue, negative red, stronger for bigger.
export function diverging(v: number, max: number, pos: string, neg: string) {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  return `rgba(${v >= 0 ? pos : neg}, ${(0.06 + 0.94 * a).toFixed(3)})`;
}

// A row of numbers as coloured cells. Hover reports which one.
export function Numbers({ values, max, height = 14, onHover, label: aria, mark }: {
  values: ArrayLike<number>; max?: number; height?: number; onHover?: (i: number | null) => void; label: string; mark?: number | null;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const n = values.length;
  const scale = max ?? Math.max(1e-9, ...Array.from(values, Math.abs));
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth;
    c.width = Math.round(w * dpr);
    c.height = Math.round(height * dpr);
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const pos = cssRgb("--pos-rgb");
    const neg = cssRgb("--neg-rgb");
    const cw = w / n;
    for (let i = 0; i < n; i++) {
      g.fillStyle = diverging(values[i], scale, pos, neg);
      g.fillRect(i * cw, 0, Math.max(cw - (cw > 4 ? 1 : 0), 0.6), height);
    }
    if (mark !== null && mark !== undefined) {
      g.strokeStyle = cssRgb("--ink");
      g.lineWidth = 1.5;
      g.strokeRect(mark * cw + 0.75, 0.75, Math.max(cw - 1.5, 2), height - 1.5);
    }
  }, [values, scale, height, theme, mark]);
  return (
    <canvas ref={ref} className="numbers" style={{ height }} role="img" aria-label={aria}
      onMouseMove={onHover ? (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onHover(Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * n))));
      } : undefined}
      onMouseLeave={onHover ? () => onHover(null) : undefined} />
  );
}

export function TokenStrip({ tokens, tokenizer, value, onChange, allowStart = false, label: aria, tint }: {
  tokens: number[]; tokenizer: Tokenizer; value: number; onChange: (t: number) => void; allowStart?: boolean; label: string; tint?: (t: number) => string | undefined;
}) {
  return (
    <div className="feat-strip" role="group" aria-label={aria}>
      {tokens.map((id, t) => (
        <button key={t} className={`feat-tok${t === value ? " feat-tok-on" : ""}`} disabled={t === 0 && !allowStart} onClick={() => onChange(t)}
          aria-pressed={t === value} style={tint?.(t) ? { background: tint(t) } : undefined}>
          {t === 0 ? "⟨start⟩" : show(tokenizer.piece(id))}
        </button>
      ))}
    </div>
  );
}

export function HeadStepper({ id, onChange, L, H }: { id: string; onChange: (id: string) => void; L: number; H: number }) {
  const [l, h] = id.split(".").map(Number);
  const set = (nl: number, nh: number) => onChange(`${Math.max(0, Math.min(L - 1, nl))}.${Math.max(0, Math.min(H - 1, nh))}`);
  return (
    <div className="stepper" role="group" aria-label="Choose a head">
      <span className="stepper-label">layer</span>
      <button className="step" onClick={() => set(l - 1, h)} disabled={l === 0} aria-label="Previous layer">‹</button>
      <span className="stepper-value">{l}</span>
      <button className="step" onClick={() => set(l + 1, h)} disabled={l === L - 1} aria-label="Next layer">›</button>
      <span className="stepper-label">head</span>
      <button className="step" onClick={() => set(l, h - 1)} disabled={h === 0} aria-label="Previous head">‹</button>
      <span className="stepper-value">{h}</span>
      <button className="step" onClick={() => set(l, h + 1)} disabled={h === H - 1} aria-label="Next head">›</button>
    </div>
  );
}

export function LayerStepper({ layer, onChange, L }: { layer: number; onChange: (l: number) => void; L: number }) {
  return (
    <div className="stepper" role="group" aria-label="Choose a layer">
      <span className="stepper-label">layer</span>
      <button className="step" onClick={() => onChange(layer - 1)} disabled={layer === 0} aria-label="Previous layer">‹</button>
      <span className="stepper-value">{layer}</span>
      <button className="step" onClick={() => onChange(layer + 1)} disabled={layer === L - 1} aria-label="Next layer">›</button>
    </div>
  );
}

// A part's name and its one-line description.
export function partName(id: string) {
  if (id === "embed") return "Embedding";
  if (id === "bias") return "Biases";
  if (id.startsWith("m")) return `MLP ${id.slice(1)}`;
  return `Head ${id}`;
}

export function partGist(id: string) {
  if (id === "embed") return "The token and its position, before any layer";
  if (id === "bias") return "Fixed amounts every run adds, the same whatever the prompt";
  if (id.startsWith("m")) return id === "m0" ? "Layer 0's MLP, which GPT-2 uses to finish reading each token in" : "Works on each token by itself";
  return label(id) ?? research(id).circuits[0]?.role ?? habits(id)[0];
}

export function Section({ n, title, children, className = "" }: { n?: string; title: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card part-section ${className}`}>
      <h2 className="part-h2">{n && <span className="part-n">{n}</span>}{title}</h2>
      {children}
    </section>
  );
}

export const fmt = (v: number, digits = 2) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(digits)).replace("-", "−");
