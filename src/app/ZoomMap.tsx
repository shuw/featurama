// Every token GPT-2 knows on one map, laid out so tokens with similar
// embeddings sit together, with your prompt and the logit lens's path on it.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, readingLabel, show, word } from "./analysis.ts";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { loadTokenMap } from "./zoom.ts";
import { cssRgb, Section, TokenStrip, useThemeVersion } from "./zoom-ui.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

interface View {
  cx: number;
  cy: number;
  zoom: number;
}

const GRID = 128; // buckets per side, for finding the token under the pointer

export function ZoomMap({ engine, tokenizer, prompt, setPrompt, shuffle, writer }: Props) {
  const L = engine.config.n_layers;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const [map, setMap] = useState<Float32Array | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    loadTokenMap().then((q) => setMap(Float32Array.from(q, (v) => v / 32767))).catch(() => setFailed(true));
  }, []);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, lens: { positions: t.map((_, i) => i), topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run);
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const T = shown.length;
  const [query, setQuery] = useState<number | null>(null);
  const q = query === null || query >= T ? T - 1 : query;
  // UMAP leaves a few small islands far out; frame the bulk of the tokens.
  const home = useMemo<View>(() => {
    if (!map) return { cx: 0.5, cy: 0.5, zoom: 1 };
    const span = (k: number) => {
      const v = Array.from({ length: map.length / 2 }, (_, i) => map[i * 2 + k]).sort((a, b) => a - b);
      return [v[Math.floor(v.length * 0.01)], v[Math.floor(v.length * 0.99)]];
    };
    const [x0, x1] = span(0);
    const [y0, y1] = span(1);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, zoom: 0.9 / Math.max(x1 - x0, y1 - y0) };
  }, [map]);
  const [view, setView] = useState<View>({ cx: 0.5, cy: 0.5, zoom: 1 });
  useEffect(() => setView(home), [home]);
  const [search, setSearch] = useState("");
  const [notFound, setNotFound] = useState(false);

  // The lens's top guess after each layer, at the chosen token.
  const path = useMemo(() => {
    if (!trace?.lens) return [];
    const P = trace.lens.positions.length;
    const K = trace.lens.topK;
    return Array.from({ length: L + 1 }, (_, r) => trace.lens!.ids[(r * P + q) * K]);
  }, [trace, q, L]);

  const [found, setFound] = useState<number | null>(null);
  const go = (id: number, zoom = Math.max(home.zoom * 4, 10)) => {
    if (!map) return;
    setFound(id);
    setView({ cx: map[id * 2], cy: map[id * 2 + 1], zoom });
  };
  const find = () => {
    const ids = tokenizer.encode(search.startsWith(" ") ? search : ` ${search.trim()}`);
    const alt = tokenizer.encode(search.trim());
    const id = ids.length === 1 ? ids[0] : alt.length === 1 ? alt[0] : null;
    setNotFound(id === null);
    if (id !== null) go(id);
  };

  return (
    <>
      <PromptBox writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current && trace?.lens ? predictions(trace, L, tokenizer) : null} stale={!current} />
      <div className="zoom-field">
        <span className="card-label">Lens path from</span>
        <TokenStrip tokens={shown} tokenizer={tokenizer} value={q} onChange={setQuery} allowStart label="Choose the token whose guesses to follow" />
      </div>

      <Section title="All 50,257 tokens">
        <p className="muted">Each dot is one token, placed so tokens whose <Term tip="The 768 numbers GPT-2 starts from for each token. It uses the same table at the end, to score every token as the next one.">embeddings</Term> are alike sit close together (a UMAP layout). Scroll or pinch to zoom, drag to move. Your prompt's tokens are blue; the orange path follows GPT-2's best guess after “{q === 0 ? "⟨start⟩" : word(tokenizer.piece(shown[q]))}” layer by layer.</p>
        <div className="map-tools">
          <form className="map-find" onSubmit={(e) => { e.preventDefault(); find(); }}>
            <input className="feat-search" placeholder="Find a token, like “Paris”" value={search} onChange={(e) => { setSearch(e.target.value); setNotFound(false); }} aria-label="Find a token on the map" />
            <button className="secondary" type="submit">Find</button>
          </form>
          <div className="map-zoom">
            <button className="step" onClick={() => setView((v) => ({ ...v, zoom: Math.min(400, v.zoom * 1.6) }))} aria-label="Zoom in">+</button>
            <button className="step" onClick={() => setView((v) => ({ ...v, zoom: Math.max(0.8, v.zoom / 1.6) }))} aria-label="Zoom out">−</button>
            <button className="link-small" onClick={() => { setView(home); setFound(null); }}>Whole map</button>
          </div>
        </div>
        {notFound && <p className="muted small">That isn't a single token. Try one word.</p>}
        {failed ? <p className="muted">Couldn't load the map.</p> : !map ? <p className="muted">Loading the map…</p> : (
          <MapCanvas map={map} tokenizer={tokenizer} view={view} setView={setView} prompt={shown.slice(1)} path={path} found={found} onPick={(id) => go(id, view.zoom)} />
        )}
        {path.length > 0 && (
          <ol className="map-path">
            {path.map((id, r) => (
              <li key={r}><button className="link-small" onClick={() => go(id)}>{readingLabel(r)}: <span className="mono">{show(tokenizer.piece(id))}</span></button></li>
            ))}
          </ol>
        )}
      </Section>
    </>
  );
}

function MapCanvas({ map, tokenizer, view, setView, prompt, path, found, onPick }: {
  map: Float32Array; tokenizer: Tokenizer; view: View; setView: (f: (v: View) => View) => void; prompt: number[]; path: number[]; found: number | null; onPick: (id: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const [hover, setHover] = useState<number | null>(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const V = map.length / 2;
  const grid = useMemo(() => {
    const cells: number[][] = Array.from({ length: GRID * GRID }, () => []);
    for (let i = 0; i < V; i++) cells[Math.min(GRID - 1, Math.floor(map[i * 2 + 1] * GRID)) * GRID + Math.min(GRID - 1, Math.floor(map[i * 2] * GRID))].push(i);
    return cells;
  }, [map, V]);
  // Word-like tokens get labels first when there's room.
  const labelOrder = useMemo(() => {
    const score = Array.from({ length: V }, (_, i) => {
      const p = tokenizer.piece(i);
      return /^ \p{L}{3,}$/u.test(p) ? 0 : /^ ?\p{L}+$/u.test(p) ? 1 : 2;
    });
    return Array.from({ length: V }, (_, i) => i).sort((a, b) => score[a] - score[b] || a - b);
  }, [tokenizer, V]);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ro = new ResizeObserver(() => setSize({ w: c.clientWidth, h: c.clientHeight }));
    ro.observe(c);
    return () => ro.disconnect();
  }, []);

  const toScreen = useCallback((x: number, y: number) => {
    const s = Math.min(size.w, size.h) * view.zoom;
    return [(x - view.cx) * s + size.w / 2, (y - view.cy) * s + size.h / 2] as const;
  }, [view, size]);
  const toMap = (sx: number, sy: number) => {
    const s = Math.min(size.w, size.h) * view.zoom;
    return [(sx - size.w / 2) / s + view.cx, (sy - size.h / 2) / s + view.cy] as const;
  };

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, size.w, size.h);
    const faint = cssRgb("--faint");
    const ink = cssRgb("--ink");
    const resid = cssRgb("--resid");
    const attn = cssRgb("--attn");
    const r = Math.min(3, 0.8 + view.zoom * 0.08);
    g.fillStyle = faint;
    g.globalAlpha = view.zoom < 3 ? 0.35 : 0.6;
    for (let i = 0; i < V; i++) {
      const [x, y] = toScreen(map[i * 2], map[i * 2 + 1]);
      if (x < -2 || y < -2 || x > size.w + 2 || y > size.h + 2) continue;
      g.fillRect(x - r / 2, y - r / 2, r, r);
    }
    g.globalAlpha = 1;
    g.font = '11px "IBM Plex Mono", monospace';
    g.textBaseline = "middle";
    // Labels when zoomed in, word-like tokens first, never overlapping.
    if (view.zoom >= 6) {
      const placed: [number, number, number, number][] = [];
      g.fillStyle = ink;
      g.globalAlpha = 0.75;
      for (const i of labelOrder) {
        const [x, y] = toScreen(map[i * 2], map[i * 2 + 1]);
        if (x < 0 || y < 8 || x > size.w - 40 || y > size.h - 8) continue;
        const text = show(tokenizer.piece(i));
        const box: [number, number, number, number] = [x + 3, y - 7, x + 7 + g.measureText(text).width, y + 7];
        if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
        placed.push(box);
        g.fillText(text, x + 4, y);
        if (placed.length >= 220) break;
      }
      g.globalAlpha = 1;
    }
    const dot = (id: number, color: string, rad: number, text?: string) => {
      const [x, y] = toScreen(map[id * 2], map[id * 2 + 1]);
      g.fillStyle = color;
      g.beginPath();
      g.arc(x, y, rad, 0, Math.PI * 2);
      g.fill();
      if (text) {
        g.font = '600 12px "IBM Plex Mono", monospace';
        g.lineWidth = 3;
        g.strokeStyle = cssRgb("--bg");
        g.strokeText(text, x + rad + 3, y);
        g.fillText(text, x + rad + 3, y);
      }
    };
    for (const id of new Set(prompt)) dot(id, resid, 4, show(tokenizer.piece(id)));
    if (path.length) {
      g.strokeStyle = attn;
      g.lineWidth = 1.5;
      g.globalAlpha = 0.55;
      g.beginPath();
      path.forEach((id, k) => {
        const [x, y] = toScreen(map[id * 2], map[id * 2 + 1]);
        if (k) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      g.stroke();
      g.globalAlpha = 1;
      const seen = new Set<number>();
      path.forEach((id, k) => {
        const last = k === path.length - 1;
        if (seen.has(id) && !last) return;
        seen.add(id);
        dot(id, attn, last ? 6 : 4, last ? `${show(tokenizer.piece(id))} (final)` : show(tokenizer.piece(id)));
      });
    }
    if (found !== null) {
      const [x, y] = toScreen(map[found * 2], map[found * 2 + 1]);
      g.strokeStyle = ink;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x, y, 9, 0, Math.PI * 2);
      g.stroke();
      dot(found, ink, 3, show(tokenizer.piece(found)));
    }
    if (hover !== null) dot(hover, ink, 4, show(tokenizer.piece(hover)));
  }, [map, view, size, theme, prompt, path, hover, found, labelOrder, toScreen, tokenizer, V]);

  const nearest = (sx: number, sy: number) => {
    const [mx, my] = toMap(sx, sy);
    const gx = Math.floor(mx * GRID);
    const gy = Math.floor(my * GRID);
    let best: number | null = null;
    let bd = 10 * 10;
    const reach = Math.max(1, Math.ceil((10 / (Math.min(size.w, size.h) * view.zoom)) * GRID));
    for (let y = gy - reach; y <= gy + reach; y++)
      for (let x = gx - reach; x <= gx + reach; x++) {
        if (x < 0 || y < 0 || x >= GRID || y >= GRID) continue;
        for (const i of grid[y * GRID + x]) {
          const [px, py] = toScreen(map[i * 2], map[i * 2 + 1]);
          const dd = (px - sx) ** 2 + (py - sy) ** 2;
          if (dd < bd) {
            bd = dd;
            best = i;
          }
        }
      }
    return best;
  };

  // Dragging pans; a wheel or two fingers zoom around the pointer.
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<number | null>(null);
  const zoomAt = (sx: number, sy: number, factor: number) =>
    setView((v) => {
      const zoom = Math.max(0.8, Math.min(400, v.zoom * factor));
      const s0 = Math.min(size.w, size.h) * v.zoom;
      const s1 = Math.min(size.w, size.h) * zoom;
      const mx = (sx - size.w / 2) / s0 + v.cx;
      const my = (sy - size.h / 2) / s0 + v.cy;
      return { zoom, cx: mx - (sx - size.w / 2) / s1, cy: my - (sy - size.h / 2) / s1 };
    });
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.002));
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  });

  return (
    <canvas ref={ref} className="map-canvas" role="img" aria-label="A map of every token"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
        drag.current = { x: e.clientX, y: e.clientY, moved: false };
      }}
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const prev = pointers.current.get(e.pointerId);
        if (prev) pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (pointers.current.size === 2) {
          const [a, b] = [...pointers.current.values()];
          const dist = Math.hypot(a.x - b.x, a.y - b.y);
          if (pinch.current) zoomAt((a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top, dist / pinch.current);
          pinch.current = dist;
          drag.current = null;
          return;
        }
        if (drag.current && prev) {
          const dx = e.clientX - prev.x;
          const dy = e.clientY - prev.y;
          if (Math.abs(e.clientX - drag.current.x) + Math.abs(e.clientY - drag.current.y) > 4) drag.current.moved = true;
          const s = Math.min(size.w, size.h);
          setView((v) => ({ ...v, cx: v.cx - dx / (s * v.zoom), cy: v.cy - dy / (s * v.zoom) }));
          return;
        }
        setHover(nearest(e.clientX - r.left, e.clientY - r.top));
      }}
      onPointerUp={(e) => {
        pointers.current.delete(e.pointerId);
        if (pointers.current.size < 2) pinch.current = null;
        const r = e.currentTarget.getBoundingClientRect();
        if (drag.current && !drag.current.moved) {
          const id = nearest(e.clientX - r.left, e.clientY - r.top);
          if (id !== null) onPick(id);
        }
        drag.current = null;
      }}
      onPointerLeave={() => setHover(null)} />
  );
}
