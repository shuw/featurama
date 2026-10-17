// Every weight of GPT-2 in one 3D tower you can fly into, down to a single
// weight and its 16 bits, with the forward pass painted onto it. The pass
// plays as a short film, paced to be followed rather than in real time.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show, word } from "./analysis.ts";
import { tokenize } from "./model.ts";
import { pct, PromptBox } from "./PromptBox.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { TokenStrip } from "./inside-ui.tsx";
import { eyeOf, FOV, hitPlane, project, ray, viewProj, type View } from "./tower/camera.ts";
import { TowerRenderer, type Acts } from "./tower/gpu.ts";
import { buildTower, FLOOR, toTensor, type Tile } from "./tower/layout.ts";
import { buildStages } from "./tower/stages.ts";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

type Speed = "slow" | "normal" | "fast";
// The film: an establishing shot, then the climb (seconds), then the result.
const INTRO = 1.6;
const CLIMB: Record<Speed, number> = { slow: 18, normal: 10, fast: 5 };

// The camera tilts as you zoom: the whole tower at an angle from afar, and a
// floor seen almost from above up close, so matrices read like a map.
const FAR = 30000;
const NEAR = 3500;
function pitchFor(dist: number) {
  const t = Math.min(1, Math.max(0, (Math.log(FAR) - Math.log(dist)) / (Math.log(FAR) - Math.log(NEAR))));
  const s = t * t * (3 - 2 * t);
  return 0.5 + 0.82 * s;
}
const withPitch = (v: Omit<View, "pitch">): View => ({ ...v, pitch: pitchFor(v.dist) });
// Close to a floor, it's in focus: others fade, and those above dissolve.
function focusOf(v: View) {
  return { floor: v.target[1] / FLOOR, fade: Math.min(1, Math.max(0, (16000 - v.dist) / 9000)) };
}
// The highest floor you can point at.
const topFloor = (v: View) => {
  const f = focusOf(v);
  return f.fade > 0.4 ? f.floor + 0.5 : Infinity;
};
const START = withPitch({ target: [0, 6.2 * FLOOR, 0], dist: 41000, yaw: 0.6 });
const lerpView = (a: View, b: View, k: number): View =>
  withPitch({
    target: [a.target[0] + (b.target[0] - a.target[0]) * k, a.target[1] + (b.target[1] - a.target[1]) * k, a.target[2] + (b.target[2] - a.target[2]) * k],
    dist: Math.exp(Math.log(a.dist) + (Math.log(b.dist) - Math.log(a.dist)) * k), yaw: a.yaw + (b.yaw - a.yaw) * k,
  });
const smooth = (x: number) => x * x * (3 - 2 * x);

let F16: Float32Array | null = null;
function f16(h: number) {
  F16 ??= (() => {
    const t = new Float32Array(65536);
    for (let x = 0; x < 65536; x++) {
      const s = x & 0x8000 ? -1 : 1;
      const e = (x >> 10) & 0x1f;
      const f = x & 0x3ff;
      t[x] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return t;
  })();
  return F16[h];
}

interface Readout {
  tile: Tile;
  row: number;
  col: number;
  value: number;
  bits: number | null; // the raw fp16 bits, for weights
  input: number | null; // what it's multiplied by in the forward pass
}

interface Label {
  x: number;
  y: number;
  text: string;
  big?: boolean;
  bits?: string;
  active?: boolean;
  kind: "floor" | "tile" | "cell";
}

// A floating label tied by a line to a point in the tower.
interface Callout {
  ax: number; // the anchor, on screen
  ay: number;
  x: number; // the label
  y: number;
}

interface Spine {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  hx: number;
  hy: number;
}

export function Tower({ engine, tokenizer, weights, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, d_model: d } = engine.config;
  const tower = useMemo(() => buildTower(L, d, engine.config.d_mlp, engine.config.d_vocab, engine.config.n_ctx), [engine, L, d]);
  const FLOORS = tower.floors.length;
  // Each floor lights in two halves: attention, then the MLP (or the two tables).
  const halfOf = useMemo(() => {
    const stages = buildStages(L);
    const m = new Map<string, number>();
    for (let f = 0; f < FLOORS; f++) {
      const here = stages.filter((st) => st.floor === f);
      here.forEach((st, k) => st.tiles.forEach((id) => m.set(id, k < here.length / 2 ? 0 : 1)));
    }
    return tower.tiles.map((t) => m.get(t.id) ?? 0);
  }, [tower, L, FLOORS]);
  // Time spent on each floor: a little longer on the ground and the roof.
  const weight = useMemo(() => tower.floors.map((_, f) => (f === 0 || f === FLOORS - 1 ? 1.6 : 1)), [tower, FLOORS]);

  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { resid: true, residMid: true, z: true, neurons: true }, lens: { positions: t.map((_, i) => i), topK: 3 } }), [engine]);
  const { result: trace } = useRun(tokens, run);
  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const [query, setQuery] = useState<number | null>(null);
  const q = query === null || query >= shown.length ? shown.length - 1 : query;
  const acts = useMemo(() => (trace ? activations(trace, weights, q) : null), [trace, weights, q]);

  const canvas = useRef<HTMLCanvasElement>(null);
  const stageEl = useRef<HTMLDivElement>(null);
  // Film mode (?film), for recording the trailer: the tower fills the window,
  // the controls hide, and a script sets every frame through window.__film.
  const filmMode = useMemo(() => new URLSearchParams(location.search).has("film"), []);
  const film = useRef<{ pos: number | null; done: boolean; labels: boolean; callouts: boolean } | null>(null);
  const uploaded = useRef("");
  const lastPx = useRef(0);
  const renderer = useRef<TowerRenderer | null>(null);
  const view = useRef<View>({ ...START, target: [...START.target] });
  const goal = useRef<View | null>(null); // where an eased zoom is heading
  const [forward, setForward] = useState(true);
  const [readout, setReadout] = useState<Readout | null>(null);
  const [pxPerWeight, setPxPerWeight] = useState(0);
  const [labels, setLabels] = useState<Label[]>([]);
  const [spine, setSpine] = useState<Spine | null>(null);
  const [callouts, setCallouts] = useState<{ input: Callout | null; answer: Callout | null }>({ input: null, answer: null });
  const [where, setWhere] = useState<{ floor: number | null; tile: number | null }>({ floor: null, tile: null });
  const [full, setFull] = useState(false);
  const dirty = useRef(true);
  const pointer = useRef<{ x: number; y: number } | null>(null);

  // The film's clock, in seconds; its pace; whether the camera is guided.
  const [animate, setAnimate] = useState(true);
  const [playing, setPlaying] = useState(!filmMode);
  const [speed, setSpeed] = useState<Speed>("normal");
  const [follow, setFollow] = useState(true);
  const clock = useRef(0);
  const [phase, setPhase] = useState<{ floor: number; done: boolean; intro: boolean; at: number }>({ floor: -1, done: false, intro: true, at: 0 });

  const guess = trace?.lens ? topAt(trace, L, q, tokenizer) : null;
  const answerId = trace?.lens ? trace.lens.ids[(L * trace.lens.positions.length + q) * trace.lens.topK] : null;
  const state = useRef({ forward, acts, animate, playing, speed, follow, answerId });
  state.current = { forward, acts, animate, playing, speed, follow, answerId };
  const end = () => INTRO + CLIMB[state.current.speed];

  // Position in floors (0..FLOORS) at a clock time: eased, with floors weighted.
  const posAt = (t: number, sp: Speed) => {
    if (t <= INTRO) return 0;
    const u = smooth(Math.min(1, (t - INTRO) / CLIMB[sp]));
    const total = weight.reduce((a, b) => a + b, 0);
    let left = u * total;
    for (let f = 0; f < FLOORS; f++) {
      if (left <= weight[f]) return f + left / weight[f];
      left -= weight[f];
    }
    return FLOORS;
  };

  useEffect(() => {
    const r = new TowerRenderer(engine, canvas.current!, tower.tiles, tower.floors, ACT_SIZE(L, d, engine.config.d_mlp));
    renderer.current = r;
    dirty.current = true;
    return () => {
      r.destroy();
      renderer.current = null;
    };
  }, [engine, tower]);

  useEffect(() => {
    if (!acts || !renderer.current) return;
    renderer.current.setActs(acts, tower.tiles.map((t) => scaleOf(t, weights, acts)));
    uploaded.current = trace!.tokens.join(",");
    dirty.current = true;
  }, [acts, tower, weights]);
  useEffect(() => {
    dirty.current = true;
  }, [forward, animate]);

  // A new token to follow (one generated, typed or picked) plays the pass
  // again, once things settle; repeats skip the establishing shot.
  const replay = (intro: boolean) => {
    clock.current = intro ? 0 : INTRO - 0.2;
    setPlaying(true);
    dirty.current = true;
  };
  const followed = trace ? `${q}:${trace.tokens[q]}` : "";
  const first = useRef(true);
  useEffect(() => {
    if (!followed) return;
    if (first.current) {
      first.current = false;
      return;
    }
    const t = setTimeout(() => state.current.animate && replay(false), 450);
    return () => clearTimeout(t);
  }, [followed]);

  useEffect(() => {
    const on = () => {
      setFull(document.fullscreenElement === stageEl.current);
      dirty.current = true;
    };
    document.addEventListener("fullscreenchange", on);
    return () => document.removeEventListener("fullscreenchange", on);
  }, []);
  const toggleFull = () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else stageEl.current?.requestFullscreen().catch(() => {});
    canvas.current?.focus();
  };

  // --- Camera ----------------------------------------------------------------

  const flight = useRef(0);
  const fly = (to: Omit<View, "pitch">, ms = 900) => {
    goal.current = null;
    const from = { ...view.current, target: [...view.current.target] as View["target"] };
    const dest = withPitch(to);
    const t0 = performance.now();
    const id = ++flight.current;
    const go = () => {
      if (id !== flight.current) return;
      const k = Math.min(1, (performance.now() - t0) / ms);
      view.current = lerpView(from, dest, k < 0.5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2);
      dirty.current = true;
      if (k < 1) requestAnimationFrame(go);
    };
    go();
  };
  const aspect = () => {
    const c = canvas.current;
    return c ? c.clientWidth / Math.max(1, c.clientHeight) : 16 / 9;
  };
  // Frame a box on a floor, facing straight on (rows left to right, like a map).
  const frame = (x0: number, x1: number, z0: number, z1: number, y: number, ms = 900) => {
    const w = Math.max(x1 - x0, 1400);
    const fit = Math.max(w / aspect(), (z1 - z0) * 1.15) / (2 * Math.tan(FOV / 2));
    fly({ target: [(x0 + x1) / 2, y, (z0 + z1) / 2], dist: Math.max(40, fit * 1.25), yaw: 0 }, ms);
  };
  const toTower = () => fly({ ...START, target: [...START.target], yaw: view.current.yaw });
  const toFloor = (i: number) => {
    const f = tower.floors[i];
    frame(f.x0, f.x1, f.z0, f.z1, f.y);
  };
  const toTile = (i: number) => {
    const t = tower.tiles[i];
    frame(t.x, t.x + t.w, t.z, t.z + t.d, t.floor * FLOOR);
  };
  // The guided camera: the whole tower, turning slowly and rising with the pass.
  const guided = (t: number, pos: number): View => {
    const k = smooth(Math.min(1, pos / FLOORS));
    return withPitch({ target: [0, (4.6 + 3.2 * k) * FLOOR, 0], dist: 41000 - 6000 * k, yaw: 0.45 + 0.035 * Math.min(t, end()) });
  };
  const takeOver = () => {
    flight.current++;
    if (state.current.follow) setFollow(false);
  };
  const clampView = (v: View): View => {
    const xs = tower.floors.map((f) => [f.x0, f.x1]).flat();
    const zs = tower.floors.map((f) => [f.z0, f.z1]).flat();
    const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
    return withPitch({ ...v, dist: clamp(v.dist, 0.15, 60000), target: [clamp(v.target[0], Math.min(...xs), Math.max(...xs)), clamp(v.target[1], 0, (L + 1) * FLOOR), clamp(v.target[2], Math.min(...zs), Math.max(...zs))] });
  };
  const locate = (v: View) => {
    const floor = v.dist < 16000 ? Math.max(0, Math.min(L + 1, Math.round(v.target[1] / FLOOR))) : null;
    let tile: number | null = null;
    if (floor !== null) {
      const i = tower.tiles.findIndex((t) => t.floor === floor && v.target[0] >= t.x && v.target[0] <= t.x + t.w && v.target[2] >= t.z && v.target[2] <= t.z + t.d);
      if (i >= 0 && v.dist < Math.max(tower.tiles[i].w, tower.tiles[i].d) * 2.2) tile = i;
    }
    return { floor, tile };
  };
  const goUp = () => {
    takeOver();
    const w = locate(view.current);
    if (w.tile !== null && w.floor !== null) toFloor(w.floor);
    else toTower();
  };

  // --- The loop: run the film, ease the camera, draw ---------------------------

  useEffect(() => {
    let id = 0;
    let last = performance.now();
    let shown = "";
    const tick = () => {
      id = requestAnimationFrame(tick);
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const s = state.current;
      const stop = INTRO + CLIMB[s.speed];
      const f = filmMode ? film.current : null;
      if (s.animate && s.playing && s.acts && !filmMode) {
        clock.current = Math.min(stop, clock.current + dt);
        if (clock.current >= stop) setPlaying(false);
        dirty.current = true;
      }
      const pos = f && f.pos !== null ? f.pos : s.animate ? posAt(clock.current, s.speed) : FLOORS;
      const intro = !f && s.animate && clock.current < INTRO;
      const floor = Math.min(FLOORS - 1, Math.floor(pos));
      const done = f ? f.done : !s.animate || clock.current >= stop;
      const key = `${floor}:${done}:${intro}`;
      if (key !== shown) {
        shown = key;
        setPhase({ floor, done, intro, at: clock.current });
      }
      // Camera: eased zoom toward a goal; or, guided, drift with the film.
      if (goal.current) {
        const k = 1 - Math.exp(-dt * 14);
        view.current = lerpView(view.current, goal.current, k);
        if (Math.abs(Math.log(view.current.dist / goal.current.dist)) < 0.002) {
          view.current = goal.current;
          goal.current = null;
        }
        dirty.current = true;
      } else if (!filmMode && s.animate && s.follow && (s.playing || !done)) {
        view.current = lerpView(view.current, guided(clock.current, pos), 1 - Math.exp(-dt * 3));
        dirty.current = true;
      }
      const r = renderer.current;
      const c = canvas.current;
      if (!r || !c || !dirty.current) return;
      dirty.current = false;
      const a = c.clientWidth / Math.max(1, c.clientHeight);
      const v = view.current;
      // Each half floor's front sweeps from just above the tile to just below it.
      const progress = new Float32Array(tower.tiles.length).map((_, i) => {
        if (!s.animate) return 2;
        const x = (pos - tower.tiles[i].floor) * 2 - halfOf[i];
        return Math.min(1.1, Math.max(-0.1, x * 1.2 - 0.1));
      });
      const { floor: focus, fade } = focusOf(v);
      r.render(viewProj(v, a), eyeOf(v), { forward: s.forward && !!s.acts, slow: s.animate, focus, fade }, s.acts, progress);
      setLabels(f && !f.labels ? [] : labelsFor(v, c, tower, weights, s.acts, fade > 0.5 ? Math.round(focus) : null, s.animate && !done ? floor : null));
      setWhere(locate(v));
      // The running summary as a line beside the tower, lit up to where the pass is.
      if (s.animate) {
        const f1 = tower.floors[1];
        const sx = f1.x0 - 400;
        const sz = (f1.z0 + f1.z1) / 2;
        const p0 = project(v, a, c.clientWidth, c.clientHeight, [sx, 0, sz]);
        const p1 = project(v, a, c.clientWidth, c.clientHeight, [sx, (FLOORS - 1) * FLOOR, sz]);
        // The head sits on the floor being lit (floor f lights while pos runs from f to f + 1).
        const ph = project(v, a, c.clientWidth, c.clientHeight, [sx, Math.min(FLOORS - 1, Math.max(0, pos - 0.5)) * FLOOR, sz]);
        setSpine(p0 && p1 && ph && fade < 0.5 ? { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y, hx: ph.x, hy: ph.y } : null);
      } else setSpine(null);
      // The input's row on the ground, and the answer's row on the roof.
      const rowPoint = (tileId: string, row: number) => {
        const t = tower.tiles.find((x) => x.id === tileId)!;
        const fold = t.fold ? Math.floor(row / t.fold.rows) : 0;
        const r = t.fold ? row % t.fold.rows : (row / t.rows) * t.d;
        return [t.x + fold * (t.cols + (t.fold?.gap ?? 0)) + t.cols / 2, t.floor * FLOOR, t.z + r + 0.5];
      };
      const call = (pt: number[], dx: number, dy: number): Callout | null => {
        const sp = project(v, a, c.clientWidth, c.clientHeight, pt);
        if (!sp || sp.x < -20 || sp.x > c.clientWidth + 20 || sp.y < -20 || sp.y > c.clientHeight + 20) return null;
        return { ax: sp.x, ay: sp.y, x: Math.min(c.clientWidth - 200, Math.max(12, sp.x + dx)), y: Math.min(c.clientHeight - 60, Math.max(56, sp.y + dy)) };
      };
      const roofLit = done || pos >= FLOORS - 0.5;
      if (f && !f.callouts) setCallouts({ input: null, answer: null });
      else setCallouts({
        input: s.acts ? call(rowPoint("wte", s.acts.token), -230, -70) : null,
        answer: s.acts && s.answerId !== null && (roofLit || !s.animate) ? call(rowPoint("wte2", s.answerId), 60, -90) : null,
      });
      const center = ray(v, a, 0, 0);
      const hit = surfaceHit(center.origin, center.dir, tower, topFloor(v)) ?? hitPlane(center.origin, center.dir, v.target[1]);
      if (hit) {
        const p0 = project(v, a, c.clientWidth, c.clientHeight, hit);
        const p1 = project(v, a, c.clientWidth, c.clientHeight, [hit[0] + 1, hit[1], hit[2]]);
        lastPx.current = p0 && p1 ? Math.hypot(p1.x - p0.x, p1.y - p0.y) : 0;
        setPxPerWeight(lastPx.current);
      }
      if (pointer.current) setReadout(pick(v, c, pointer.current, tower, weights, s.acts));
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [tower, weights, halfOf, FLOORS]);

  // --- Film mode ---------------------------------------------------------------

  useEffect(() => {
    if (!filmMode) return;
    const frames = (n: number) => new Promise<void>((done) => {
      const step = (k: number) => (k <= 0 ? done() : requestAnimationFrame(() => step(k - 1)));
      step(n);
    });
    (window as any).__film = {
      // Set the prompt and wait until its run is on the GPU.
      prompt: async (text: string) => {
        setPrompt(text);
        const want = tokenize(tokenizer, text).join(",");
        for (let i = 0; i < 400 && uploaded.current !== want; i++) await new Promise((r) => setTimeout(r, 25));
        await frames(3);
        return uploaded.current === want;
      },
      // One frame: a camera (or the guided one for a pass position), how far
      // the pass has got (floors, 0..14), whether it's finished, labels or not.
      shot: async (o: { view?: Omit<View, "pitch">; pos: number; done?: boolean; labels?: boolean; callouts?: boolean; t?: number }) => {
        film.current = { pos: o.pos, done: !!o.done, labels: o.labels ?? false, callouts: o.callouts ?? true };
        goal.current = null;
        flight.current++;
        view.current = o.view ? withPitch(o.view) : guided(o.t ?? 0, o.pos);
        dirty.current = true;
        await frames(2);
        return lastPx.current;
      },
      tile: (id: string) => tower.tiles.find((t) => t.id === id),
      floorY: FLOOR,
    };
    return () => {
      delete (window as any).__film;
    };
  }, [filmMode, tokenizer, setPrompt, tower]);

  // --- Input -------------------------------------------------------------------

  useEffect(() => {
    const c = canvas.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      takeOver();
      // Zoom toward the pointer, eased: the goal moves at once, the view follows.
      const v = goal.current ?? view.current;
      const rect = c.getBoundingClientRect();
      const { origin, dir } = ray(v, rect.width / rect.height, ((e.clientX - rect.left) / rect.width) * 2 - 1, 1 - ((e.clientY - rect.top) / rect.height) * 2);
      const hit = surfaceHit(origin, dir, tower, topFloor(v)) ?? hitPlane(origin, dir, v.target[1]) ?? v.target;
      const factor = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0022));
      const dist = Math.min(60000, Math.max(0.15, v.dist * factor));
      const k = dist / v.dist;
      goal.current = clampView({ ...v, dist, target: [hit[0] + (v.target[0] - hit[0]) * k, hit[1] + (v.target[1] - hit[1]) * k, hit[2] + (v.target[2] - hit[2]) * k] });
      dirty.current = true;
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  }, [tower]);

  const drag = useRef<{ x: number; y: number; turn: boolean; moved: boolean } | null>(null);
  const touches = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<number | null>(null);
  const screenRay = (x: number, y: number) => {
    const c = canvas.current!;
    return ray(view.current, c.clientWidth / c.clientHeight, (x / c.clientWidth) * 2 - 1, 1 - (y / c.clientHeight) * 2);
  };
  const zoomBy = (f: number) => {
    takeOver();
    const v = goal.current ?? view.current;
    goal.current = clampView({ ...v, dist: v.dist * f });
  };

  const togglePlay = () => {
    if (!playing && clock.current >= end()) replay(false);
    else setPlaying(!playing);
    setFollow(true);
  };
  const scrubTo = (u: number) => {
    setPlaying(false);
    clock.current = INTRO + u * CLIMB[speed];
    dirty.current = true;
  };

  const onKey = (e: React.KeyboardEvent) => {
    const v = view.current;
    const span = v.dist * 0.25;
    const right = [Math.cos(v.yaw), -Math.sin(v.yaw)];
    const fwd = [-Math.sin(v.yaw), -Math.cos(v.yaw)];
    const move = (dx: number, dz: number) => {
      takeOver();
      goal.current = clampView({ ...(goal.current ?? v), target: [v.target[0] + dx, v.target[1], v.target[2] + dz] });
    };
    const keys: Record<string, () => void> = {
      ArrowLeft: () => move(-right[0] * span, -right[1] * span),
      ArrowRight: () => move(right[0] * span, right[1] * span),
      ArrowUp: () => move(fwd[0] * span, fwd[1] * span),
      ArrowDown: () => move(-fwd[0] * span, -fwd[1] * span),
      "+": () => zoomBy(1 / 1.6),
      "=": () => zoomBy(1 / 1.6),
      "-": () => zoomBy(1.6),
      PageUp: () => { takeOver(); toFloor(Math.min(L + 1, (where.floor ?? 0) + 1)); },
      PageDown: () => { takeOver(); toFloor(Math.max(0, (where.floor ?? 1) - 1)); },
      Escape: goUp,
      " ": () => animate && togglePlay(),
      f: toggleFull,
    };
    const k = keys[e.key];
    if (k) {
      e.preventDefault();
      k();
    }
  };

  const crumbs: { label: string; go: () => void }[] = [{ label: "Tower", go: () => { takeOver(); toTower(); } }];
  if (where.floor !== null) crumbs.push({ label: tower.floors[where.floor].label, go: () => { takeOver(); toFloor(where.floor!); } });
  if (where.tile !== null) crumbs.push({ label: tower.tiles[where.tile].name, go: () => { takeOver(); toTile(where.tile!); } });

  const piece = q === 0 ? "⟨start⟩" : `“${word(tokenizer.piece(shown[q] ?? 0))}”`;
  const caption = !acts || !trace ? "Running GPT-2…"
    : phase.intro ? `GPT-2 small has 124 million weights, stacked here as ${FLOORS} floors. Watch ${piece} go through them.`
    : phase.done ? `${piece} has been through all ${FLOORS} floors. GPT-2's next-token guess: ${guess?.map((g) => `“${word(g.piece)}” ${pct(g.p)}`).join(", ")}.`
    : describeFloor(phase.floor, L, trace, acts, q, tokenizer, engine.config.d_mlp);
  const u = Math.max(0, Math.min(1, (clock.current - INTRO) / CLIMB[speed]));

  return (
    <article className="page tower-page">
      <header className="page-head">
        <h1>Every weight, in 3D</h1>
        <p className="dek">All 124 million of GPT-2's weights, floor by floor: the token table on the ground, a floor for each layer, and the same table again on the roof. Watch a token climb through them, then zoom as far as you like, down to a single weight and its 16 bits.</p>
      </header>
      <PromptBox writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current && trace?.lens ? lastPredictions(trace, L, tokenizer) : null} stale={!current} />
      <div className="part-field">
        <span className="card-label">Forward pass for</span>
        <TokenStrip tokens={shown} tokenizer={tokenizer} value={q} onChange={setQuery} allowStart label="Choose the token whose forward pass to show" />
      </div>
      <div className="tower-tools">
        <div className="seg" role="radiogroup" aria-label="Colour by">
          <button role="radio" aria-checked={!forward} className={!forward ? "seg-on" : ""} onClick={() => setForward(false)}>The weights</button>
          <button role="radio" aria-checked={forward} className={forward ? "seg-on" : ""} onClick={() => setForward(true)}>The forward pass</button>
        </div>
        <button className="switch-chip" role="switch" aria-checked={animate} onClick={() => { const on = !animate; setAnimate(on); if (on) { replay(true); setFollow(true); } }}>
          <span className={animate ? "chip-dot chip-dot-on" : "chip-dot"} /> Animate the pass
        </button>
      </div>

      <div className={filmMode ? "tower-stage tower-film" : "tower-stage"} ref={stageEl}>
        <canvas ref={canvas} className="tower-canvas" tabIndex={0} aria-label="The weights of GPT-2 in 3D. Arrow keys move, plus and minus zoom, Page Up and Down change floor, Escape zooms out, F goes full screen."
          onKeyDown={onKey}
          onContextMenu={(e) => e.preventDefault()}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            e.currentTarget.focus();
            touches.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
            drag.current = { x: e.clientX, y: e.clientY, turn: e.shiftKey || e.button === 2, moved: false };
          }}
          onPointerMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            pointer.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            const prev = touches.current.get(e.pointerId);
            if (prev) touches.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (touches.current.size === 2) {
              const [a, b] = [...touches.current.values()];
              const dd = Math.hypot(a.x - b.x, a.y - b.y);
              if (pinch.current) {
                takeOver();
                goal.current = null;
                view.current = clampView({ ...view.current, dist: view.current.dist * (pinch.current / dd) });
              }
              pinch.current = dd;
              dirty.current = true;
              return;
            }
            const g = drag.current;
            if (g && prev) {
              if (Math.abs(e.clientX - g.x) + Math.abs(e.clientY - g.y) > 4) {
                if (!g.moved) takeOver();
                g.moved = true;
              }
              if (g.moved) {
                goal.current = null;
                const v = view.current;
                if (g.turn) {
                  view.current = withPitch({ ...v, yaw: v.yaw - (e.clientX - prev.x) * 0.006 });
                } else {
                  // The point under the pointer stays under it.
                  const a = screenRay(prev.x - rect.left, prev.y - rect.top);
                  const b = screenRay(pointer.current.x, pointer.current.y);
                  const pa = hitPlane(a.origin, a.dir, v.target[1]);
                  const pb = hitPlane(b.origin, b.dir, v.target[1]);
                  if (pa && pb) view.current = clampView({ ...v, target: [v.target[0] + pa[0] - pb[0], v.target[1], v.target[2] + pa[2] - pb[2]] });
                }
              }
            }
            dirty.current = true;
          }}
          onPointerUp={(e) => {
            touches.current.delete(e.pointerId);
            if (touches.current.size < 2) pinch.current = null;
            const g = drag.current;
            drag.current = null;
            // A click (not a drag) flies to what's under it.
            if (g && !g.moved && !g.turn) {
              const rect = e.currentTarget.getBoundingClientRect();
              const { origin, dir } = screenRay(e.clientX - rect.left, e.clientY - rect.top);
              const p = surfaceHit(origin, dir, tower, topFloor(view.current));
              const t = p ? tower.tiles.indexOf(tileAt(tower, p)!) : -1;
              if (t >= 0) {
                takeOver();
                toTile(t);
              }
            }
          }}
          onPointerLeave={() => {
            pointer.current = null;
            setReadout(null);
          }} />

        {spine && (
          <svg className="tower-spine" aria-hidden="true">
            <line x1={spine.x0} y1={spine.y0} x2={spine.x1} y2={spine.y1} className="spine-track" />
            <line x1={spine.x0} y1={spine.y0} x2={spine.hx} y2={spine.hy} className="spine-lit" />
            <circle cx={spine.hx} cy={spine.hy} r={7} className="spine-head" />
          </svg>
        )}
        <svg className="tower-spine" aria-hidden="true">
          {callouts.input && <line x1={callouts.input.ax} y1={callouts.input.ay} x2={callouts.input.x + 90} y2={callouts.input.y + 20} className="callout-line callout-line-in" />}
          {callouts.answer && <line x1={callouts.answer.ax} y1={callouts.answer.ay} x2={callouts.answer.x + 20} y2={callouts.answer.y + 24} className="callout-line callout-line-out" />}
          {callouts.input && <circle cx={callouts.input.ax} cy={callouts.input.ay} r={4} className="callout-dot callout-dot-in" />}
          {callouts.answer && <circle cx={callouts.answer.ax} cy={callouts.answer.ay} r={5} className="callout-dot callout-dot-out" />}
        </svg>
        {callouts.input && acts && (
          <div className={`callout callout-in${animate && (phase.intro || phase.floor === 0) ? " callout-hot" : ""}`} style={{ left: callouts.input.x, top: callouts.input.y }}>
            <span className="callout-kicker">in</span>
            <span className="callout-word">{piece}</span>
            <span className="callout-meta">token {acts.token.toLocaleString()} · position {acts.position}</span>
          </div>
        )}
        {callouts.answer && guess && (
          <div className="callout callout-out" style={{ left: callouts.answer.x, top: callouts.answer.y }}>
            <span className="callout-kicker">next</span>
            <span className="callout-word">“{word(guess[0].piece)}” <span className="callout-p">{pct(guess[0].p)}</span></span>
            <span className="callout-meta">{guess.slice(1).map((g) => `“${word(g.piece)}” ${pct(g.p)}`).join(" · ")}</span>
          </div>
        )}
        <div className="tower-labels" aria-hidden="true">
          {labels.map((l, i) => (
            <div key={i} className={`tower-label label-${l.kind}${l.big ? " label-big" : ""}${l.active ? " label-active" : ""}`} style={{ left: l.x, top: l.y }}>
              {l.text}
              {l.bits && <span className="tower-bits">{l.bits}</span>}
            </div>
          ))}
        </div>

        <nav className="tower-crumbs" aria-label="Where you are">
          {crumbs.map((c, i) => (
            <span key={i}>{i > 0 && <span className="crumb-sep">›</span>}<button className={i === crumbs.length - 1 ? "tower-crumb tower-crumb-on" : "tower-crumb"} onClick={c.go}>{c.label}</button></span>
          ))}
          {crumbs.length > 1 && <button className="tower-up" onClick={goUp} title="Zoom out a level (Esc)">↑ out</button>}
        </nav>
        <button className="tower-full" onClick={toggleFull} aria-label={full ? "Leave full screen" : "Full screen"} title={full ? "Leave full screen (F)" : "Full screen (F)"}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            {full ? <path d="M9 3v6H3M15 3v6h6M9 21v-6H3M15 21v-6h6" /> : <path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6" />}
          </svg>
        </button>
        <div className="tower-rail" aria-label="Floors">
          {[...tower.floors].reverse().map((f) => (
            <button key={f.index} className={`${where.floor === f.index ? "tower-rail-on" : ""}${animate && !phase.done && !phase.intro && phase.floor === f.index ? " tower-rail-pass" : ""}`}
              onClick={() => { takeOver(); toFloor(f.index); }} title={f.label}>
              {f.index === 0 ? "E" : f.index === tower.floors.length - 1 ? "U" : f.index - 1}
            </button>
          ))}
        </div>

        {readout && (
          <div className="tower-hud">
            <div className="tower-part">{readout.tile.part}</div>
            <div className="mono small">
              {readout.tile.rows > 1 && `row ${readout.row}`}{readout.tile.cols > 1 && ` · column ${readout.col}`}
              {readout.tile.heads && ` · head ${Math.floor(readout.col / 64)}`}
              {readout.tile.id.startsWith("wte") && ` · token “${show(tokenizer.piece(readout.row))}”`}
            </div>
            <div className="mono">
              {fmt(readout.value)}
              {readout.input !== null && forward && <> × {fmt(readout.input)} = <strong>{fmt(readout.value * readout.input)}</strong></>}
            </div>
            {readout.bits !== null && <div className="mono small tower-hud-bits">{bitString(readout.bits)}</div>}
          </div>
        )}

        {animate && (
          <div className="tower-player">
            <div className="tower-caption" key={`${phase.floor}${phase.done}${phase.intro}`}>
              <strong>{phase.intro ? "The tower" : phase.done ? "The answer" : tower.floors[phase.floor].label}.</strong> {caption}
            </div>
            <div className="tower-controls">
              <button className="step tower-play" onClick={togglePlay} aria-label={playing ? "Pause" : phase.done ? "Play again" : "Play"}>{playing ? "❚❚" : phase.done ? "↺" : "▶"}</button>
              <input type="range" min={0} max={1000} value={Math.round(u * 1000)} onChange={(e) => scrubTo(Number(e.target.value) / 1000)} aria-label="How far through the pass" />
              <div className="seg seg-small" role="radiogroup" aria-label="Speed">
                {(["slow", "normal", "fast"] as Speed[]).map((s) => <button key={s} role="radio" aria-checked={speed === s} className={speed === s ? "seg-on" : ""} onClick={() => setSpeed(s)}>{s}</button>)}
              </div>
              {!follow && <button className="link-small tower-follow" onClick={() => setFollow(true)}>Follow</button>}
            </div>
          </div>
        )}

        <div className="tower-legend small">
          <span><i className="lg-pos" /> positive</span><span><i className="lg-neg" /> negative</span>
          <span className="mono" data-px={pxPerWeight.toFixed(4)}>1 weight ≈ {pxPerWeight >= 1 ? `${Math.round(pxPerWeight)} px` : `1/${Math.round(1 / Math.max(pxPerWeight, 1e-6)).toLocaleString()} px`}</span>
        </div>
      </div>
      <p className="muted small">Drag to move · scroll or pinch to zoom · shift-drag to turn · click anything to fly to it · Esc zooms out · Page Up and Down change floor{animate ? " · Space plays and pauses" : ""} · F for full screen. Each matrix lies flat, its inputs running front to back and its outputs left to right. {forward ? "Colour is each weight times the input it meets for the chosen token." : "Colour is the weight itself."} From afar, each pixel averages a block of weights; up close, every square is one weight, read straight from the buffers GPT-2 is running on.</p>
    </article>
  );
}

// The predictions after the last token, from a run whose lens covers every position.
function lastPredictions(trace: Trace, L: number, tok: Tokenizer) {
  return topAt(trace, L, trace.tokens.length - 1, tok);
}

function topAt(trace: Trace, L: number, q: number, tok: Tokenizer) {
  const lens = trace.lens!;
  const P = lens.positions.length;
  const K = lens.topK;
  const row = L * P + q;
  return Array.from({ length: K }, (_, k) => ({ piece: tok.piece(lens.ids[row * K + k]), p: Math.exp(lens.logprobs[row * K + k]) }));
}

// One floor's caption, with numbers from the run.
function describeFloor(f: number, L: number, trace: Trace, acts: Acts, q: number, tok: Tokenizer, dMlp: number) {
  const vec = (key: string, n = 768) => acts.data.subarray(acts.offsets.get(key)!, acts.offsets.get(key)! + n);
  const piece = q === 0 ? "⟨start⟩" : `“${word(tok.piece(trace.tokens[q]))}”`;
  if (f === 0) return `${piece} is token ${trace.tokens[q].toLocaleString()}: its row of 768 numbers, plus the row for position ${q}, starts the running summary.`;
  if (f <= L) {
    const n = Array.from(vec(`post${f - 1}`, dMlp)).filter((x) => x > 0.5).length;
    return `Attention: 12 heads gather from earlier tokens. Then the MLP: ${n.toLocaleString()} of ${dMlp.toLocaleString()} neurons fire. Both add to the running summary.`;
  }
  const top = topAt(trace, L, q, tok).map((g) => `“${word(g.piece)}” ${pct(g.p)}`);
  return `The same token table scores all 50,257 tokens. Top guesses: ${top.join(", ")}.`;
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3)).replace("-", "−");

function bitString(h: number) {
  const b = h.toString(2).padStart(16, "0");
  return `${b[0]} ${b.slice(1, 6)} ${b.slice(6)}  (sign · exponent · fraction)`;
}

const ACT_SIZE = (L: number, d: number, dMlp: number) => (L + 1) * d + L * (3 * d + dMlp) + d;

// The vectors the forward pass feeds each matrix, at one position.
function activations(trace: Trace, w: Weights, pos: number): Acts {
  const { n_layers: L, d_model: d, d_mlp: dMlp, ln_eps } = w.config;
  const T = trace.tokens.length;
  const data = new Float32Array(ACT_SIZE(L, d, dMlp));
  const offsets = new Map<string, number>();
  let at = 0;
  const put = (key: string, v: ArrayLike<number>) => {
    offsets.set(key, at);
    data.set(v, at);
    at += v.length;
  };
  const ln = (x: Float32Array, g: Float32Array, b: Float32Array) => {
    let m = 0;
    for (let i = 0; i < d; i++) m += x[i];
    m /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[i] - m) ** 2;
    const s = 1 / Math.sqrt(v / d + ln_eps);
    return Float32Array.from(x, (xi, i) => (xi - m) * s * g[i] + b[i]);
  };
  const resid = (l: number) => trace.resid!.subarray((l * T + pos) * d, (l * T + pos + 1) * d);
  for (let l = 0; l <= L; l++) put(`resid${l}`, resid(l));
  for (let l = 0; l < L; l++) {
    put(`ln1_${l}`, ln(resid(l), w.f32(`h.${l}.ln_1.weight`), w.f32(`h.${l}.ln_1.bias`)));
    put(`z${l}`, trace.z!.subarray((l * T + pos) * d, (l * T + pos + 1) * d));
    put(`ln2_${l}`, ln(trace.residMid!.subarray((l * T + pos) * d, (l * T + pos + 1) * d), w.f32(`h.${l}.ln_2.weight`), w.f32(`h.${l}.ln_2.bias`)));
    put(`post${l}`, trace.neurons!.subarray((l * T + pos) * dMlp, (l * T + pos + 1) * dMlp));
  }
  put("lnf", ln(resid(L), w.f32("ln_f.weight"), w.f32("ln_f.bias")));
  return { data, offsets, token: trace.tokens[pos], position: pos };
}

// A typical size that a few huge values can't drag up: GPT-2 has some.
const typical = (v: ArrayLike<number>) => {
  const a = Array.from(v, Math.abs).sort((x, y) => x - y);
  return a[Math.floor(a.length * 0.8)] || 1;
};

// How bright a typical value is, so every matrix uses its own range.
function scaleOf(t: Tile, w: Weights, acts: Acts) {
  let raw = 1;
  if (t.source.kind === "act") {
    const o = acts.offsets.get(t.source.act)!;
    raw = typical(acts.data.subarray(o, o + t.rows * t.cols));
  } else {
    const data = w.raw(t.source.tensor);
    const step = Math.max(1, Math.floor((t.rows * t.cols) / 20000));
    const sample: number[] = [];
    for (let i = 0; i < t.rows * t.cols; i += step) {
      const r = Math.floor(i / t.cols);
      const c = i % t.cols;
      sample.push(f16(data[r * t.source.stride + t.source.colStart + c]));
    }
    raw = typical(sample);
  }
  const f = t.forward;
  let fwd = raw;
  if (f.mode === "row" || f.mode === "col") {
    const o = acts.offsets.get(f.act)!;
    fwd = raw * typical(acts.data.subarray(o, o + (f.mode === "row" ? t.rows : t.cols)));
  }
  return { raw, fwd };
}

// The nearest matrix or vector under a ray; empty floor space doesn't count,
// so you can point through it at the floors below.
function surfaceHit(o: number[], dir: number[], tower: ReturnType<typeof buildTower>, top = Infinity) {
  let best: [number, number, number] | null = null;
  let bestD = Infinity;
  for (const f of tower.floors) {
    if (f.index > top) continue;
    const p = hitPlane(o, dir, f.y);
    if (!p || p[0] < f.x0 || p[0] > f.x1 || p[2] < f.z0 || p[2] > f.z1 || !tileAt(tower, p)) continue;
    const dd = Math.hypot(p[0] - o[0], p[1] - o[1], p[2] - o[2]);
    if (dd < bestD) {
      bestD = dd;
      best = p;
    }
  }
  return best;
}

function tileAt(tower: ReturnType<typeof buildTower>, p: number[]) {
  const floor = Math.round(p[1] / FLOOR);
  return tower.tiles.find((t) => t.floor === floor && p[0] >= t.x && p[0] < t.x + t.w && p[2] >= t.z && p[2] < t.z + t.d) ?? null;
}

function readAt(t: Tile, w: Weights, acts: Acts | null, p: number[]): Readout | null {
  const rc = toTensor(t, p[2] - t.z, p[0] - t.x);
  if (!rc) return null;
  const [row, col] = rc;
  let value = 0;
  let bits: number | null = null;
  if (t.source.kind === "act") {
    if (!acts) return null;
    value = acts.data[acts.offsets.get(t.source.act)! + row * t.cols + col];
  } else {
    bits = w.raw(t.source.tensor)[row * t.source.stride + t.source.colStart + col];
    value = f16(bits);
  }
  let input: number | null = null;
  const f = t.forward;
  if (acts && (f.mode === "row" || f.mode === "col")) input = acts.data[acts.offsets.get(f.act)! + (f.mode === "row" ? row : col)];
  if (acts && f.mode === "pick") input = row === (f.pick === "token" ? acts.token : acts.position) ? 1 : 0;
  return { tile: t, row, col, value, bits, input };
}

function pick(v: View, c: HTMLCanvasElement, at: { x: number; y: number }, tower: ReturnType<typeof buildTower>, w: Weights, acts: Acts | null): Readout | null {
  const aspect = c.clientWidth / c.clientHeight;
  const { origin, dir } = ray(v, aspect, (at.x / c.clientWidth) * 2 - 1, 1 - (at.y / c.clientHeight) * 2);
  const p = surfaceHit(origin, dir, tower, topFloor(v));
  if (!p) return null;
  const t = tileAt(tower, p);
  return t ? readAt(t, w, acts, p) : null;
}

// Floor names from afar, matrix names closer in, and numbers on each weight up close.
// `only`: the floor in focus, whose matrices get names; `active`: the floor the pass is on.
function labelsFor(v: View, c: HTMLCanvasElement, tower: ReturnType<typeof buildTower>, w: Weights, acts: Acts | null, only: number | null, active: number | null) {
  const W = c.clientWidth;
  const H = c.clientHeight;
  const aspect = W / H;
  const out: Label[] = [];
  const onScreen = (s: { x: number; y: number } | null) => s && s.x > -40 && s.x < W + 40 && s.y > -20 && s.y < H + 20;
  for (const f of tower.floors) {
    if (only !== null && f.index !== only) continue;
    const s = project(v, aspect, W, H, [f.x1, f.y, f.z0]);
    if (onScreen(s) && v.dist > 3000) out.push({ x: s!.x, y: s!.y, text: f.label, kind: "floor", active: f.index === active });
  }
  for (const t of tower.tiles) {
    if (t.d < 100 || t.w < 100 || only === null || t.floor !== only) continue;
    const a = project(v, aspect, W, H, [t.x, t.floor * FLOOR, t.z]);
    const b = project(v, aspect, W, H, [t.x + t.w, t.floor * FLOOR, t.z]);
    if (!a || !b || !onScreen(a)) continue;
    const span = Math.hypot(b.x - a.x, b.y - a.y);
    if (span > 70 && span < 4000) out.push({ x: a.x, y: a.y, text: t.name, kind: "tile" });
  }
  // Numbers on weights, once each one is big enough to read.
  const center = ray(v, aspect, 0, 0);
  const hit = surfaceHit(center.origin, center.dir, tower, topFloor(v));
  const t = hit && tileAt(tower, hit);
  if (hit && t) {
    const cell = t.fold ? 1 : t.w / t.cols;
    const cellRows = t.fold ? 1 : t.d / t.rows;
    const p0 = project(v, aspect, W, H, hit);
    const p1 = project(v, aspect, W, H, [hit[0] + cell, hit[1], hit[2]]);
    const px = p0 && p1 ? Math.hypot(p1.x - p0.x, p1.y - p0.y) : 0;
    if (px >= 34) {
      const reach = Math.ceil(Math.max(W, H) / px / 2) + 2;
      const cx = Math.floor((hit[0] - t.x) / cell);
      const cz = Math.floor((hit[2] - t.z) / cellRows);
      let n = 0;
      for (let i = -reach; i <= reach && n < 500; i++)
        for (let j = -reach; j <= reach && n < 500; j++) {
          const x = t.x + (cx + i + 0.5) * cell;
          const z = t.z + (cz + j + 0.5) * cellRows;
          if (x < t.x || x >= t.x + t.w || z < t.z || z >= t.z + t.d) continue;
          const s = project(v, aspect, W, H, [x, hit[1], z]);
          if (!s || s.x < -px || s.x > W + px || s.y < -px || s.y > H + px) continue;
          const r = readAt(t, w, acts, [x, hit[1], z]);
          if (!r) continue;
          const big = px >= 260;
          out.push({ x: s.x, y: s.y, text: fmt(r.value), big, bits: big && r.bits !== null ? bitString(r.bits).split("  ")[0] : undefined, kind: "cell" });
          n++;
        }
    }
  }
  return out;
}
