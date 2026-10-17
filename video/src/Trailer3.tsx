// Trailer 3: one weight pulls back to the whole model and lands in the app;
// then a small mystery (why does GPT-2 say "Mary"?), a fast tour, the answer,
// and GPT-2 writing a sentence, one pass up the tower per word. All footage is the real app (scripts/capture2.mjs, scripts/film3.mjs);
// the music and sound effects are added when rendering
// (scripts/sfx3.py, then scripts/render.mjs --comp Trailer3).

import type { ReactNode } from "react";
import { AbsoluteFill, Easing, Img, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import beats from "./beats3.json";
import cut from "./cut3.json";
import opening from "./opening3.json";
import { loadFonts, MONO, SANS } from "./fonts";

export const FPS3 = 30;
export const END3 = Math.round(beats.duration * FPS3);
const f = (s: number) => Math.round(s * FPS3);
const B = beats.bars;
const HIT = beats.finalHit;
const CLIPS = cut.clips as Record<string, { frames: number; from: number; to: number; map: number[][] }>;

const BG = "#0d0f13";
const INK = "#e9ebf0";
const MUTED = "#a0a7b4";
const ATTN = "#ff7a45";
const MLP = "#2cc9ae";
const RESID = "#7b93ff";
const GOLD = "#ffe9a8";
const STAMP = "#ff5a4f";

// The app's tower panel in the hook recording, where the opening lands.
const PANEL = { x: 356, y: 298.6, w: 1208, h: 760 };
// Head cells in the Lab's grid (layer columns, head rows), in page pixels.
const cellAt = (id: string) => {
  const [l, h] = id.split(".").map(Number);
  return { x: 423 + 67.36 * l, y: 187 + 54.64 * h };
};
const CREW = ["9.9", "9.6", "10.0", "9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"];

export const Trailer3 = () => {
  loadFonts();
  const tour = [
    { name: "tokens", tag: "tokens", color: RESID, cam: [{ t: 0, cx: 800, cy: 120, z: 2.1 }, { t: 2, cx: 820, cy: 110, z: 2.3 }], aside: "feat · ur · ama" },
    { name: "lens.png", tag: "layers", color: RESID, cam: [{ t: 0, cx: 760, cy: 360, z: 1.8 }, { t: 2, cx: 1150, cy: 350, z: 2.0 }], aside: "then, then… cheese… jelly." },
    { name: "neurons", tag: "neurons", color: MLP, cam: [{ t: 0, cx: 760, cy: 420, z: 1.5 }, { t: 2, cx: 880, cy: 380, z: 1.75 }] },
    { name: "features.png", tag: "features", color: "#a080ff", cam: [{ t: 0, cx: 1200, cy: 330, z: 1.6 }, { t: 2, cx: 1370, cy: 360, z: 1.95 }], aside: "#7626: rain" },
    { name: "steer", tag: "steering", color: "#a080ff", cam: [{ t: 0, cx: 1000, cy: 480, z: 1.45 }, { t: 2, cx: 1360, cy: 500, z: 1.9 }] },
    { name: "map", tag: "50,257 tokens", color: MLP, cam: [{ t: 0, cx: 960, cy: 600, z: 1.3 }, { t: 2, cx: 900, cy: 620, z: 1.6 }] },
    { name: "puzzle", tag: "puzzles", color: ATTN, cam: [{ t: 0, cx: 900, cy: 420, z: 1.35 }, { t: 1.6, cx: 1130, cy: 340, z: 1.8 }, { t: 4, cx: 1150, cy: 340, z: 1.95 }], aside: "make it say “John”" },
  ];
  const tourAt = [B[22], B[23], B[24], B[25], B[26], B[27], B[28], B[30]];

  return (
    <AbsoluteFill style={{ background: BG, fontFamily: SANS }}>
      {/* 1. One number; the pass lights it; powers of ten out to the app. */}
      <Sequence from={0} durationInFrames={f(B[6])} layout="none">
        <Opening />
      </Sequence>

      {/* 2. Typing the question. */}
      <Sequence from={f(CLIPS.hook.from)} durationInFrames={f(CLIPS.hook.to) - f(CLIPS.hook.from)} layout="none">
        <Clip name="hook" cam={[{ t: 0, cx: 960, cy: 540, z: 1 }, { t: 0.55, cx: 960, cy: 170, z: 1.5 }, { t: 3.5, cx: 900, cy: 170, z: 1.55 }, { t: 3.97, cx: 960, cy: PANEL.y + PANEL.h / 2, z: 1080 / PANEL.h }]}>
          <Sparkles at={3.32} x={960} y={44} small />
        </Clip>
        <Stamp text="why “Mary”?" at={3.4} until={3.97} color={GOLD} size={150} />
      </Sequence>

      {/* 3. The forward pass climbs to "Mary". */}
      <Sequence from={f(CLIPS.hook.to - 0.15)} durationInFrames={f(B[13] - CLIPS.hook.to + 0.15)} layout="none">
        <FadeIn frames={5}>
          <Frames dir="footage2/climb" count={300} />
        </FadeIn>
        <CaseLabel text="the case: why “Mary”?" at={0.4} until={9.9} />
        <Sparkles at={8.45} x={860} y={92} />
      </Sequence>

      {/* 4. Investigate: dive to layer 9, then head 9.9 in the app. */}
      <Sequence from={f(B[13])} durationInFrames={f(B[15] - B[13])} layout="none">
        <ZoomThrough from={B[15] - B[13] - 0.3}>
          <Frames dir="footage3/dive" count={120} />
          <Band at={3.2} />
        </ZoomThrough>
        <CaseLabel text="suspect: layer 9, head 9.9" at={2.4} until={3.9} />
      </Sequence>
      <Sequence from={f(CLIPS.attention.from)} durationInFrames={f(CLIPS.attention.to - CLIPS.attention.from)} layout="none">
        <Clip name="attention" cam={[{ t: 0, cx: 1080, cy: 270, z: 2.7 }, { t: 0.5, cx: 1080, cy: 270, z: 1.75 }, { t: 4, cx: 1000, cy: 270, z: 1.95 }]}>
          <Ripple at={1.997} x={1382} y={331} />
        </Clip>
        <CaseLabel text="clue 1: head 9.9 stares at “Mary”" at={0.6} until={3.9} />
      </Sequence>
      <Sequence from={f(CLIPS.votes.from)} durationInFrames={f(CLIPS.votes.to - CLIPS.votes.from)} layout="none">
        <Clip name="votes" cam={[{ t: 0, cx: 1000, cy: 480, z: 1.35 }, { t: 2, cx: 1180, cy: 520, z: 1.55 }]}>
          <Ripple at={1.04} x={1028} y={697} />
        </Clip>
        <CaseLabel text="clue 2: …and it votes for her" at={0.15} until={1.95} />
      </Sequence>

      {/* 5. The test: unplug the suspects. It fights back. */}
      <Sequence from={f(CLIPS.unplug.from)} durationInFrames={f(CLIPS.unplug.to - CLIPS.unplug.from)} layout="none">
        <Clip name="unplug" cam={[{ t: 0, cx: 840, cy: 470, z: 1.3 }, { t: 2.6, cx: 900, cy: 470, z: 1.4 }, { t: 3.4, cx: 1290, cy: 330, z: 2.05 }, { t: 7.9, cx: 1270, cy: 340, z: 2.2 }]}>
          {["9.9", "9.6", "10.0"].map((id, i) => <Ripple key={id} at={[0.497, 1.493, 2.497][i]} {...cellAt(id)} />)}
        </Clip>
        <CaseLabel text="test: unplug the suspects" at={0.1} until={3.9} />
        <Shout text="…it fights back." at={B[20] - CLIPS.unplug.from + 0.15} until={7.9} />
      </Sequence>

      {/* 6. Meanwhile, the rest of the museum: one bar each. */}
      {tour.map((c, i) => (
        <Sequence key={c.name} from={f(tourAt[i])} durationInFrames={f(tourAt[i + 1]) - f(tourAt[i])} layout="none">
          <Clip name={c.name} cam={c.cam} tilt={i % 2 ? 0.7 : -0.7} />
          <Vignette />
          <Tag text={c.tag} color={c.color} at={0.05} until={tourAt[i + 1] - tourAt[i] - 0.08} />
          {c.aside && <Aside text={c.aside} at={0.7} />}
        </Sequence>
      ))}
      {tour.map((_, i) => <Hop key={i} at={tourAt[i]} flip={i % 2 === 1} />)}

      {/* 7. One last test: the backups too. */}
      <Sequence from={f(CLIPS.backups.from)} durationInFrames={f(beats.resume) - f(CLIPS.backups.from)} layout="none">
        <Clip name="backups" cam={[{ t: 0, cx: 1390, cy: 380, z: 2.2 }, { t: 3.9, cx: 1395, cy: 470, z: 2.6 }, { t: 4.4, cx: 1340, cy: 400, z: 1.9 }, { t: 7.6, cx: 900, cy: 500, z: 1.2 }, { t: 11.9, cx: 920, cy: 500, z: 1.28 }]}>
          <Ripple at={3.97} x={1395} y={525} color={STAMP} />
          {CREW.map((id, i) => <Ring key={id} at={B[34] - CLIPS.backups.from + 0.1 + i * 0.25} {...cellAt(id)} />)}
        </Clip>
        <Vignette strong until={3.97} />
        <CaseLabel text="one last test…" at={0.3} until={3.9} />
        <BigDrop at={B[32] - CLIPS.backups.from + 0.35} until={B[34] - CLIPS.backups.from - 0.1} />
        <CaseLabel text="solved: a crew of 11 heads carries “Mary”" at={B[34] - CLIPS.backups.from + 0.2} until={beats.stop - CLIPS.backups.from} />
        {/* The music stops; the frame holds under the stamp. */}
        <Dim at={beats.stop - CLIPS.backups.from} />
        <Stamp text="CASE CLOSED" at={beats.stop - CLIPS.backups.from} until={99} color={STAMP} size={170} slam />
      </Sequence>
      <Flash at={B[32]} />

      {/* 8. GPT-2 writes: the tower lights up once per word. */}
      <Sequence from={f(beats.resume)} durationInFrames={f(HIT) - f(beats.resume) + 1} layout="none">
        <Frames dir="footage3/generate" count={247} />
        <Writing />
      </Sequence>
      <Flash at={beats.resume + 0.02} soft />
      <Sequence from={f(HIT)} durationInFrames={END3 - f(HIT)} layout="none">
        <EndCard />
      </Sequence>
      <Flash at={HIT} />
    </AbsoluteFill>
  );
};

// --- The opening -------------------------------------------------------------

const BITS = "0 01011 1100010010";
// The pull-back passes these levels; the counter follows the film's camera.
const LEVELS = [
  { dist: 0.8, n: 1, label: "" },
  { dist: 70, n: 768, label: "a row" },
  { dist: 1500, n: 589824, label: "a matrix" },
  { dist: 8500, n: 7087872, label: "a layer" },
  { dist: 30000, n: 124439808, label: "all of GPT-2 small" },
];
const DIST = opening.dist;
const COUNT_FROM = B[1] + 0.35;
const reachedAt = LEVELS.map((l) => Math.max(0, DIST.findIndex((d) => d >= l.dist)));
function countAt(d: number) {
  if (d <= LEVELS[0].dist) return 1;
  for (let i = 1; i < LEVELS.length; i++) {
    const a = LEVELS[i - 1];
    const b = LEVELS[i];
    if (d <= b.dist) {
      const k = (Math.log(d) - Math.log(a.dist)) / (Math.log(b.dist) - Math.log(a.dist));
      return Math.round(Math.exp(Math.log(a.n) + (Math.log(b.n) - Math.log(a.n)) * k));
    }
  }
  return LEVELS[LEVELS.length - 1].n;
}

function Opening() {
  const frame = useCurrentFrame();
  const t = frame / FPS3;
  // Shrink the film into the app's tower panel, over the page.
  const k = interpolate(t, [B[5] + 1.5, B[6] - 0.05], [0, 1], { easing: Easing.bezier(0.6, 0, 0.25, 1), extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const s = 1 + (PANEL.h / 1080 - 1) * k;
  const cy = 540 + (PANEL.y + PANEL.h / 2 - 540) * k;
  const side = ((1920 * s - PANEL.w) / 2 / s) * k;
  const filmOut = interpolate(t, [B[6] - 0.35, B[6]], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  // Dim until the pass reaches the number.
  const lit = interpolate(t, [0, 0.5, B[1] - 0.1, B[1] + 0.2], [0, 0.4, 0.4, 1], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill>
      {k > 0 && <Img src={staticFile("footage3/hook/0000.jpg")} style={{ position: "absolute", inset: 0, width: 1920, height: 1080, opacity: Math.min(1, k * 2) }} />}
      <AbsoluteFill style={{ transform: `translateY(${cy - 540}px) scale(${s})`, clipPath: `inset(0 ${side}px round ${12 * k}px)`, opacity: filmOut, filter: `brightness(${lit})` }}>
        <Frames dir="footage3/opening" count={361} />
      </AbsoluteFill>
      <OneNumber />
      <ScaleCount />
    </AbsoluteFill>
  );
}

// The weight's 16 bits tick in, become its value, glow when the pass hits,
// then shrink onto the label in the 3D view.
function OneNumber() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const handoff = interpolate(t, [B[1] + 0.45, B[1] + 0.85], [0, 1], { easing: Easing.bezier(0.5, 0, 0.3, 1), extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  if (handoff >= 1) return null;
  const value = spring({ frame: frame - f(1.45), fps, config: { damping: 11, stiffness: 160 } });
  const glow = interpolate(t, [B[1] - 0.1, B[1] + 0.15, B[1] + 1.2], [0, 1, 0.25], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  let bit = 0;
  const bits = BITS.split("").map((ch, i) => {
    if (ch === " ") return <span key={i}> </span>;
    const at = 0.3 + bit++ * 0.07;
    const on = t >= at;
    const fresh = on && t - at < 0.12;
    return <span key={i} style={{ display: "inline-block", opacity: on ? 1 : 0, color: fresh ? GOLD : INK, transform: `translateY(${fresh ? -6 : 0}px)` }}>{ch}</span>;
  });
  const shrink = 1 - 0.93 * handoff;
  const y = -12 * handoff;
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: 1 - Math.min(1, handoff * 1.6) }}>
      <div style={{ transform: `translateY(${y}px) scale(${shrink})`, display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ font: `600 220px ${MONO}`, color: INK, height: 260 * value, opacity: value, transform: `scale(${0.7 + 0.3 * value})`, textShadow: `0 0 ${80 * glow}px rgba(255,214,120,${0.9 * glow}), 0 10px 60px rgba(0,0,0,0.6)`, lineHeight: "260px", overflow: "visible" }}>0.110</div>
        <div style={{ font: `500 ${interpolate(value, [0, 1], [64, 40])}px ${MONO}`, color: MUTED, letterSpacing: "0.1em" }}>{bits}</div>
      </div>
      <Sparkles at={B[1] + 0.05} x={960} y={500} />
    </AbsoluteFill>
  );
}

// Powers of ten: how many numbers are in view, rolling with the camera.
function ScaleCount() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  if (t < COUNT_FROM) return null;
  const n = countAt(DIST[Math.min(DIST.length - 1, frame)]);
  let i = 0;
  while (i + 1 < LEVELS.length && frame >= reachedAt[i + 1]) i++;
  const last = i === LEVELS.length - 1;
  const pop = i > 0 ? spring({ frame: frame - reachedAt[i], fps, config: { damping: 8, stiffness: 220 } }) : 1;
  const inV = spring({ frame: frame - f(COUNT_FROM), fps, config: { damping: 14, stiffness: 140 } });
  const out = interpolate(t, [B[5] + 1.6, B[5] + 1.85], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const toy = spring({ frame: frame - reachedAt[LEVELS.length - 1] - 6, fps, config: { damping: 12, stiffness: 150 } });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 96, opacity: inV * out }}>
      {i > 0 && (
        <div style={{ marginBottom: 6, padding: "4px 18px", borderRadius: 999, background: "rgba(13,15,19,0.8)", border: `2px solid ${last ? GOLD : RESID}`, color: last ? GOLD : INK, font: `600 34px ${SANS}`, opacity: Math.min(1, pop * 1.4), transform: `scale(${0.6 + 0.4 * pop}) rotate(${(1 - pop) * -8}deg)` }}>{LEVELS[i].label}</div>
      )}
      <div style={{ display: "flex", alignItems: "baseline", gap: 22 }}>
        <div style={{ font: `600 104px ${MONO}`, color: last ? GOLD : INK, fontVariantNumeric: "tabular-nums", textShadow: "0 8px 40px rgba(0,0,0,0.85)" }}>{n.toLocaleString("en-US")}</div>
        <div style={{ font: `500 40px ${SANS}`, color: MUTED, textShadow: "0 4px 20px rgba(0,0,0,0.9)" }}>{n === 1 ? "number" : "numbers"}</div>
      </div>
      {last && (
        <div style={{ marginTop: 8, font: `500 34px ${SANS}`, color: MUTED, opacity: toy, transform: `translateY(${(1 - toy) * 16}px)`, textShadow: "0 4px 20px rgba(0,0,0,0.9)" }}>
          from 2019. <span style={{ color: GOLD }}>a toy, by today’s standards.</span>
        </div>
      )}
    </AbsoluteFill>
  );
}

// --- Footage -----------------------------------------------------------------

// A folder of rendered frames at 30 fps.
function Frames({ dir, count }: { dir: string; count: number }) {
  const frame = useCurrentFrame();
  const i = Math.min(count - 1, frame);
  return <Img src={staticFile(`${dir}/${String(i).padStart(4, "0")}.jpg`)} style={{ position: "absolute", inset: 0, width: 1920, height: 1080 }} />;
}

type Key = { t: number; cx: number; cy: number; z: number };
const ease = Easing.bezier(0.45, 0, 0.2, 1);
function camAt(keys: Key[], t: number) {
  if (t <= keys[0].t) return keys[0];
  for (let i = 0; i + 1 < keys.length; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (t <= b.t) {
      const k = ease((t - a.t) / (b.t - a.t));
      return { t, cx: a.cx + (b.cx - a.cx) * k, cy: a.cy + (b.cy - a.cy) * k, z: Math.exp(Math.log(a.z) + (Math.log(b.z) - Math.log(a.z)) * k) };
    }
  }
  return keys[keys.length - 1];
}
function remap(map: number[][], t: number) {
  if (t <= map[0][0]) return map[0][1];
  for (let i = 0; i + 1 < map.length; i++) {
    const [a, x] = map[i];
    const [b, y] = map[i + 1];
    if (t <= b) return x + ((y - x) * (t - a)) / (b - a);
  }
  return map[map.length - 1][1];
}

// A recording (or screenshot, if the name has an extension) of the app under
// a moving camera. Children are drawn in page pixels and move with it.
function Clip({ name, cam, tilt = 0, children }: { name: string; cam: Key[]; tilt?: number; children?: ReactNode }) {
  const frame = useCurrentFrame();
  const t = frame / FPS3;
  const c = camAt(cam, t);
  const z = c.z;
  const cx = clamp(c.cx, 960 / z, 1920 - 960 / z);
  const cy = clamp(c.cy, 540 / z, 1080 - 540 / z);
  let src: string;
  if (name.includes(".")) src = `footage3/${name}`;
  else {
    const clip = CLIPS[name];
    const ct = remap(clip.map, t);
    src = `footage3/${name}/${String(Math.min(clip.frames - 1, Math.max(0, Math.round(ct * FPS3)))).padStart(4, "0")}.jpg`;
  }
  const rot = tilt * Math.exp(-frame / 8);
  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: 1920, height: 1080, transformOrigin: "0 0", transform: `translate(${960 - cx * z}px, ${540 - cy * z}px) scale(${z}) rotate(${rot}deg)` }}>
        <Img src={staticFile(src)} style={{ position: "absolute", inset: 0, width: 1920, height: 1080 }} />
        {children}
      </div>
    </AbsoluteFill>
  );
}

function FadeIn({ frames, children }: { frames: number; children: ReactNode }) {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ opacity: Math.min(1, frame / frames) }}>{children}</AbsoluteFill>;
}

// The last moments of a shot rush toward the camera and fade.
function ZoomThrough({ from, children }: { from: number; children: ReactNode }) {
  const frame = useCurrentFrame();
  const k = interpolate(frame / FPS3, [from, from + 0.3], [0, 1], { easing: Easing.in(Easing.cubic), extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <AbsoluteFill style={{ transform: `scale(${1 + 1.6 * k})`, opacity: 1 - k, filter: k > 0 ? `blur(${6 * k}px)` : undefined }}>{children}</AbsoluteFill>;
}

// --- Overlays ----------------------------------------------------------------

// Darkens the frame from `at` on, quickly.
function Dim({ at }: { at: number }) {
  const frame = useCurrentFrame();
  const k = interpolate(frame, [f(at), f(at) + 3], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <AbsoluteFill style={{ background: `radial-gradient(ellipse at center, rgba(0,0,0,${0.25 * k}) 30%, rgba(0,0,0,${0.75 * k}) 100%)` }} />;
}

// The sentence writes itself beside the tower: each word lands as its pass
// reaches the roof (the same timing as scripts/film3.mjs).
const ASK = "I asked the computer what it was thinking, and it said";
const GEN = [" it", " was", " thinking", " about", " the", " computer", "."];
const FILL = B.find((b) => b > beats.resume + 5)!;
const PER = (FILL - beats.resume) / (GEN.length - 1);
function Writing() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const inV = spring({ frame, fps, config: { damping: 14, stiffness: 160 } });
  const fade = interpolate(t, [HIT - beats.resume - 0.25, HIT - beats.resume], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "flex-end", padding: "0 64px 0 0", opacity: fade }}>
      <div style={{ width: 700, padding: "28px 36px", borderRadius: 22, background: "rgba(13,15,19,0.86)", border: "1px solid rgba(255,255,255,0.1)", font: `500 50px ${SANS}`, lineHeight: 1.3, opacity: inV, transform: `translateX(${(1 - inV) * 60}px)` }}>
        <span style={{ color: MUTED }}>{ASK}</span>
        {GEN.map((w, k) => {
          const landed = t - (k * PER + PER * 0.78);
          if (landed < 0) return null;
          const s = spring({ frame: f(landed), fps, config: { damping: 8, stiffness: 220 } });
          const gold = k >= GEN.length - 2;
          return <span key={k} style={{ display: "inline-block", whiteSpace: "pre", color: gold ? GOLD : INK, transform: `translateY(${(1 - s) * -28}px)`, opacity: Math.min(1, s * 1.5) }}>{w}</span>;
        })}
      </div>
    </AbsoluteFill>
  );
}

// Head 9.9's 64 columns of layer 9's queries, outlined where the dive ends.
function Band({ at }: { at: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < f(at)) return null;
  const s = spring({ frame: frame - f(at), fps, config: { damping: 10, stiffness: 200 } });
  const pts = "939,194 994,194 987,916 923,916";
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
      <polygon points={pts} fill={`rgba(255,214,120,${0.18 * s})`} stroke={GOLD} strokeWidth={5} opacity={s} style={{ filter: "drop-shadow(0 0 14px rgba(255,214,120,0.9))" }} />
      <text x={966} y={176 - 10 * s} textAnchor="middle" style={{ font: `600 40px ${MONO}`, fill: GOLD, opacity: s }}>9.9</text>
    </svg>
  );
}

function Vignette({ strong = false, until }: { strong?: boolean; until?: number }) {
  const frame = useCurrentFrame();
  const s = strong && (until === undefined || frame / FPS3 < until) ? 1 : 0;
  return <AbsoluteFill style={{ background: `radial-gradient(ellipse at center, rgba(0,0,0,0) ${55 - 20 * s}%, rgba(0,0,0,${0.5 + 0.25 * s}) 100%)` }} />;
}

function Flash({ at, soft = false }: { at: number; soft?: boolean }) {
  return (
    <Sequence from={f(at)} durationInFrames={9} layout="none">
      <FlashInner soft={soft} />
    </Sequence>
  );
}
function FlashInner({ soft }: { soft: boolean }) {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ background: "white", opacity: interpolate(frame, [0, 9], [soft ? 0.3 : 0.7, 0], { extrapolateRight: "clamp" }) }} />;
}

// A case-file note, typed out top left.
function CaseLabel({ text, at, until }: { text: string; at: number; until: number }) {
  const frame = useCurrentFrame();
  const t = frame / FPS3;
  if (t < at || t > until) return null;
  const n = Math.min(text.length, Math.floor((t - at) / 0.028));
  const out = interpolate(t, [until - 0.15, until], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const cursor = n < text.length || Math.floor(t * 3) % 2 === 0;
  return (
    <AbsoluteFill style={{ padding: "64px 76px", alignItems: "flex-start", opacity: out }}>
      <div style={{ font: `600 44px ${MONO}`, color: INK, background: "rgba(13,15,19,0.88)", padding: "12px 22px", borderRadius: 10, borderLeft: `6px solid ${STAMP}`, boxShadow: "0 10px 40px rgba(0,0,0,0.5)" }}>
        {text.slice(0, n)}
        <span style={{ color: STAMP, opacity: cursor ? 1 : 0 }}>▍</span>
      </div>
    </AbsoluteFill>
  );
}

// A rubber stamp that slams down with a little bounce.
function Stamp({ text, at, until, color, size, slam = false }: { text: string; at: number; until: number; color: string; size: number; slam?: boolean }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  if (t < at || t > until) return null;
  const s = spring({ frame: frame - f(at), fps, config: { damping: 9, stiffness: 220 } });
  const k = frame - f(at);
  const drop = interpolate(k, [0, 4], [2.2, 1], { easing: Easing.in(Easing.quad), extrapolateRight: "clamp" });
  const bump = k >= 4 ? 0.05 * Math.sin(((k - 4) / 6) * Math.PI) * (k < 10 ? 1 : 0) : 0;
  const scale = slam ? drop + bump : 0.5 + 0.5 * s;
  const shake = slam && k >= 4 && k < 12 ? (12 - k) * 2.2 * (k % 2 ? 1 : -1) : 0;
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", transform: `translate(${shake}px, ${-shake * 0.6}px)` }}>
      <div style={{
        font: `600 ${size}px ${slam ? MONO : SANS}`, color, letterSpacing: slam ? "0.06em" : "-0.03em", padding: slam ? "6px 40px" : 0,
        border: slam ? `10px solid ${color}` : undefined, borderRadius: 18, transform: `rotate(${slam ? -7 : -4}deg) scale(${scale})`, opacity: slam ? Math.min(1, (k + 1) / 2) : Math.min(1, s * 2),
        textShadow: slam ? undefined : "0 10px 60px rgba(0,0,0,0.9)", boxShadow: slam ? `0 0 80px ${color}55` : undefined, background: slam ? "rgba(13,15,19,0.55)" : undefined,
      }}>{text}</div>
    </AbsoluteFill>
  );
}

function Shout({ text, at, until }: { text: string; at: number; until: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  if (t < at || t > until) return null;
  const s = spring({ frame: frame - f(at), fps, config: { damping: 7, stiffness: 200 } });
  const shake = Math.sin(t * 40) * 4 * Math.exp(-(t - at) * 4);
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "flex-start", padding: "0 0 110px 110px" }}>
      <div style={{ font: `600 96px ${SANS}`, color: GOLD, letterSpacing: "-0.03em", opacity: s, transform: `translateX(${shake}px) rotate(${-3 + (1 - s) * 8}deg) scale(${0.7 + 0.3 * s})`, textShadow: "0 10px 50px rgba(0,0,0,0.95)" }}>{text}</div>
    </AbsoluteFill>
  );
}

// Mary's chance after the backups go too.
function BigDrop({ at, until }: { at: number; until: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  if (t < at || t > until) return null;
  const a = spring({ frame: frame - f(at), fps, config: { damping: 12, stiffness: 160 } });
  const b = spring({ frame: frame - f(at + 0.5), fps, config: { damping: 6, stiffness: 240 } });
  const fall = Math.max(0, t - at - 0.5);
  const out = interpolate(t, [until - 0.2, until], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: out, background: `rgba(5,6,9,${0.55 * a})` }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 36, padding: "26px 54px", borderRadius: 28, background: "rgba(13,15,19,0.9)", boxShadow: "0 20px 80px rgba(0,0,0,0.6)", opacity: a, transform: `scale(${0.8 + 0.2 * a})` }}>
        <span style={{ font: `600 120px ${MONO}`, color: INK }}>Mary</span>
        <span style={{ font: `600 120px ${MONO}`, color: MUTED, textDecoration: b > 0.5 ? "line-through" : undefined, textDecorationColor: STAMP }}>68%</span>
        <span style={{ font: `600 90px ${SANS}`, color: MUTED }}>→</span>
        <span style={{ display: "inline-block", font: `600 160px ${MONO}`, color: STAMP, opacity: b, transform: `scale(${0.4 + 0.6 * b}) translateY(${Math.min(30, fall * 20)}px) rotate(${Math.min(8, fall * 5)}deg)` }}>2%</span>
      </div>
    </AbsoluteFill>
  );
}

// A small pill that springs in, a little tilted, and leaves on time.
function Tag({ text, color, at, until }: { text: string; color: string; at: number; until: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - f(at), fps, config: { damping: 9, stiffness: 200 } });
  const out = interpolate(frame, [f(until) - 4, f(until)], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "flex-start", padding: "0 76px 76px" }}>
      <div style={{
        padding: "10px 28px", borderRadius: 999, background: "rgba(13,15,19,0.9)", border: `2px solid ${color}`, color,
        font: `600 46px ${SANS}`, letterSpacing: "-0.01em", opacity: Math.min(1, s * 1.3) * out,
        transform: `scale(${0.6 + 0.4 * s}) rotate(${(1 - s) * -10 - 2}deg)`, boxShadow: `0 0 40px ${color}55`,
      }}>{text}</div>
    </AbsoluteFill>
  );
}

function Aside({ text, at }: { text: string; at: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - f(at), fps, config: { damping: 12, stiffness: 180 } });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "flex-end", padding: "0 90px 80px" }}>
      <div style={{ font: `600 46px ${MONO}`, color: INK, opacity: s, transform: `translateY(${(1 - s) * 30}px)`, padding: "10px 22px", borderRadius: 14, background: "rgba(13,15,19,0.88)" }}>{text}</div>
    </AbsoluteFill>
  );
}

// Rings where the cursor clicks, in page pixels.
function Ripple({ at, x, y, color = GOLD }: { at: number; x: number; y: number; color?: string }) {
  const frame = useCurrentFrame() - f(at);
  if (frame < 0 || frame > 14) return null;
  const k = frame / 14;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
      {[0, 0.3].map((d) => {
        const kk = Math.max(0, (k - d) / (1 - d));
        return <circle key={d} cx={x} cy={y} r={8 + 46 * (1 - (1 - kk) ** 3)} fill="none" stroke={color} strokeWidth={4 * (1 - kk)} opacity={kk > 0 ? 1 - kk : 0} />;
      })}
    </svg>
  );
}

// A ring that pops onto one of the crew's cells and stays.
function Ring({ at, x, y }: { at: number; x: number; y: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < f(at)) return null;
  const s = spring({ frame: frame - f(at), fps, config: { damping: 7, stiffness: 260 } });
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
      <rect x={x - 33} y={y - 27} width={66} height={54} rx={8} fill="none" stroke={GOLD} strokeWidth={4} transform={`translate(${x} ${y}) scale(${s}) translate(${-x} ${-y})`} style={{ filter: "drop-shadow(0 0 10px rgba(255,214,120,0.8))" }} />
    </svg>
  );
}

// A little burst of gold.
function Sparkles({ at, x, y, small = false }: { at: number; x: number; y: number; small?: boolean }) {
  const frame = useCurrentFrame() - f(at);
  if (frame < 0 || frame > 26) return null;
  const k = frame / 26;
  const r0 = small ? 0.45 : 1;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
      {Array.from({ length: 14 }, (_, i) => {
        const a = (i / 14) * Math.PI * 2 + 0.3;
        const r = (30 + 150 * (1 - (1 - k) ** 3) * (0.7 + (0.3 * ((i * 7) % 5)) / 4)) * r0;
        const size = (i % 3 === 0 ? 7 : 4) * (1 - k) * (small ? 0.7 : 1);
        return <circle key={i} cx={x + Math.cos(a) * r} cy={y + Math.sin(a) * r * 0.8 + 40 * k * k * r0} r={size} fill={i % 4 === 0 ? ATTN : GOLD} opacity={1 - k * 0.6} />;
      })}
    </svg>
  );
}

// The site's hop arc, swooping across to carry in the next shot.
function Hop({ at, flip }: { at: number; flip: boolean }) {
  return (
    <Sequence from={f(at) - 4} durationInFrames={12} layout="none">
      <HopInner flip={flip} />
    </Sequence>
  );
}
function HopInner({ flip }: { flip: boolean }) {
  const frame = useCurrentFrame();
  const k = interpolate(frame, [0, 11], [0, 1], { easing: Easing.bezier(0.5, 0, 0.3, 1) });
  const head = Math.min(1, k * 1.6);
  const tail = Math.max(0, k * 1.6 - 0.6);
  return (
    <AbsoluteFill style={{ transform: flip ? "scaleX(-1)" : undefined }}>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080">
        <path d="M-100 1000 C 400 -250, 1500 -250, 2020 1000" pathLength={1} fill="none" stroke={ATTN} strokeWidth={34} strokeLinecap="round" strokeDasharray={`${head - tail} 2`} strokeDashoffset={-tail} opacity={0.95} />
      </svg>
    </AbsoluteFill>
  );
}

// The end card: the wordmark draws, the dot pops, and the address types itself.
const CH = 33.6;
const A = [2, 6, 8].map((i) => (i + 0.5) * CH);
const URL_TEXT = "shuw.github.io/featurama";
function EndCard() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const draw = (delay: number, dur = 16) => interpolate(frame - delay, [0, dur], [1, 0], { easing: Easing.bezier(0.4, 0, 0.2, 1), extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const pop = spring({ frame: frame - 10, fps, config: { damping: 6, stiffness: 200 } });
  const rise = (delay: number) => spring({ frame: frame - delay, fps, config: { damping: 15, stiffness: 150 } });
  const typed = Math.max(0, Math.min(URL_TEXT.length, Math.floor((frame - 30) / 1.5)));
  const cursor = Math.floor(frame / 8) % 2 === 0;
  return (
    <AbsoluteFill style={{ background: BG, justifyContent: "center", alignItems: "center" }}>
      <svg width={316 * 3.2} height={118 * 3.2} viewBox="-6 16 316 118" style={{ overflow: "visible" }}>
        <path d="M0 128H303" pathLength={1} fill="none" stroke={RESID} strokeWidth={2.5} strokeDasharray={1} strokeDashoffset={draw(0)} />
        <path d={`M${A[2]} 84 C ${A[2]} 50, ${A[1]} 50, ${A[1]} 84`} pathLength={1} fill="none" stroke={ATTN} strokeWidth={4} strokeLinecap="round" strokeDasharray={1} strokeDashoffset={draw(6)} />
        <path d={`M${A[2]} 84 C ${A[2]} 8, ${A[0]} 8, ${A[0]} 84`} pathLength={1} fill="none" stroke={ATTN} strokeWidth={3} strokeLinecap="round" opacity={0.75} strokeDasharray={1} strokeDashoffset={draw(12)} />
        <circle cx={A[2]} cy={84} r={5.5 * pop} fill={MLP} />
        <text x="0" y="118" style={{ font: `600 56px ${MONO}`, fill: INK }}>featurama</text>
      </svg>
      <div style={{ marginTop: 34, fontSize: 58, fontWeight: 600, color: INK, letterSpacing: "-0.02em", opacity: rise(16), transform: `translateY(${(1 - rise(16)) * 26}px)` }}>Look inside a real language model.</div>
      <div style={{ marginTop: 28, padding: "14px 30px", borderRadius: 16, background: "rgba(255,122,69,0.1)", border: `2px solid ${ATTN}`, font: `600 54px ${MONO}`, color: ATTN, opacity: rise(26), minWidth: 900, textAlign: "center" }}>
        {URL_TEXT.slice(0, typed)}<span style={{ opacity: cursor ? 1 : 0 }}>▍</span>
      </div>
      <div style={{ marginTop: 22, fontSize: 28, color: MUTED, opacity: rise(56) }}>GPT-2 small (124M, 2019) · free · runs entirely in your browser</div>
    </AbsoluteFill>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, v));
}
