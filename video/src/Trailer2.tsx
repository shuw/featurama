// Trailer 2: one weight, the whole tower, a forward pass, a quick tour of the
// app and GPT-2 writing a sentence, cut to "Breakbeat Groove". All footage is
// the real app (scripts/capture.mjs, scripts/film.mjs); the music is added
// when rendering (scripts/render.mjs --comp Trailer2).

import { AbsoluteFill, Easing, Img, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import beats from "./beats2.json";
import { loadFonts, MONO, SANS } from "./fonts";

export const FPS2 = 30;
export const END2 = 60 * FPS2;
const f = (s: number) => Math.round(s * FPS2);
const sec = (label: string) => beats.sections.find((s) => s.label === label)!;
const INTRO = sec("intro");
const GROOVE = sec("groove");
const BREAK = sec("break");
const DROP2 = sec("drop2");
const FILL = sec("fill");
const HIT = beats.finalHit;
// Downbeats in the groove, for the montage's cuts.
const bars = beats.beats.filter((_, i) => i % 4 === 0);
const barAfter = (t: number, n: number) => bars.filter((b) => b >= t - 0.02)[n];

const BG = "#0d0f13";
const INK = "#e9ebf0";
const MUTED = "#a0a7b4";
const ATTN = "#ff7a45";
const MLP = "#2cc9ae";
const RESID = "#7b93ff";
const GOLD = "#ffe9a8";

const ASK = "I asked the computer what it was thinking, and it said";
const GEN = [" it", " was", " thinking", " about", " the", " computer", "."];

export const Trailer2 = () => {
  loadFonts();
  const montage = [
    { tag: "attention", color: ATTN, src: "arrows", frames: 113, speed: 1, cam: [{ cx: 900, cy: 560, z: 1.5 }, { cx: 860, cy: 560, z: 1.9 }] },
    { tag: "layers", color: RESID, src: "lens.png", cam: [{ cx: 520, cy: 790, z: 2.3 }, { cx: 1100, cy: 790, z: 2.5 }] },
    { tag: "neurons", color: MLP, src: "neurons", frames: 45, speed: 0.8, cam: [{ cx: 560, cy: 760, z: 1.6 }, { cx: 600, cy: 740, z: 2.0 }], bars: 1 },
    { tag: "steer", color: "#a080ff", src: "steer", frames: 83, speed: 0.75, cam: [{ cx: 900, cy: 600, z: 1.4 }, { cx: 1210, cy: 590, z: 2.1 }] },
    { tag: "break it", color: ATTN, src: "unplug2.png", cam: [{ cx: 1230, cy: 640, z: 1.9 }, { cx: 1230, cy: 720, z: 2.8 }] },
  ];
  // Lay the montage on the groove's bars, after the climb.
  // The climb runs 10 s; the tour starts on the nearest bar.
  const montageStart = bars.reduce((best, b) => (Math.abs(b - (GROOVE.start + 10)) < Math.abs(best - (GROOVE.start + 10)) ? b : best), bars[0]);
  let t = montageStart;
  const cuts = montage.map((m) => {
    const from = t;
    const to = m === montage[montage.length - 1] ? BREAK.start : barAfter(t + 0.1, (m.bars ?? 2) - 1) ?? BREAK.start;
    t = to;
    return { ...m, from, to };
  });

  return (
    <AbsoluteFill style={{ background: BG, fontFamily: SANS }}>
      {/* 1. One weight, then the pull-back. */}
      <Sequence from={0} durationInFrames={f(INTRO.end)} layout="none">
        <Frames dir="footage2/pullback" count={361} />
        <ColdOpen />
        <Counter />
      </Sequence>

      {/* 2. The drop: the forward pass climbs the tower. */}
      <Sequence from={f(GROOVE.start)} durationInFrames={f(montageStart - GROOVE.start)} layout="none">
        <Frames dir="footage2/climb" count={300} />
        <Tag text="watch it think" color={GOLD} at={0.4} until={8.2} big />
        <Sparkles at={8.45} x={860} y={92} />
      </Sequence>
      <Flash at={GROOVE.start} />

      {/* 3. The tour. */}
      {cuts.map((c, i) => (
        <Sequence key={i} from={f(c.from)} durationInFrames={f(c.to) - f(c.from)} layout="none">
          <Cam src={c.src} frames={c.frames} speed={c.speed} cam={c.cam} length={c.to - c.from} tilt={i % 2 ? 0.8 : -0.8} />
          <Vignette />
          <Tag text={c.tag} color={c.color} at={0.08} until={c.to - c.from - 0.1} />
          {c.tag === "layers" && <Aside text="Bush… Bush… Washington." at={1.6} />}
          {c.tag === "break it" && <Aside text="Mary: 68% → 2%" at={1.0} />}
        </Sequence>
      ))}
      {cuts.map((c, i) => <Hop key={i} at={c.from} flip={i % 2 === 1} />)}

      {/* 4. The break: every number, labelled. */}
      <Sequence from={f(BREAK.start)} durationInFrames={f(BREAK.end - BREAK.start)} layout="none">
        <Frames dir="footage2/glide" count={119} />
        <Vignette strong />
        <BreakText />
      </Sequence>

      {/* 5. GPT-2 writes; the tower fires for every new token. */}
      <Sequence from={f(DROP2.start)} durationInFrames={f(HIT) - f(DROP2.start)} layout="none">
        <Sequence from={0} durationInFrames={f(FILL.start - DROP2.start)} layout="none">
          <Frames dir="footage2/generate" count={239} />
        </Sequence>
        <Sequence from={f(FILL.start - DROP2.start)} durationInFrames={f(HIT - FILL.start) + 1} layout="none">
          <Frames dir="footage2/spin" count={59} />
        </Sequence>
        <Writing />
      </Sequence>
      <Flash at={DROP2.start} soft />

      {/* 6. The end card. */}
      <Sequence from={f(HIT)} durationInFrames={END2 - f(HIT)} layout="none">
        <EndCard />
      </Sequence>
      <Flash at={HIT} />
    </AbsoluteFill>
  );
};

// A folder of rendered frames, played at 30 fps.
function Frames({ dir, count, start = 0 }: { dir: string; count: number; start?: number }) {
  const frame = useCurrentFrame();
  const i = Math.min(count - 1, start + frame);
  return <Img src={staticFile(`${dir}/${String(i).padStart(4, "0")}.jpg`)} style={{ position: "absolute", inset: 0, width: 1920, height: 1080, objectFit: "cover" }} />;
}

// A recording or screenshot of the app (1600 × 900 points), under a camera
// that eases between two framings, with a punch-in at the cut.
function Cam({ src, frames, speed = 1, cam, length, tilt }: { src: string; frames?: number; speed?: number; cam: { cx: number; cy: number; z: number }[]; length: number; tilt: number }) {
  const frame = useCurrentFrame();
  const k = interpolate(frame, [0, length * FPS2], [0, 1], { easing: Easing.bezier(0.3, 0, 0.2, 1), extrapolateRight: "clamp" });
  const [a, b] = cam;
  const z = (a.z + (b.z - a.z) * k) * (1 + 0.05 * Math.exp(-frame / 3));
  const cx = clamp(a.cx + (b.cx - a.cx) * k, 800 / z, 1600 - 800 / z);
  const cy = clamp(a.cy + (b.cy - a.cy) * k, 450 / z, 900 - 450 / z);
  const s = 1.2 * z;
  const file = frames ? `footage/${src}/${String(Math.min(frames - 1, Math.floor(frame * speed))).padStart(4, "0")}.jpg` : `footage/${src}`;
  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <Img src={staticFile(file)} style={{ position: "absolute", width: 1600 * s, height: 900 * s, left: 960 - cx * s, top: 540 - cy * s, transform: `rotate(${tilt * (1 - k)}deg)` }} />
    </AbsoluteFill>
  );
}

function Vignette({ strong = false }: { strong?: boolean }) {
  return <AbsoluteFill style={{ background: `radial-gradient(ellipse at center, rgba(0,0,0,0) ${strong ? 35 : 55}%, rgba(0,0,0,${strong ? 0.75 : 0.5}) 100%)` }} />;
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
  return <AbsoluteFill style={{ background: "white", opacity: interpolate(frame, [0, 9], [soft ? 0.35 : 0.8, 0], { extrapolateRight: "clamp" }) }} />;
}

// The cold open: the weight's value, big, and its 16 bits spinning like a
// slot machine until they lock in, two per beat.
const BITS = "0 01011 1100010010";
function ColdOpen() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const out = interpolate(t, [3.6, 4.4], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  if (out <= 0) return null;
  const inV = spring({ frame: frame - 6, fps, config: { damping: 12, stiffness: 120 } });
  const beatsIn = beats.beats.filter((b) => b < 4);
  let bitIndex = 0;
  const chars = BITS.split("").map((ch, i) => {
    if (ch === " ") return <span key={i}> </span>;
    const which = bitIndex++;
    const lockAt = beatsIn[Math.min(beatsIn.length - 1, 1 + Math.floor(which / 2))] ?? 3.5;
    const locked = t >= lockAt;
    const spin = Math.floor(t * 30 + which * 7) % 2;
    const justLocked = locked && t - lockAt < 0.15;
    return <span key={i} style={{ color: locked ? (justLocked ? GOLD : INK) : "#4a505c", transform: justLocked ? "translateY(-4px)" : undefined, display: "inline-block" }}>{locked ? ch : spin}</span>;
  });
  const tag = spring({ frame: frame - f(1.5), fps, config: { damping: 10, stiffness: 160 } });
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: out }}>
      <div style={{ font: `600 210px ${MONO}`, color: INK, letterSpacing: "-0.02em", transform: `scale(${0.85 + 0.15 * inV})`, opacity: inV, textShadow: "0 10px 60px rgba(0,0,0,0.6)" }}>0.110</div>
      <div style={{ marginTop: 10, font: `500 44px ${MONO}`, letterSpacing: "0.12em" }}>{chars}</div>
      <div style={{ marginTop: 40, font: `600 40px ${SANS}`, color: GOLD, opacity: tag, transform: `translateY(${(1 - tag) * 20}px) rotate(${(1 - tag) * -6}deg)` }}>one weight.</div>
    </AbsoluteFill>
  );
}

// As the camera pulls back, a counter races to GPT-2's exact weight count.
const TOTAL = 124439808;
function Counter() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const start = 4.3;
  if (t < start) return null;
  const end = INTRO.end - 1.1;
  const k = Math.min(1, (t - start) / (end - start));
  // Fast at the end, like the zoom.
  const value = k >= 1 ? TOTAL : Math.round(Math.exp(Math.log(1) + (Math.log(TOTAL) - Math.log(1)) * k ** 1.3));
  const settle = spring({ frame: frame - f(end), fps, config: { damping: 7, stiffness: 200 } });
  const pop = k >= 1 ? 1 + 0.12 * (1 - settle) * Math.sin(settle * 10) : 1;
  const inV = interpolate(t, [start, start + 0.3], [0, 1], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 110, opacity: inV }}>
      <div style={{ font: `600 120px ${MONO}`, color: k >= 1 ? GOLD : INK, transform: `scale(${pop})`, textShadow: "0 8px 40px rgba(0,0,0,0.7)", fontVariantNumeric: "tabular-nums" }}>{value.toLocaleString("en-US")}</div>
      <div style={{ font: `500 34px ${SANS}`, color: MUTED, marginTop: 4 }}>{k >= 1 ? "weights. every one of them, right here." : "weights"}</div>
    </AbsoluteFill>
  );
}

// A small pill that springs in, a little tilted, and leaves on time.
function Tag({ text, color, at, until, big = false }: { text: string; color: string; at: number; until: number; big?: boolean }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - f(at), fps, config: { damping: 9, stiffness: 180 } });
  const out = interpolate(frame, [f(until) - 5, f(until)], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: big ? "flex-end" : "flex-start", alignItems: big ? "center" : "flex-start", padding: big ? "0 0 90px" : "70px 80px" }}>
      <div style={{
        padding: big ? "16px 34px" : "10px 26px", borderRadius: 999, background: "rgba(13,15,19,0.88)", border: `2px solid ${color}`, color,
        font: `600 ${big ? 54 : 40}px ${SANS}`, letterSpacing: "-0.01em", opacity: Math.min(1, s * 1.3) * out,
        transform: `scale(${0.6 + 0.4 * s}) rotate(${(1 - s) * -10 + (big ? 0 : -2)}deg)`, boxShadow: `0 0 40px ${color}55`,
      }}>{text}</div>
    </AbsoluteFill>
  );
}

// A little burst of gold when the answer pops out of the roof.
function Sparkles({ at, x, y }: { at: number; x: number; y: number }) {
  const frame = useCurrentFrame() - f(at);
  if (frame < 0 || frame > 26) return null;
  const k = frame / 26;
  return (
    <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
      {Array.from({ length: 14 }, (_, i) => {
        const a = (i / 14) * Math.PI * 2 + 0.3;
        const r = 30 + 150 * (1 - (1 - k) ** 3) * (0.7 + 0.3 * ((i * 7) % 5) / 4);
        const size = (i % 3 === 0 ? 7 : 4) * (1 - k);
        return <circle key={i} cx={x + Math.cos(a) * r} cy={y + Math.sin(a) * r * 0.8 + 40 * k * k} r={size} fill={i % 4 === 0 ? ATTN : GOLD} opacity={1 - k * 0.6} />;
      })}
    </svg>
  );
}

// A small aside under the tag.
function Aside({ text, at }: { text: string; at: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - f(at), fps, config: { damping: 12, stiffness: 160 } });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "flex-end", padding: "0 90px 80px" }}>
      <div style={{ font: `600 46px ${MONO}`, color: INK, opacity: s, transform: `translateY(${(1 - s) * 30}px)`, padding: "10px 22px", borderRadius: 14, background: "rgba(13,15,19,0.85)" }}>{text}</div>
    </AbsoluteFill>
  );
}

// The site's hop arc, swooping across the screen to carry in the next shot.
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
        <path d="M-100 1000 C 400 -250, 1500 -250, 2020 1000" pathLength={1} fill="none" stroke={ATTN} strokeWidth={34} strokeLinecap="round"
          strokeDasharray={`${head - tail} 2`} strokeDashoffset={-tail} opacity={0.95} />
      </svg>
    </AbsoluteFill>
  );
}

function BreakText() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = spring({ frame: frame - 6, fps, config: { damping: 14, stiffness: 120 } });
  const b = spring({ frame: frame - f(1.9), fps, config: { damping: 8, stiffness: 160 } });
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
      <div style={{ font: `600 120px ${SANS}`, color: INK, letterSpacing: "-0.03em", opacity: a, transform: `translateY(${(1 - a) * 30}px)`, textShadow: "0 10px 60px rgba(0,0,0,0.9)" }}>every number.</div>
      <div style={{ marginTop: 14, font: `500 44px ${SANS}`, color: GOLD, opacity: b, transform: `rotate(${-4 + (1 - b) * 12}deg) scale(${0.7 + 0.3 * b})`, textShadow: "0 6px 30px rgba(0,0,0,0.9)" }}>(yes, really.)</div>
    </AbsoluteFill>
  );
}

// The sentence writes itself: each word lands when the tower predicts it.
function Writing() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const per = (DROP2.end - DROP2.start) / GEN.length;
  const fade = interpolate(t, [HIT - DROP2.start - 0.25, HIT - DROP2.start], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", padding: "0 120px 70px", opacity: fade }}>
      <div style={{ maxWidth: 1500, padding: "22px 34px", borderRadius: 20, background: "rgba(13,15,19,0.86)", border: "1px solid rgba(255,255,255,0.1)", font: `500 46px ${SANS}`, lineHeight: 1.3, textAlign: "center" }}>
        <span style={{ color: MUTED }}>{ASK}</span>
        {GEN.map((w, k) => {
          const landed = t - (k * per + per * 0.78);
          if (landed < 0) return null;
          const s = spring({ frame: f(landed), fps, config: { damping: 8, stiffness: 220 } });
          const last = k === GEN.length - 1;
          return <span key={k} style={{ display: "inline-block", whiteSpace: "pre", color: k === GEN.length - 2 || last ? GOLD : INK, transform: `translateY(${(1 - s) * -28}px) scale(${last ? 0.6 + 0.4 * s : 1})`, opacity: Math.min(1, s * 1.5) }}>{w}</span>;
        })}
      </div>
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
  const typed = Math.max(0, Math.min(URL_TEXT.length, Math.floor((frame - 34) / 1.6)));
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
      <div style={{ marginTop: 34, fontSize: 58, fontWeight: 600, color: INK, letterSpacing: "-0.02em", opacity: rise(18), transform: `translateY(${(1 - rise(18)) * 26}px)` }}>Look inside a real language model.</div>
      <div style={{ marginTop: 28, padding: "14px 30px", borderRadius: 16, background: "rgba(255,122,69,0.1)", border: `2px solid ${ATTN}`, font: `600 54px ${MONO}`, color: ATTN, opacity: rise(30), minWidth: 900, textAlign: "center" }}>
        {URL_TEXT.slice(0, typed)}<span style={{ opacity: cursor && typed < URL_TEXT.length + 1 ? 1 : 0 }}>▍</span>
      </div>
      <div style={{ marginTop: 22, fontSize: 28, color: MUTED, opacity: rise(60) }}>free · runs on your GPU · nothing leaves your browser</div>
    </AbsoluteFill>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, v));
}
