// The trailer: real footage of Featurama under a moving camera, captions on
// the beat, and the wordmark at the end. The music is added when rendering
// (scripts/render.mjs).

import { AbsoluteFill, Easing, Img, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { loadFonts, MONO, SANS } from "./fonts";
import { at, BEAT, CAPTIONS, END, END_CARD, HITS, SHOTS, type Caption, type Shot } from "./timeline";

const BG = "#0d0f13";
const INK = "#e9ebf0";
const MUTED = "#a0a7b4";
const ATTN = "#ff7a45";
const MLP = "#2cc9ae";
const RESID = "#7b93ff";
const ACCENTS = [ATTN, MLP, RESID];

export const Trailer = () => {
  loadFonts();
  return (
    <AbsoluteFill style={{ background: BG, fontFamily: SANS }}>
      {SHOTS.map((s, i) => (
        <Sequence key={i} from={at(s.from)} durationInFrames={at(s.to) - at(s.from)} layout="none">
          <ShotView shot={s} index={i} />
        </Sequence>
      ))}
      <Shade />
      {CAPTIONS.map((c, i) => (
        <Sequence key={i} from={at(c.from)} durationInFrames={at(c.to) - at(c.from)} layout="none">
          <CaptionView caption={c} index={i} />
        </Sequence>
      ))}
      <Sequence from={at(END_CARD)} durationInFrames={END - at(END_CARD)} layout="none">
        <EndCard />
      </Sequence>
      {HITS.map((b) => (
        <Sequence key={b} from={at(b)} durationInFrames={8} layout="none">
          <Flash />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

// One shot: a screenshot or a recording, framed by a camera that eases from
// its first framing to its second, with a small punch-in at the cut.
function ShotView({ shot, index }: { shot: Shot; index: number }) {
  const frame = useCurrentFrame();
  const length = at(shot.to) - at(shot.from);
  const t = interpolate(frame, [0, length], [0, 1], { easing: Easing.bezier(0.33, 0, 0.2, 1), extrapolateRight: "clamp" });
  const [a, b] = shot.cam;
  const z = a.z + (b.z - a.z) * t;
  const punch = 1 + 0.035 * Math.exp(-frame / 4);
  // Keep the frame inside the screenshot.
  const cx = clamp(a.cx + (b.cx - a.cx) * t, 800 / z, 1600 - 800 / z);
  const cy = clamp(a.cy + (b.cy - a.cy) * t, 450 / z, 900 - 450 / z);
  const k = 1.2 * z * punch; // output pixels per viewport point
  const src = shot.clip
    ? `footage/${shot.src}/${String(Math.min(shot.clip.frames - 1, Math.floor((shot.clip.start ?? 0) + frame * (shot.clip.speed ?? 1)))).padStart(4, "0")}.jpg`
    : `footage/${shot.src}`;
  // Alternate a slight tilt so consecutive shots don't feel static.
  const tilt = index % 2 ? 0.6 : -0.6;
  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <Img src={staticFile(src)} style={{
        position: "absolute", width: 1600 * k, height: 900 * k, left: 960 - cx * k, top: 540 - cy * k,
        filter: shot.blur ? `blur(${shot.blur}px)` : undefined, transform: `rotate(${tilt * (1 - t)}deg)`,
      }} />
      {shot.dim ? <AbsoluteFill style={{ background: BG, opacity: shot.dim }} /> : null}
    </AbsoluteFill>
  );
}

// Darkens the bottom and edges so captions read over any footage.
function Shade() {
  return (
    <AbsoluteFill style={{
      background: `linear-gradient(to top, rgba(13,15,19,0.92) 0%, rgba(13,15,19,0.55) 26%, rgba(13,15,19,0) 46%), radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,0.45) 100%)`,
    }} />
  );
}

function CaptionView({ caption, index }: { caption: Caption; index: number }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const step = (caption.step ?? 0.5) * BEAT;
  const length = at(caption.to) - at(caption.from);
  const out = interpolate(frame, [length - 5, length], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const center = caption.place === "center";
  const color = ACCENTS[index % ACCENTS.length];
  return (
    <AbsoluteFill style={{ justifyContent: center ? "center" : "flex-end", alignItems: center ? "center" : "flex-start", padding: center ? 0 : "0 120px 96px", opacity: out }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0 28px", justifyContent: center ? "center" : "flex-start", maxWidth: 1600 }}>
        {caption.words.map((w, i) => {
          const s = spring({ frame: frame - i * step, fps, config: { damping: 14, stiffness: 220, mass: 0.6 } });
          return (
            <span key={i} style={{
              display: "inline-block", fontSize: center ? 128 : 112, fontWeight: 650, letterSpacing: "-0.03em", lineHeight: 1.05,
              color: caption.accent?.includes(i) ? color : INK, opacity: Math.min(1, s * 1.4),
              transform: `translateY(${(1 - s) * 60}px) scale(${0.85 + 0.15 * s})`, textShadow: "0 6px 30px rgba(0,0,0,0.5)",
            }}>{w}</span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
}

function Flash() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ background: "white", opacity: interpolate(frame, [0, 8], [0.75, 0], { extrapolateRight: "clamp" }) }} />;
}

// The wordmark draws itself, as on the site: the last "a" looks back at the earlier ones.
const CH = 33.6;
const A = [2, 6, 8].map((i) => (i + 0.5) * CH);

function EndCard() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const draw = (delay: number, dur = 18) => interpolate(frame - delay, [0, dur], [1, 0], { easing: Easing.bezier(0.4, 0, 0.2, 1), extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const pop = spring({ frame: frame - 4, fps, config: { damping: 9, stiffness: 180 } });
  const rise = (delay: number) => spring({ frame: frame - delay, fps, config: { damping: 16, stiffness: 160 } });
  const fade = interpolate(frame, [END - at(END_CARD) - 10, END - at(END_CARD)], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ background: BG, justifyContent: "center", alignItems: "center", opacity: fade }}>
      <svg width={316 * 3} height={118 * 3} viewBox="-6 16 316 118" style={{ overflow: "visible" }}>
        <path d="M0 128H303" pathLength={1} fill="none" stroke={RESID} strokeWidth={2.5} strokeDasharray={1} strokeDashoffset={draw(0)} />
        <path d={`M${A[2]} 84 C ${A[2]} 50, ${A[1]} 50, ${A[1]} 84`} pathLength={1} fill="none" stroke={ATTN} strokeWidth={4} strokeLinecap="round" strokeDasharray={1} strokeDashoffset={draw(10)} />
        <path d={`M${A[2]} 84 C ${A[2]} 8, ${A[0]} 8, ${A[0]} 84`} pathLength={1} fill="none" stroke={ATTN} strokeWidth={3} strokeLinecap="round" opacity={0.75} strokeDasharray={1} strokeDashoffset={draw(16)} />
        <circle cx={A[2]} cy={84} r={5.5 * pop} fill={MLP} />
        <text x="0" y="118" style={{ font: `600 56px ${MONO}`, fill: INK }}>featurama</text>
      </svg>
      <div style={{ marginTop: 36, fontSize: 64, fontWeight: 600, color: INK, letterSpacing: "-0.02em", opacity: rise(22), transform: `translateY(${(1 - rise(22)) * 30}px)` }}>
        Look inside a real language model.
      </div>
      <div style={{ marginTop: 26, fontFamily: MONO, fontSize: 40, color: ATTN, opacity: rise(34), transform: `translateY(${(1 - rise(34)) * 30}px)` }}>
        shuw.github.io/featurama
      </div>
      <div style={{ marginTop: 22, fontSize: 28, color: MUTED, opacity: rise(44) }}>
        Free · runs on your GPU · nothing leaves your browser
      </div>
    </AbsoluteFill>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, v));
}
