// The trailer, beat by beat. The music is 128 BPM (scripts/music.py), so
// every cut and caption is placed in beats; 4 beats make a bar.

export const FPS = 30;
export const BPM = 128;
export const BEAT = (60 / BPM) * FPS; // frames per beat, 14.0625
export const END = 60 * FPS;
export const at = (beat: number) => Math.round(beat * BEAT);

// A point in the app's 1600 × 900 viewport, and how far the camera zooms in.
export interface Cam {
  cx: number;
  cy: number;
  z: number;
}

export interface Shot {
  from: number; // beat
  to: number; // beat
  src: string; // a screenshot (name.png) or a recorded clip (its folder)
  clip?: { frames: number; start?: number; speed?: number };
  cam: [Cam, Cam];
  blur?: number;
  dim?: number;
}

export interface Caption {
  from: number; // beat
  to: number; // beat
  words: string[];
  step?: number; // beats between words (default 0.5)
  accent?: number[]; // which words are coloured
  place?: "low" | "center";
}

export const SHOTS: Shot[] = [
  // Cold open: the Machine, blurred and dark behind the question.
  { from: 0, to: 8, src: "grid.png", cam: [{ cx: 800, cy: 500, z: 1.6 }, { cx: 800, cy: 520, z: 1.9 }], blur: 14, dim: 0.75 },
  // A prompt typed into GPT-2.
  { from: 8, to: 16, src: "type", clip: { frames: 150, speed: 1.3 }, cam: [{ cx: 760, cy: 380, z: 1.15 }, { cx: 640, cy: 345, z: 1.65 }] },
  // The Machine: every head, then votes.
  { from: 16, to: 20, src: "grid.png", cam: [{ cx: 620, cy: 640, z: 1.6 }, { cx: 640, cy: 600, z: 1.3 }] },
  { from: 20, to: 24, src: "grid", clip: { frames: 120, speed: 1.6 }, cam: [{ cx: 640, cy: 600, z: 1.3 }, { cx: 620, cy: 640, z: 1.5 }] },
  // Attention arrows.
  { from: 24, to: 32, src: "arrows", clip: { frames: 113 }, cam: [{ cx: 900, cy: 560, z: 1.45 }, { cx: 880, cy: 560, z: 1.8 }] },
  // Drop: inside head 9.9, down to multiplications.
  { from: 32, to: 36, src: "ask.png", cam: [{ cx: 760, cy: 480, z: 1.25 }, { cx: 820, cy: 440, z: 1.7 }] },
  { from: 36, to: 40, src: "score.png", cam: [{ cx: 640, cy: 200, z: 2.4 }, { cx: 560, cy: 225, z: 2.9 }] },
  // The logit lens: Bush for most of the way, then Washington.
  { from: 40, to: 44, src: "lens.png", cam: [{ cx: 760, cy: 600, z: 1.3 }, { cx: 820, cy: 640, z: 1.6 }] },
  { from: 44, to: 48, src: "lens.png", cam: [{ cx: 560, cy: 790, z: 2.4 }, { cx: 1080, cy: 790, z: 2.4 }] },
  // Neurons.
  { from: 48, to: 52, src: "neurons", clip: { frames: 45, speed: 0.8 }, cam: [{ cx: 560, cy: 760, z: 1.5 }, { cx: 560, cy: 760, z: 1.8 }] },
  { from: 52, to: 56, src: "neurons.png", cam: [{ cx: 1210, cy: 700, z: 1.9 }, { cx: 1210, cy: 760, z: 2.2 }] },
  // Features.
  { from: 56, to: 64, src: "features.png", cam: [{ cx: 760, cy: 520, z: 1.2 }, { cx: 1180, cy: 560, z: 1.9 }] },
  // Steering: written with and without the planets feature.
  { from: 64, to: 72, src: "steer", clip: { frames: 83, speed: 0.72 }, cam: [{ cx: 760, cy: 600, z: 1.35 }, { cx: 1180, cy: 560, z: 1.9 }] },
  { from: 72, to: 80, src: "steer.png", cam: [{ cx: 1210, cy: 600, z: 2.2 }, { cx: 1210, cy: 620, z: 2.6 }] },
  // Break: unplug the name movers, and GPT-2 repairs itself.
  { from: 80, to: 88, src: "unplug.png", cam: [{ cx: 800, cy: 560, z: 1.25 }, { cx: 1230, cy: 600, z: 2.0 }] },
  // Drop 2: unplug the backups too, and it breaks.
  { from: 88, to: 92, src: "unplug", clip: { frames: 75, speed: 1.3 }, cam: [{ cx: 1230, cy: 640, z: 1.9 }, { cx: 1230, cy: 640, z: 1.9 }] },
  { from: 92, to: 96, src: "unplug2.png", cam: [{ cx: 1230, cy: 720, z: 2.5 }, { cx: 1230, cy: 720, z: 2.9 }] },
  // Trace it back: 156 runs.
  { from: 96, to: 104, src: "trace", clip: { frames: 279, speed: 2.45 }, cam: [{ cx: 760, cy: 480, z: 1.2 }, { cx: 760, cy: 460, z: 1.4 }] },
  // The token map.
  { from: 104, to: 108, src: "map", clip: { frames: 82, speed: 1.4 }, cam: [{ cx: 800, cy: 450, z: 1.05 }, { cx: 800, cy: 420, z: 1.2 }] },
  { from: 108, to: 112, src: "map2.png", cam: [{ cx: 700, cy: 380, z: 1.6 }, { cx: 820, cy: 400, z: 2.0 }] },
  // Puzzles.
  { from: 112, to: 116, src: "play", clip: { frames: 96, speed: 1.6 }, cam: [{ cx: 640, cy: 380, z: 1.5 }, { cx: 760, cy: 340, z: 1.7 }] },
  { from: 116, to: 120, src: "hub.png", cam: [{ cx: 800, cy: 440, z: 1.15 }, { cx: 640, cy: 420, z: 1.5 }] },
];

export const CAPTIONS: Caption[] = [
  { from: 1, to: 8, words: ["What", "is", "an", "AI", "actually", "thinking?"], step: 1, accent: [5], place: "center" },
  { from: 8.5, to: 12, words: ["A", "real", "language", "model."], accent: [2, 3] },
  { from: 12, to: 16, words: ["Running", "in", "your", "browser."], accent: [3] },
  { from: 16, to: 20, words: ["144", "heads.", "12", "layers."], step: 1, accent: [0, 2] },
  { from: 20, to: 24, words: ["Every", "one", "visible."], accent: [2] },
  { from: 24, to: 32, words: ["Watch", "it", "decide", "where", "to", "look."], accent: [2] },
  { from: 32, to: 40, words: ["Down", "to", "every", "multiplication."], step: 1, accent: [3] },
  { from: 40, to: 44, words: ["Watch", "it", "change", "its", "mind."], accent: [2, 3, 4] },
  { from: 44, to: 48, words: ["George", "Bush…", "Bush…", "Washington."], step: 1, accent: [3] },
  { from: 48, to: 56, words: ["3,072", "neurons", "a", "layer."], accent: [0] },
  { from: 56, to: 64, words: ["Concepts", "hiding", "in", "the", "numbers."], accent: [0] },
  { from: 64, to: 72, words: ["Turn", "one", "up…"], step: 1, accent: [2] },
  { from: 72, to: 80, words: ["…and", "it", "writes", "about", "planets."], accent: [4] },
  { from: 80, to: 88, words: ["Break", "it", "on", "purpose."], step: 1, accent: [0] },
  { from: 88, to: 92, words: ["Pull", "the", "backups", "too…"], accent: [2] },
  { from: 92, to: 96, words: ["Mary:", "68%", "→", "2%."], accent: [3] },
  { from: 96, to: 104, words: ["Trace", "every", "answer", "back."], accent: [0] },
  { from: 104, to: 112, words: ["50,257", "tokens.", "One", "map."], step: 1, accent: [0] },
  { from: 112, to: 120, words: ["13", "puzzles.", "Beat", "the", "machine."], accent: [2, 3, 4] },
];

// Beats where the drop hits: a white flash and a punch-in.
export const HITS = [32, 88, 120];
export const END_CARD = 120; // beat
