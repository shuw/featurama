// What each head tends to do on ordinary text, measured offline by
// tools/reference/heads.py and turned into plain sentences here.

import atlas from "./head-atlas.json";

interface Stats {
  rest: number;
  self: number;
  prev: number;
  dist: number;
  punct: number;
  copying: number;
  drawn: [string, number][];
}

const HEADS = atlas.heads as unknown as Record<string, Stats>;
export const ATLAS_SOURCE = "Measured on 300 paragraphs of Wikipedia (WikiText-103).";

const pc = (x: number) => `${Math.round(x * 100)}%`;

// Up to three sentences, most telling first.
export function habits(id: string): string[] {
  const s = HEADS[id];
  if (!s) return [];
  const layer = Number(id.split(".")[0]);
  const out: string[] = [];
  if (s.rest >= 0.6) out.push(`Mostly idle: ${pc(s.rest)} of its attention rests on the start token, until something it's looking for turns up.`);
  if (s.prev >= 0.3) out.push(`Looks at the token just before (${pc(s.prev)} of the time).`);
  else if (s.self >= 0.25) out.push(`Looks at its own token (${pc(s.self)} of the time).`);
  else if (s.rest < 0.6 && s.dist <= 3) out.push(`Looks at nearby tokens, about ${s.dist.toFixed(1)} back on average.`);
  else if (s.dist >= 15) out.push(`Looks far back: about ${Math.round(s.dist)} tokens on average.`);
  // Whole words only: fragments like "amaru" say more about the text than the head.
  const drawn = s.drawn.map(([t]) => t).filter((t) => /^ \p{L}{2,}$/u.test(t)).map((t) => t.trim()).slice(0, 3);
  if (drawn.length >= 2) out.push(`Drawn to tokens like ${drawn.map((t) => `“${t}”`).join(", ")}.`);
  // How a head's output changes the next-token odds only means much late in the model.
  if (layer >= 6 && s.copying >= 0.8) out.push("Copier: what it looks at becomes more likely as the next token.");
  if (layer >= 6 && s.copying <= -0.5) out.push("Suppressor: what it looks at becomes less likely as the next token.");
  if (!out.length) out.push("Spreads its attention around without a strong habit we can name.");
  return out.slice(0, 3);
}
