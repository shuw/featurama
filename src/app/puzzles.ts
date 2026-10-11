// Puzzle levels: goals to reach with the machine's tools. Every level was
// solved with the reference model before it went in; `par` is the fewest
// heads that worked.

import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention } from "../engine/hooks.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { tokenize } from "./model.ts";

export interface Goal {
  token: string; // a single GPT-2 token, with its leading space
  is: "top" | "not-top" | "above" | "below";
  p?: number;
}

interface Base {
  id: string;
  title: string;
  story: string;
  prompt: string;
  hint: string;
  after: string; // shown once solved
}

export type Level = Base & (
  | { kind: "unplug"; goal: Goal; budget: number; layers?: [number, number]; par: number }
  | { kind: "steer"; goal: Goal }
  | { kind: "find"; target: "previous" | { token: string; nth: number }; min: number }
  | { kind: "lens"; answer: string }
);

const IOI = "When Mary and John went to the store, John gave a drink to";
const DAYS = "Monday, Tuesday, Wednesday,";
const REPEAT = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy";

export const LEVELS: Level[] = [
  {
    id: "movers", kind: "unplug", title: "Quiet the name movers", prompt: IOI,
    story: "GPT-2 is 68% sure the drink goes to Mary. Unplug heads until it's under 55% sure.",
    goal: { token: " Mary", is: "below", p: 0.55 }, budget: 3, par: 1,
    hint: "The Name movers chip outlines three famous heads. They'll get you there, but one head much earlier does it alone.",
    after: "The three name movers (9.9, 9.6, 10.0) only bring Mary down to about 50%: backup heads step in. Researchers call that self-repair. Head 0.10, in the very first layer, does more damage on its own.",
  },
  {
    id: "john", kind: "unplug", title: "Make it say John", prompt: IOI,
    story: "Make “John” GPT-2's top guess, unplugging at most three heads.",
    goal: { token: " John", is: "top" }, budget: 3, par: 2,
    hint: "GPT-2 picks Mary because John appeared twice. Heads that notice repeated tokens sit in the first layers.",
    after: "Without its early heads, GPT-2 can't tell that John was repeated, so it no longer knows which name to skip. Unplugging 0.10 and 1.10 is enough.",
  },
  {
    id: "late", kind: "unplug", title: "Late-layer surgery", prompt: IOI, layers: [7, 11],
    story: "Knock Mary off the top spot, but you may only unplug heads in layers 7 to 11.",
    goal: { token: " Mary", is: "not-top" }, budget: 4, par: 3,
    hint: "The Name inhibitors chip outlines heads that tell the movers whom to skip. Combine one with a late head.",
    after: "11.0, 8.3 and 8.6 together do it. 8.6 is a name inhibitor: without it, the movers lose track of which name to copy.",
  },
  {
    id: "week", kind: "unplug", title: "Lose track of the week", prompt: DAYS,
    story: "GPT-2 knows Thursday comes next (82%). Make it guess something else.",
    goal: { token: " Thursday", is: "not-top" }, budget: 3, par: 1,
    hint: "Days of the week need GPT-2 to see the repeated pattern. One early head carries a lot of that.",
    after: "Head 0.10 again: without it, GPT-2 guesses Wednesday. Later heads like the successor head 9.1 can't work without what the early layers build.",
  },
  {
    id: "lazy", kind: "unplug", title: "Break the repeat", prompt: REPEAT,
    story: "The sentence repeats, so GPT-2 is 99.7% sure “dog” comes next. Get it under 50%.",
    goal: { token: " dog", is: "below", p: 0.5 }, budget: 4, par: 3,
    hint: "Unplugging the famous induction heads barely dents it. Look for heads that feed them, early on.",
    after: "Even with all five well-known induction heads unplugged, GPT-2 still says dog 99.8%: copying repeats is spread over many heads. Early heads 0.10, 1.10 and 2.10 together bring it under half.",
  },
  {
    id: "walk", kind: "steer", title: "Walk the dog", prompt: "I went for a walk with my",
    story: "Steer GPT-2 until it gives “dog” at least a 25% chance (it starts at 8%).",
    goal: { token: " dog", is: "above", p: 0.25 },
    hint: "Search for a feature about dogs, then try a gentle strength. Too strong backfires.",
    after: "Feature #8243 fires on “dog”. At +20 it lifts “dog” to about 29%; at +60 it overshoots and “dog” drops again.",
  },
  {
    id: "rain", kind: "steer", title: "Make it rain", prompt: "Tomorrow the weather will bring lots of",
    story: "Rain is already GPT-2's top guess at 22%. Push it past 40%.",
    goal: { token: " rain", is: "above", p: 0.4 },
    hint: "Search “rain”. Somewhere between +15 and +45 works.",
    after: "Feature #7626 fires on “rain” and pushes toward showers and storms. Too much, and GPT-2 starts writing punctuation instead.",
  },
  {
    id: "earth", kind: "steer", title: "Back to Earth", prompt: "Last summer my family visited the planet",
    story: "GPT-2 guesses Venus first. Steer it until “Earth” gets at least 30%.",
    goal: { token: " Earth", is: "above", p: 0.3 },
    hint: "A feature about planets pushes toward Earth. It needs a firm push.",
    after: "Feature #24120 fires on “planet” and pushes toward Earth, orbiting and Venus. Around +60, Earth reaches a third.",
  },
  {
    id: "pirates", kind: "steer", title: "Pirates!", prompt: "The ship was attacked by",
    story: "Get “pirates” to at least 19% (it starts at 8%).",
    goal: { token: " pirates", is: "above", p: 0.19 },
    hint: "There's a pirate feature. The window is narrow: try around +20 to +30.",
    after: "Feature #5531 fires on “pirate” and “Pirates of the Caribbean”. It roughly doubles the odds, but “a” stays a strong rival.",
  },
  {
    id: "previous", kind: "find", title: "Find a look-back head", prompt: DAYS,
    story: "Find a head that puts at least 60% of its attention, from the last comma, on the token just before it.",
    target: "previous", min: 0.6,
    hint: "Try the Previous token chip.",
    after: "Previous-token heads are some of the simplest in GPT-2. Induction heads build on them.",
  },
  {
    id: "mover", kind: "find", title: "Find a name mover", prompt: IOI,
    story: "Find a head that, from the final “to”, puts at least 50% of its attention on “Mary”.",
    target: { token: " Mary", nth: 0 }, min: 0.5,
    hint: "Name movers live in layers 9 and 10.",
    after: "Head 9.9 puts 67% on Mary. It's one of the heads you quieted in the first puzzle.",
  },
  {
    id: "induction", kind: "find", title: "Find an induction head", prompt: REPEAT,
    story: "From the second “lazy”, find a head that puts at least 40% of its attention on the “dog” that followed the first “lazy”.",
    target: { token: " dog", nth: 0 }, min: 0.4,
    hint: "Induction heads look at whatever came after the last copy of the current token. Try the Induction chip.",
    after: "That's induction: “last time I saw ‘lazy’, ‘dog’ came next”. It's how GPT-2 copies repeated text.",
  },
  {
    id: "george", kind: "lens", title: "When does George become Washington?", prompt: "The first president of the United States was George",
    story: "Read GPT-2's best guess after each layer. After which layer does “Washington” first take the lead?",
    answer: " Washington",
    hint: "The Layers page shows this with the logit lens. Until late on, GPT-2 has a more recent George in mind.",
    after: "GPT-2 guesses Bush through most of its layers and only switches to Washington near the end. Knowing which George is meant takes most of the model.",
  },
];

export type Progress = Record<string, { solved: boolean; best?: number }>;
const KEY = "featurama-puzzles";
export function loadProgress(): Progress {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}
export function saveProgress(p: Progress) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {}
}

export function tokenOf(tok: Tokenizer, text: string) {
  const ids = tok.encode(text);
  if (ids.length !== 1) throw new Error(`“${text}” is not one token.`);
  return ids[0];
}

// The next-token outcome of a run, and whether it meets the goal.
export async function check(engine: GpuEngine, tok: Tokenizer, prompt: string, goal: Goal, interventions: Intervention[]) {
  const L = engine.config.n_layers;
  const tokens = tokenize(tok, prompt);
  const target = tokenOf(tok, goal.token);
  const trace = await engine.run({ tokens, interventions, lens: { positions: [tokens.length - 1], topK: 5, targets: [target] } });
  const K = trace.lens!.topK;
  const top = Array.from(trace.lens!.ids.subarray(L * K, L * K + 3), (id, i) => ({ id, p: Math.exp(trace.lens!.logprobs[L * K + i]) }));
  const p = Math.exp(trace.lens!.targetLogprobs[L]);
  const met =
    goal.is === "top" ? top[0].id === target
    : goal.is === "not-top" ? top[0].id !== target
    : goal.is === "above" ? p >= goal.p!
    : p < goal.p!;
  return { top, p, met };
}

// Share text in the style of a daily word game.
export function shareText(progress: Progress, url: string) {
  const row = LEVELS.map((l) => (progress[l.id]?.solved ? "🟩" : "⬜")).join("");
  const n = LEVELS.filter((l) => progress[l.id]?.solved).length;
  return `Featurama puzzles ${n}/${LEVELS.length}\n${row}\n${url}`;
}
