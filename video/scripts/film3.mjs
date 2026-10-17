// Renders trailer 3's tower shots frame by frame from the real app in film
// mode (/weights?film), into public/footage3/<shot>/NNNN.jpg at 30 fps.
// Needs the dev server: `bun run dev` in the repo.
//
//   node scripts/film3.mjs              every shot
//   node scripts/film3.mjs opening      just these

import { chromium } from "../../node_modules/playwright-core/index.mjs";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "public/footage3");
const base = process.env.APP ?? "http://localhost:5190/";
const only = process.argv.slice(2);
const FPS = 30;
const beats = JSON.parse(fs.readFileSync(path.join(root, "src/beats3.json"), "utf8"));
const bar = (i) => beats.bars[i];
const BAR = 2.0;

const IOI = "When Mary and John went to the store, John gave a drink to";

const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"] });
const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1.5, colorScheme: "dark" });
await context.addInitScript(() => {
  try {
    localStorage.setItem("featurama-theme", "dark");
  } catch {}
});
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));

await page.goto(`${base}weights?film&p=${encodeURIComponent(IOI)}`);
await page.waitForFunction(() => window.__film, null, { timeout: 180000 });
const film = (fn, arg) => page.evaluate(fn, arg);
await film((t) => window.__film.prompt(t), IOI);
const FLOOR = await film(() => window.__film.floorY);

const lerp = (a, b, k) => a + (b - a) * k;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (x) => x * x * (3 - 2 * x);
const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const easeOut = (x) => 1 - (1 - x) ** 4;
const mix = (a, b, k) => a.map((x, i) => lerp(x, b[i], k));

async function shot(name, frames, frameFn) {
  if (only.length && !only.includes(name)) return;
  const dir = path.join(out, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  for (let i = 0; i < frames; i++) {
    await film((x) => window.__film.shot(x), await frameFn(i / FPS, i));
    await page.screenshot({ path: path.join(dir, `${String(i).padStart(4, "0")}.jpg`), type: "jpeg", quality: 90 });
    if (i % 60 === 0) process.stdout.write(`\r${name} ${i}/${frames}   `);
  }
  console.log(`\r${name}: ${frames} frames`);
}

const q9 = await film(() => window.__film.tile("9q"));
const F = q9.floor;
const cell = [q9.x + 139.5, F * FLOOR, q9.z + 286.5];
const matrix = [q9.x + q9.w / 2, F * FLOOR, q9.z + q9.d / 2];
const layer = [0, F * FLOOR, 0];
// Where the app's tower panel starts, so the film can shrink into it.
const HOME = { target: [0, 4.6 * FLOOR, 0], dist: 41000, yaw: 0.45 };

// 1. Powers of ten: one number lights up as the pass reaches it, then one
// steady pull-back (constant speed in log distance, eased at both ends) out
// through a row, a matrix and a layer to the whole model. The distance per
// frame is saved for the trailer's counter (src/opening3.json).
const Z0 = bar(1) + 0.75;
const Z1 = bar(5) + 0.9;
const LEVELS = [
  { dist: 0.8, target: cell },
  { dist: 70, target: cell },
  { dist: 1500, target: mix(cell, matrix, 0.85) },
  { dist: 8500, target: layer },
  { dist: HOME.dist, target: HOME.target },
];
// Trapezoid speed: speed up over the first 18%, slow down over the last 25%.
const glide = (x, a = 0.18, b = 0.25) => {
  x = clamp01(x);
  const v = 1 / (1 - a / 2 - b / 2);
  if (x < a) return (v * x * x) / (2 * a);
  if (x <= 1 - b) return v * (a / 2 + (x - a));
  const y = 1 - x;
  return 1 - (v * y * y) / (2 * b);
};
const L0 = Math.log(LEVELS[0].dist);
const L1 = Math.log(HOME.dist);
const logDistAt = (t) => (t < Z0 ? lerp(Math.log(1.0), L0, clamp01(t / Z0)) : lerp(L0, L1, glide((t - Z0) / (Z1 - Z0))));
const targetAt = (logd) => {
  let target = LEVELS[0].target;
  for (let i = 1; i < LEVELS.length; i++) {
    const a = Math.log(LEVELS[i - 1].dist);
    const b = Math.log(LEVELS[i].dist);
    target = mix(target, LEVELS[i].target, smooth(clamp01((logd - a) / (b - a))));
  }
  return target;
};
const openingDists = [];
await shot("opening", Math.round(bar(6) * FPS), (t) => {
  const logd = logDistAt(t);
  openingDists.push(+Math.exp(logd).toFixed(3));
  // The pass: dark, then its front's glow sits on the number (pos ≈ F + 0.2)
  // for most of a beat, finishes the layer while we pull out, and climbs to
  // the roof once the whole tower is in view.
  const b1 = bar(1) - 0.1;
  let pos;
  if (t < b1 - 0.2) pos = F;
  else if (t < b1) pos = lerp(F, F + 0.175, smooth((t - b1 + 0.2) / 0.2));
  else if (t < b1 + 0.7) pos = lerp(F + 0.175, F + 0.225, (t - b1) / 0.7);
  else if (t < b1 + 1.0) pos = lerp(F + 0.225, F + 0.3, smooth((t - b1 - 0.7) / 0.3));
  else if (t < bar(5) - 0.6) pos = lerp(F + 0.3, F + 0.95, (t - b1 - 1.0) / (bar(5) - 0.6 - b1 - 1.0));
  else pos = lerp(F + 0.95, 14, smooth(clamp01((t - bar(5) + 0.6) / 1.9)));
  const yaw = lerp(-0.04, HOME.yaw, smooth((logd - L0) / (L1 - L0)));
  return { view: { target: targetAt(logd), dist: Math.exp(logd), yaw }, pos, done: t > bar(5) + 1.3, labels: true, callouts: false };
});
if (openingDists.length) fs.writeFileSync(path.join(root, "src/opening3.json"), JSON.stringify({ dist: openingDists }));

// 2. The investigation: from the whole tower down to head 9.9's slice of
// layer 9's queries (heads are 64 columns each).
const slice = [q9.x + 9 * 64 + 32, F * FLOOR, q9.z + q9.d / 2];
await shot("dive", Math.round(2 * BAR * FPS), (t) => {
  const k = easeInOut(clamp01(t / (1.7 * BAR)));
  const dist = Math.exp(lerp(Math.log(36000), Math.log(1250), k));
  return { view: { target: mix([0, 6.5 * FLOOR, 0], slice, smooth(clamp01(k * 1.4))), dist, yaw: lerp(0.9, 0.02, k) }, pos: 14, done: true, labels: true, callouts: false };
});

// 3. After the case closes, GPT-2 writes: one quick pass per new token, two
// beats each, then the camera whips round for the last hit.
const ASK = "I asked the computer what it was thinking, and it said";
// GPT-2's own top choices after ASK (checked against TransformerLens).
const GEN = [" it", " was", " thinking", " about", " the", " computer", "."];
const fill = beats.bars.find((b) => b > beats.resume + 5);
const per = (fill - beats.resume) / (GEN.length - 1);
const genLen = beats.finalHit - beats.resume;
if (!only.length || only.includes("generate")) {
  const dir = path.join(out, "generate");
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const total = Math.round(genLen * FPS);
  let k = -1;
  const spinFrom = (GEN.length - 1) * per + 0.8 * per;
  for (let i = 0; i < total; i++) {
    const t = i / FPS;
    const kk = Math.min(GEN.length - 1, Math.floor(t / per));
    if (kk !== k) {
      k = kk;
      await film((x) => window.__film.prompt(x), ASK + GEN.slice(0, k).join(""));
    }
    const p = (t - k * per) / per;
    const pos = Math.min(14, (p / 0.78) * 14);
    const whip = easeInOut(clamp01((t - spinFrom) / (genLen - spinFrom)));
    await film((x) => window.__film.shot(x), {
      view: { target: [0, lerp(7.3, 7.0, whip) * FLOOR, 0], dist: lerp(29000, 26000, whip), yaw: 0.25 + 0.09 * t + 2.4 * whip },
      pos, done: p >= 0.78, labels: whip < 0.3, callouts: whip < 0.3,
    });
    await page.screenshot({ path: path.join(dir, `${String(i).padStart(4, "0")}.jpg`), type: "jpeg", quality: 90 });
    if (i % 60 === 0) process.stdout.write(`\rgenerate ${i}/${total}   `);
  }
  console.log(`\rgenerate: ${total} frames`);
}

console.log("errors:", JSON.stringify(errors));
await browser.close();
