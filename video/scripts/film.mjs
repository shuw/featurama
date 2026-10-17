// Renders the trailer's tower shots frame by frame from the real app in film
// mode (/weights?film), into public/footage2/<shot>/NNNN.jpg at 30 fps.
// Needs the dev server: `bun run dev` in the repo.
//
//   node scripts/film.mjs              every shot
//   node scripts/film.mjs climb glide  just these

import { chromium } from "../../node_modules/playwright-core/index.mjs";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "public/footage2");
const base = process.env.APP ?? "http://localhost:5190/";
const only = process.argv.slice(2);
const FPS = 30;
const beats = JSON.parse(fs.readFileSync(path.join(root, "src/beats2.json"), "utf8"));
const sec = (label) => beats.sections.find((s) => s.label === label);

const IOI = "When Mary and John went to the store, John gave a drink to";
const ASK = "I asked the computer what it was thinking, and it said";
// GPT-2's own top choices after ASK (checked against TransformerLens).
const GEN = [" it", " was", " thinking", " about", " the", " computer", "."];

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
const TOWER = [0, 6.2 * FLOOR, 0];

const lerp = (a, b, k) => a + (b - a) * k;
const smooth = (x) => x * x * (3 - 2 * x);
const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);

async function shot(name, frames, frameFn) {
  if (only.length && !only.includes(name)) return;
  const dir = path.join(out, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const meta = [];
  for (let i = 0; i < frames; i++) {
    const o = await frameFn(i / FPS, i);
    const px = await film((x) => window.__film.shot(x), o);
    meta.push(px);
    await page.screenshot({ path: path.join(dir, `${String(i).padStart(4, "0")}.jpg`), type: "jpeg", quality: 90 });
    if (i % 60 === 0) process.stdout.write(`\r${name} ${i}/${frames}   `);
  }
  fs.writeFileSync(path.join(dir, "px.json"), JSON.stringify(meta));
  console.log(`\r${name}: ${frames} frames`);
}

// 1. One weight, then the pull-back to the whole tower.
const q9 = await film(() => window.__film.tile("9q"));
const cell = [q9.x + 139.5, q9.floor * FLOOR, q9.z + 286.5];
const intro = sec("intro");
await shot("pullback", Math.round((intro.end - intro.start) * FPS), (t) => {
  const hold = 4.0;
  if (t < hold) {
    // Hovering over the weight, drifting in a touch.
    return { view: { target: cell, dist: lerp(0.95, 0.8, t / hold), yaw: lerp(-0.04, 0.0, t / hold) }, pos: 14, done: false, labels: true, callouts: false };
  }
  const u = easeInOut(Math.min(1, (t - hold) / (intro.end - intro.start - hold - 0.25)));
  const logd = lerp(Math.log(0.8), Math.log(41000), u);
  const dist = Math.exp(logd);
  // Stay over the weight while close; drift to the tower's centre as we pull out.
  const w = smooth(Math.min(1, Math.max(0, (logd - Math.log(200)) / (Math.log(41000) - Math.log(200)))));
  return {
    view: { target: [lerp(cell[0], TOWER[0], w), lerp(cell[1], TOWER[1], w), lerp(cell[2], TOWER[2], w)], dist, yaw: lerp(0, 0.6, smooth(u)) },
    pos: 14, done: false, labels: true, callouts: false,
  };
});

// 2. The forward pass climbs; the answer appears at the roof.
const climbLen = 10.0;
await shot("climb", Math.round(climbLen * FPS), (t) => {
  const run = 8.4;
  const pos = t < run ? 14 * easeInOut(t / run) : 14;
  // The camera rises with the pulse, close enough for the callouts to read.
  const k = smooth(Math.min(1, t / climbLen));
  return { view: { target: [0, lerp(4.0, 9.4, k) * FLOOR, 0], dist: lerp(25000, 23500, k), yaw: lerp(0.32, 0.78, k) }, pos: t < 0.15 ? 0 : pos, done: t >= run, labels: true, callouts: true };
});

// 3. The break: a slow glide across one matrix, every weight labelled.
const brk = sec("break");
const fc5 = await film(() => window.__film.tile("5fc"));
await shot("glide", Math.round((brk.end - brk.start) * FPS), (t) => {
  const k = t / (brk.end - brk.start);
  return { view: { target: [fc5.x + 1200 + 26 * k, fc5.floor * FLOOR, fc5.z + 300 + 4 * k], dist: 13, yaw: lerp(-0.05, 0.05, k) }, pos: 14, done: false, labels: true, callouts: false };
});

// 4. Generation: one quick pass per new token.
const drop2 = sec("drop2");
const per = (drop2.end - drop2.start) / GEN.length;
if (!only.length || only.includes("generate")) {
  const dir = path.join(out, "generate");
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  let frame = 0;
  for (let k = 0; k < GEN.length; k++) {
    await film((t) => window.__film.prompt(t), ASK + GEN.slice(0, k).join(""));
    const n = Math.round((drop2.start + (k + 1) * per) * FPS) - Math.round((drop2.start + k * per) * FPS);
    for (let i = 0; i < n; i++, frame++) {
      const p = i / n;
      const pos = Math.min(14, (p / 0.78) * 14);
      const gt = frame / FPS;
      await film((x) => window.__film.shot(x), {
        view: { target: [0, 7.1 * FLOOR, 0], dist: 29500, yaw: 0.25 + 0.09 * gt },
        pos, done: p >= 0.78, labels: true, callouts: true,
      });
      await page.screenshot({ path: path.join(dir, `${String(frame).padStart(4, "0")}.jpg`), type: "jpeg", quality: 90 });
    }
    process.stdout.write(`\rgenerate ${k + 1}/${GEN.length}   `);
  }
  console.log(`\rgenerate: ${frame} frames`);
}

// 5. A quick spin of the lit tower for the fill.
const fill = sec("fill");
if (!only.length || only.includes("spin")) await film((t) => window.__film.prompt(t), ASK + GEN.join(""));
await shot("spin", Math.round((fill.end - fill.start) * FPS), (t) => {
  const k = t / (fill.end - fill.start);
  return { view: { target: [0, lerp(5.5, 6.8, k) * FLOOR, 0], dist: lerp(36000, 26000, k), yaw: 1.2 + 2.4 * easeInOut(k) }, pos: 14, done: true, labels: false, callouts: false };
});

console.log("errors:", JSON.stringify(errors));
await browser.close();
