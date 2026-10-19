// Films the real app for the trailer. Drives Featurama in Chrome (WebGPU,
// the real GPT-2) and saves screenshots and screen recordings, resampled to
// 30 fps, into public/footage. Needs the dev server: `bun run dev` in the repo.
//
//   node scripts/capture.mjs            every shot
//   node scripts/capture.mjs steer map  just these

import { chromium } from "../../node_modules/playwright-core/index.mjs";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "public/footage");
const base = process.env.APP ?? "http://localhost:5190/";
const only = process.argv.slice(2);
const FPS = 30;
const W = 1600;
const H = 900;

const IOI = "When Mary and John went to the store, John gave a drink to";
const GEORGE = "The first president of the United States was George";
const WALK = "I went for a walk in the";
const q = (p) => `p=${encodeURIComponent(p)}`;

const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"] });
const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2, colorScheme: "dark" });
// Dark theme, and a visible pointer, since recordings don't show the real one.
await context.addInitScript(() => {
  try {
    localStorage.setItem("featurama-theme", "dark");
  } catch {}
  addEventListener("DOMContentLoaded", () => {
    const dot = document.createElement("div");
    dot.id = "cursor";
    dot.style.cssText = "position:fixed;left:-40px;top:-40px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:rgba(255,255,255,.9);box-shadow:0 0 0 3px rgba(0,0,0,.35),0 4px 14px rgba(0,0,0,.4);pointer-events:none;z-index:99999;transition:transform .12s";
    document.body.appendChild(dot);
    addEventListener("mousemove", (e) => { dot.style.left = e.clientX + "px"; dot.style.top = e.clientY + "px"; }, true);
    addEventListener("mousedown", () => (dot.style.transform = "scale(.7)"), true);
    addEventListener("mouseup", () => (dot.style.transform = ""), true);
  });
});
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const cdp = await context.newCDPSession(page);

fs.mkdirSync(out, { recursive: true });

async function open(url, ready) {
  await page.goto(base + url);
  await page.waitForSelector(ready, { timeout: 180000 });
  await page.waitForTimeout(600);
}

async function shot(name, opts = {}) {
  await page.screenshot({ path: path.join(out, `${name}.png`), ...opts });
  console.log("shot", name);
}

// Records the page while `action` runs, then keeps `tail` seconds more.
async function clip(name, action, tail = 0.6) {
  const frames = [];
  const onFrame = async ({ data, metadata, sessionId }) => {
    frames.push({ t: metadata.timestamp, data });
    cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  };
  cdp.on("Page.screencastFrame", onFrame);
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 90, maxWidth: W * 2, maxHeight: H * 2, everyNthFrame: 1 });
  const t0 = Date.now() / 1000;
  await action();
  await page.waitForTimeout(tail * 1000);
  const t1 = Date.now() / 1000;
  await cdp.send("Page.stopScreencast");
  cdp.off("Page.screencastFrame", onFrame);
  // Frames arrive only when something changes; hold each until the next.
  const dir = path.join(out, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const n = Math.round((t1 - t0) * FPS);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / FPS;
    while (k + 1 < frames.length && frames[k + 1].t <= t) k++;
    if (!frames.length) break;
    fs.writeFileSync(path.join(dir, `${String(i).padStart(4, "0")}.jpg`), Buffer.from(frames[k].data, "base64"));
  }
  console.log("clip", name, `${n} frames (${frames.length} captured)`);
}

const glide = async (x, y, steps = 18) => page.mouse.move(x, y, { steps });
const center = async (sel) => {
  const b = await page.locator(sel).first().boundingBox();
  return [b.x + b.width / 2, b.y + b.height / 2];
};
const scrollTo = async (sel, offset = 90) => {
  await page.locator(sel).first().evaluate((el, o) => scrollTo({ top: el.getBoundingClientRect().top + scrollY - o }), offset);
  await page.waitForTimeout(400);
};

// Scroll down by `by` pixels over about half a second, as a person would.
const smoothScroll = async (by) => {
  for (let i = 0; i < 16; i++) {
    await page.mouse.wheel(0, by / 16);
    await page.waitForTimeout(30);
  }
};

const SHOTS = {
  // A prompt typed into the Machine: tokens appear, then GPT-2's guess.
  async type() {
    await open(`machine?${q("x")}`, ".table .head");
    const box = page.locator("textarea").first();
    await box.click();
    await page.keyboard.press("Meta+A");
    await page.keyboard.press("Backspace");
    await page.waitForTimeout(400);
    await clip("type", async () => {
      await page.keyboard.type("The first president of the United States was", { delay: 55 });
      await page.waitForTimeout(1600);
    });
  },

  // The Machine's grid, then the same grid shaded by votes.
  async grid() {
    await open(`machine?${q(IOI)}`, ".table .head");
    await shot("grid");
    await clip("grid", async () => {
      const [x, y] = await center(".seg button:nth-child(2)");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForTimeout(900);
      for (const id of ["9.9", "10.7", "9.6"]) {
        const [hx, hy] = await center(`[aria-label^="Head ${id}:"], [aria-label^="Head ${id},"], [aria-label="Head ${id}"]`);
        await glide(hx, hy, 14);
        await page.waitForTimeout(500);
      }
    });
  },

  // Attention arrows, pointing at tokens, then inside head 9.9.
  async head() {
    await open(`head/9.9?${q(IOI)}`, ".qk-grid");
    await clip("arrows", async () => {
      const toks = page.locator(".arc-token");
      const n = await toks.count();
      for (const i of [n - 4, n - 6, n - 9, n - 1]) {
        const b = await toks.nth(i).boundingBox();
        await glide(b.x + b.width / 2, b.y + b.height / 2, 12);
        await page.waitForTimeout(450);
      }
      const b = await toks.nth(n - 1).boundingBox();
      await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
      await page.waitForTimeout(600);
    });
    await shot("arrows");
    await scrollTo(".deeper-head", 70);
    await shot("ask");
    await scrollTo(".one-score", 70);
    await shot("score");
    await scrollTo(".copy-table", 160);
    await shot("copy");
  },

  // The logit lens on George: Bush for most of the way, then Washington.
  async lens() {
    await open(`layers?${q(GEORGE)}`, ".signal");
    await page.waitForTimeout(1500);
    await shot("lens");
  },

  // Every neuron in MLP 7, then one neuron's card.
  async neurons() {
    await open(`mlp/7?${q(IOI)}`, ".neuron-grid");
    await page.waitForSelector(".features-side .example-text", { timeout: 60000 });
    await clip("neurons", async () => {
      const b = await page.locator(".neuron-grid").boundingBox();
      for (let i = 0; i <= 8; i++) await glide(b.x + b.width * (0.1 + 0.1 * i), b.y + b.height * (0.2 + 0.07 * i), 6);
    });
    await shot("neurons");
  },

  // A feature's card.
  async features() {
    await open(`features?f=24120&${q(WALK)}`, ".features-side .head-card");
    await page.waitForSelector(".features-side .example-text", { timeout: 60000 });
    await shot("features");
  },

  // Steering: the same walk, written with and without the planets feature.
  async steer() {
    await open(`lab/steer?f=24120&${q(WALK)}`, ".steer-row");
    await page.locator('input[type="range"]').evaluate((el) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, "60");
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await clip("steer", async () => {
      const [x, y] = await center("button.primary");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForSelector("button:has-text('Write with and without it'):not([disabled])", { timeout: 60000 });
    }, 1.2);
    await shot("steer");
  },

  // Unplug the name movers, then their backups.
  async unplug() {
    await open(`lab/unplug?${q(IOI)}&off=9.6,9.9,10.0`, ".effect-list");
    await page.waitForTimeout(800);
    await shot("unplug");
    await clip("unplug", async () => {
      const [x, y] = await center("button:has-text('backup name movers')");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForTimeout(1600);
    });
    await shot("unplug2");
  },

  // Trace the answer back: 156 runs.
  async trace() {
    await open(`lab/trace?${q(IOI)}`, ".vote-list");
    await page.locator(".vote-pick select").nth(1).selectOption("3");
    await page.waitForTimeout(600);
    await scrollTo(".vote-pick", 80);
    await shot("votes");
    await scrollTo("button:has-text('Unplug each part')", 300);
    await clip("trace", async () => {
      const [x, y] = await center("button:has-text('Unplug each part')");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForSelector(".trace-row", { timeout: 60000 });
      await page.waitForTimeout(300);
      await smoothScroll(520);
      // Follow the first part from a layer after 0, so there is something to trace.
      const names = await page.locator(".trace-row .vote-name").allTextContents();
      const pick = names.findIndex((n) => !/^Head 0\.|^MLP 0$/.test(n));
      const row = page.locator(".trace-row").nth(Math.max(0, pick));
      const rb = await row.boundingBox();
      await glide(rb.x + 60, rb.y + rb.height / 2);
      await page.mouse.click(rb.x + 60, rb.y + rb.height / 2);
      await page.waitForSelector(".dep-row, .trace-end", { timeout: 60000 });
      await page.waitForTimeout(200);
      await smoothScroll(420);
    }, 1.0);
    await shot("trace");
  },

  // Fly into the token map along the lens path.
  async map() {
    await open(`layers/map?${q(GEORGE)}`, ".map-path");
    await page.waitForTimeout(1200);
    await scrollTo(".map-canvas", 120);
    await shot("map");
    const b = await page.locator(".map-canvas").boundingBox();
    await clip("map", async () => {
      await glide(b.x + b.width * 0.5, b.y + b.height * 0.5);
      for (let i = 0; i < 26; i++) {
        await page.mouse.wheel(0, -60);
        await page.waitForTimeout(45);
      }
      await page.waitForTimeout(500);
    });
    await shot("map2");
  },

  // Solve a puzzle, then the puzzle list.
  async play() {
    await open("play/john", ".outcome");
    await clip("play", async () => {
      for (const id of ["0.10", "1.10"]) {
        const [x, y] = await center(`[data-head="${id}"]`);
        await glide(x, y);
        await page.mouse.click(x, y);
        await page.waitForTimeout(700);
      }
      await page.waitForSelector(".outcome-met", { timeout: 30000 });
    }, 1.2);
    await shot("play");
    await open("play", ".puzzle-grid");
    await shot("hub");
  },
};

// Load the model once before filming.
await open("machine", ".table .head");
for (const [name, run] of Object.entries(SHOTS)) {
  if (only.length && !only.includes(name)) continue;
  await run();
}
console.log("errors:", JSON.stringify(errors));
await browser.close();
