// Films the real app for trailer 2, at 1920 × 1080: screen recordings in real
// time (resampled to 30 fps) and screenshots, into public/footage3. Needs the
// dev server: `bun run dev` in the repo.
//
//   node scripts/capture2.mjs             every shot
//   node scripts/capture2.mjs hook lab    just these

import { chromium } from "../../node_modules/playwright-core/index.mjs";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "public/footage3");
const base = process.env.APP ?? "http://localhost:5190/";
const only = process.argv.slice(2);
const FPS = 30;
const W = 1920;
const H = 1080;

const IOI = "When Mary and John went to the store, John gave a drink to";
const GEORGE = "The first president of the United States was George";
const WALK = "I went for a walk in the";
const q = (p) => `p=${encodeURIComponent(p)}`;

const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"] });
const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1.5, colorScheme: "dark" });
// Dark theme, and a visible pointer, since recordings don't show the real one.
await context.addInitScript(() => {
  try {
    localStorage.setItem("featurama-theme", "dark");
  } catch {}
  addEventListener("DOMContentLoaded", () => {
    const dot = document.createElement("div");
    dot.id = "cursor";
    dot.style.cssText = "position:fixed;left:-40px;top:-40px;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;background:rgba(255,255,255,.9);box-shadow:0 0 0 3px rgba(0,0,0,.35),0 4px 14px rgba(0,0,0,.4);pointer-events:none;z-index:99999;transition:transform .12s";
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
// Clicks and keystrokes are logged per clip (events.json), for the sound effects.
let marks = null;
const mark = (kind) => marks?.list.push({ kind, t: +(Date.now() / 1000 - marks.t0).toFixed(3) });
const click = page.mouse.click.bind(page.mouse);
page.mouse.click = async (...a) => (mark("click"), click(...a));
const type = page.keyboard.type.bind(page.keyboard);
page.keyboard.type = async (...a) => (mark("key"), type(...a));

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
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 90, maxWidth: W * 1.5, maxHeight: H * 1.5, everyNthFrame: 1 });
  const t0 = Date.now() / 1000;
  marks = { t0, list: [] };
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
  fs.writeFileSync(path.join(dir, "events.json"), JSON.stringify(marks.list));
  marks = null;
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


const PB = "Peanut butter and";
// A human typing rhythm: quicker inside words, a beat after spaces and commas.
async function typeLike(text) {
  for (const ch of text) {
    await page.keyboard.type(ch);
    await page.waitForTimeout(ch === " " ? 95 : ch === "," ? 220 : 45 + ((ch.charCodeAt(0) * 37) % 50));
  }
}
const headBtn = (id) => `[aria-label^="Head ${id},"], [aria-label="Head ${id}"]`;

const SHOTS = {
  // The hook: typing the prompt above the tower; it runs the new token.
  async hook() {
    await open(`weights?${q("x")}`, ".tower-canvas");
    await scrollTo(".promptbox, textarea", 16);
    const box = page.locator("textarea").first();
    await box.click();
    await page.keyboard.press("Meta+A");
    await page.keyboard.press("Backspace");
    await page.mouse.move(1720, 230);
    await page.waitForTimeout(600);
    await clip("hook", async () => {
      await typeLike(IOI);
      await page.waitForTimeout(2600);
    }, 0.2);
  },

  // Head 9.9 looks at Mary.
  async attention() {
    await open(`head/9.9?${q(IOI)}`, ".arc-token");
    await scrollTo(".attn-layout, .stage", 120);
    await clip("attention", async () => {
      const toks = page.locator(".arc-token");
      const n = await toks.count();
      for (const i of [n - 5, n - 3, n - 1]) {
        const b = await toks.nth(i).boundingBox();
        await glide(b.x + b.width / 2, b.y + b.height / 2, 14);
        await page.waitForTimeout(420);
      }
      const b = await toks.nth(n - 1).boundingBox();
      await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
      await page.waitForTimeout(1400);
    });
  },

  // The Machine shaded by votes: 9.9 votes for Mary.
  async votes() {
    await open(`?${q(IOI)}`, ".table .head");
    await scrollTo(".machine-tools", 110);
    await clip("votes", async () => {
      const [x, y] = await center(".seg button:nth-child(2)");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForTimeout(700);
      const [hx, hy] = await center(headBtn("9.9"));
      await glide(hx, hy, 16);
      await page.mouse.click(hx, hy);
      await page.waitForTimeout(1500);
    });
  },

  // Unplug the name movers one by one; then, later, the backups.
  async lab() {
    await open(`lab/unplug?${q(IOI)}`, ".table .head");
    await page.waitForTimeout(800);
    await scrollTo(".machine-body", 70);
    await clip("unplug", async () => {
      for (const id of ["9.9", "9.6", "10.0"]) {
        const [x, y] = await center(headBtn(id));
        await glide(x, y, 14);
        await page.mouse.click(x, y);
        await page.waitForTimeout(900);
      }
      await page.waitForTimeout(1200);
    });
    await clip("backups", async () => {
      const [x, y] = await center("button:has-text('backup name movers')");
      await glide(x - 140, y - 60, 10);
      await glide(x, y, 30);
      await page.waitForTimeout(2200);
      await page.mouse.click(x, y);
      await page.waitForTimeout(2600);
    });
  },

  // Tour: tokens, the lens, neurons, a feature, steering, the map, a puzzle.
  async tokens() {
    await open(`?${q("Welcome to")}`, ".table .head");
    const box = page.locator("textarea").first();
    await box.click();
    await page.keyboard.press("End");
    await page.waitForTimeout(300);
    await clip("tokens", async () => {
      await typeLike(" featurama, a museum of");
      await page.waitForTimeout(900);
    });
  },
  async lens() {
    await open(`layers?${q(PB)}`, ".signal");
    await page.waitForTimeout(1500);
    await scrollTo(".signal", 260);
    await shot("lens");
  },
  async neurons() {
    await open(`mlp/7?${q(IOI)}`, ".neuron-grid");
    await page.waitForSelector(".features-side .example-text", { timeout: 60000 });
    await scrollTo(".neuron-grid", 200);
    await clip("neurons", async () => {
      const b = await page.locator(".neuron-grid").boundingBox();
      for (let i = 0; i <= 10; i++) await glide(b.x + b.width * (0.08 + 0.08 * i), b.y + b.height * (0.2 + 0.06 * i), 5);
    });
  },
  async features() {
    await open(`features?f=7626&${q("I went for a walk in the")}`, ".features-side .head-card");
    await page.waitForSelector(".features-side .example-text", { timeout: 60000 });
    await scrollTo(".features", 120);
    await shot("features");
  },
  async steer() {
    await open(`lab/steer?f=24120&${q("I went for a walk in the")}`, ".steer-row");
    await page.locator('input[type="range"]').evaluate((el) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, "60");
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await scrollTo(".features", 120);
    await clip("steer", async () => {
      const [x, y] = await center("button.primary");
      await glide(x, y);
      await page.mouse.click(x, y);
      await page.waitForSelector("button:has-text('Write with and without it'):not([disabled])", { timeout: 60000 });
    }, 0.8);
  },
  async map() {
    await open(`layers/map?${q(PB)}`, ".map-path");
    await page.waitForTimeout(1200);
    await scrollTo(".map-canvas", 60);
    const b = await page.locator(".map-canvas").boundingBox();
    await clip("map", async () => {
      await glide(b.x + b.width * 0.5, b.y + b.height * 0.5);
      for (let i = 0; i < 30; i++) {
        await page.mouse.wheel(0, -60);
        await page.waitForTimeout(40);
      }
      await page.waitForTimeout(400);
    });
  },
  async puzzle() {
    await open("play/john", ".outcome");
    await clip("puzzle", async () => {
      for (const id of ["0.10", "1.10"]) {
        const [x, y] = await center(`[data-head="${id}"]`);
        await glide(x, y);
        await page.mouse.click(x, y);
        await page.waitForTimeout(650);
      }
      await page.waitForSelector(".outcome-met", { timeout: 30000 });
    }, 1.0);
  },
};

// Load the model once before filming.
await open("", ".table .head");
for (const [name, run] of Object.entries(SHOTS)) {
  if (only.length && !only.includes(name)) continue;
  await run();
}
console.log("errors:", JSON.stringify(errors));
await browser.close();
