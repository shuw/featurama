// Runs the engine check page in Chrome and prints what it found.
// Start the dev server first (bun run dev).
import { chromium } from "playwright-core";

const url = (process.env.URL ?? "http://localhost:5190/") + "tests/gpu/";
const browser = await chromium.launch({
  channel: "chrome",
  headless: process.env.HEADED ? false : true,
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
page.on("console", (m) => m.type() === "error" && console.error("console:", m.text()));
const res = await page.goto(url);
if (!res?.ok()) {
  console.error(`${url} answered ${res?.status()}. Is the dev server running?`);
  process.exit(1);
}
await page.waitForFunction(() => window.results, null, { timeout: 300_000, polling: 250 });
console.log(await page.$eval("#log", (e) => e.textContent));
const results = await page.evaluate(() => window.results);
if (process.env.JSON) console.log(JSON.stringify(results, null, 1));
await browser.close();
process.exit(results.ok ? 0 : 1);
