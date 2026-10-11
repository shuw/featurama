// Screenshots of every page for review: bun run dev, then node scripts/look.mjs [dir].
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const out = process.argv[2] ?? "shots";
const base = process.env.URL ?? "http://localhost:5190/";
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"] });
const page = await browser.newPage({ viewport: { width: 1360, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const shot = (name) => page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
const type = async (text) => {
  const input = page.locator(".promptbox-input textarea");
  await input.focus();
  await input.evaluate((el) => el.setSelectionRange(el.value.length, el.value.length));
  await input.pressSequentially(text, { delay: 30 });
  await page.waitForTimeout(600);
};

await page.goto(base);
await page.waitForSelector(".table .head", { timeout: 120000 });
await page.waitForTimeout(600);
await shot("1-machine");

await page.goto(base + "#/attention");
await page.waitForSelector(".card-predict");
await page.waitForTimeout(400);
await shot("2-attention-predict");
await page.getByRole("button", { name: "Mary", exact: true }).click();
await page.getByRole("button", { name: "Show me" }).click();
await page.waitForTimeout(500);
await shot("3-attention");

await page.goto(base + "#/layers");
await page.waitForSelector(".card-predict .chips");
await page.waitForTimeout(400);
await shot("4-layers-predict");
await page.getByRole("button", { name: "10", exact: true }).click();
await page.getByRole("button", { name: "Run it" }).click();
await page.waitForTimeout(500);
await shot("5-layers");
await type(" Washington, and the second was John");
await shot("6-layers-typed");
await page.emulateMedia({ colorScheme: "dark" });
await shot("7-layers-dark");

console.log(JSON.stringify({ next: await page.locator(".promptbox-next").textContent(), errors }, null, 1));
await browser.close();
