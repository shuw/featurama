// Renders tools/og/card.html to public/og.png, the image link previews show.
import { chromium } from "playwright-core";
import { fileURLToPath } from "node:url";

const root = new URL("../../", import.meta.url);
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.goto(new URL("tools/og/card.html", root).href);
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: fileURLToPath(new URL("public/og.png", root)) });
await browser.close();
