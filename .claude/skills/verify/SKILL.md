---
name: verify
description: How to run and drive Featurama to see a change working in the browser.
---

# Verifying Featurama

The app is a static page that runs GPT-2 on WebGPU, so the surface is Chrome.

1. Model files must exist in `public/model/gpt2` (`bun run model` if not).
2. Start the dev server: `bun run dev` (port 5190; rebuilds on change, refresh to see it). If `scripts/serve.ts` itself changed, restart it.
3. Drive with playwright-core and system Chrome (Playwright's own WebKit is killed by Santa on Shu's Mac):
   `chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"] })`. Headless works with WebGPU.
4. `node scripts/look.mjs <dir>` walks the main pages (Machine, Attention predict and reveal, Layers predict, reveal, typing, dark mode); Zoom pages need their own drive and saves full-page screenshots.
5. Engine changes: `bun run test:gpu` with the dev server running compares the WebGPU engine with TransformerLens fixtures (and the layer-7 sparse autoencoder with fixtures/sae.json) and prints timings.
6. The sparse autoencoder needs `public/model/sae-l7` (`tools/reference/.venv/bin/python -I tools/reference/sae.py`). Feature data in public/features is committed; regenerate it with tools/reference/features.py only on Apple silicon or a GPU.
7. Zoom data in public/neurons and public/zoom-data is committed (tools/reference/neurons.py and weights.py, Apple silicon or a GPU). `bun run fixtures` regenerates fixtures, including q/k/v, neurons and an MLP unplug.
8. Puzzles: every level in src/app/puzzles.ts must stay solvable. After changing one, solve it in the browser at /play/<id>.

Gotchas:
- A fresh browser context starts in the predict beat; the prompt box is disabled until you pick a layer and press "Run it".
- Hovering lens squares by locator is flaky; move the mouse to the cell's bounding box instead.
- Weights are cached in the Cache API, so a second load in the same context makes no weight requests.
