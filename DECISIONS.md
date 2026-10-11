# Decisions

Significant technical and design choices, newest last. Each says what we chose and why.

## Engine: a hand-written WebGPU forward pass (2026-10-10)

GPT-2 small runs on WebGPU kernels written for Featurama (`src/engine/gpu/`), not ONNX Runtime Web or transformers.js. Every activation lives in a buffer we own, so any of them can be read back or overwritten mid-run without re-exporting a model. Existing in-browser GPT-2s (Transformer Explainer, tinygpt-webgpu) can read activations but none can change them.

Measured on an M3 Max in Chrome: a forward pass takes 15 ms at 16 tokens and 25 ms at 256; the logit lens adds about 2 ms for one position or 17 ms for every position of a 16-token prompt. Not yet measured on an ordinary laptop or in Safari.

## Precision: fp16 weights, fp32 arithmetic (2026-10-10)

Weights are stored as fp16 (249 MB, four 64 MB shards) and unpacked in the shader with `unpack2x16float`, so no GPU needs `shader-f16`. All arithmetic is fp32, because GPT-2's residual stream carries values near 3,000 on the first token and fp16 can't square them for LayerNorm.

Against TransformerLens with the original fp32 weights, rounding the weights to fp16 changes residuals by 0.3% (relative), and the logit lens's top guess agrees on 99.3–100% of (layer, position) rows. Every final prediction is unchanged. int8 would halve the download, but there is evidence it damages SAE features and we haven't measured it; revisit if 249 MB proves too heavy.

## Every prompt starts with `<|endoftext|>` (2026-10-10)

As TransformerLens does by default, and as the GPT-2 sparse autoencoders were trained. It changes some answers: "The Eiffel Tower is in the city of" predicts " Paris" without it and " London" with it. Exhibit prompts are chosen from real runs, never adjusted to look right.

## Checking correctness (2026-10-10)

`tools/reference/fixtures.py` records TransformerLens activations for five prompts in two sets: the original weights (`fixtures/fp32`, ground truth) and the same weights rounded to fp16 (`fixtures/fp16`), so a kernel bug can be told apart from rounding. Three intervention cases (zeroing the IOI name-mover heads, adding a vector at every position, patching one position from another prompt) check that writes behave too.

- `bun test`: tokenizer against Hugging Face, and a plain CPU forward pass against the fp16 set (within 1e-4).
- `bun run test:gpu` (with `bun run dev` running): the WebGPU engine in Chrome against both sets. Against fp16 it agrees within 1e-5 on residuals and attention and on every lens top-5.

Fixtures are 14 MB per set and not committed; `bun run fixtures` rebuilds them. Hook names follow TransformerLens, so learners who move on to it meet the same words.

## Toolchain: Bun, React, TypeScript (2026-10-10)

Bun's bundler and dev server instead of Vite, because Santa blocks esbuild on Shu's Mac. Python appears only in `tools/reference` (uv venv, versions pinned in `requirements.txt` without index URLs, so the repo doesn't record a private package mirror).

## Hosting (2026-10-10)

The app is static and goes on GitHub Pages. The weights go on the Hugging Face Hub, which serves large files with CORS; the browser checks each shard's SHA-256 and keeps it in the Cache API so later visits load from disk.

## Look: abstract, accurate, explorable (2026-10-10)

The model is drawn as it really is: one residual line per token, attention as the only place lines exchange information (and only from earlier tokens), the MLP working on each line alone, LayerNorms included. It is drawn abstractly (lines, arcs, fields of colour), not as an illustrated machine. Shu chose this after trying 1939-fair poster looks and a cutaway "giant machine" (look board: https://claude.ai/artifact/BLUNX9u8jjxQmnFrV5nkZN).

Three zoom levels share one engine: **Table** (all 144 heads and 12 MLPs as a map to explore, known jobs marked), **Signal** (the flow of one prompt, the core teaching view), **Field** (one activation's 768 numbers, opening into features).

## Patching uses the IOI prompt (2026-10-10)

"When Mary and John went to the store, John gave a drink to" replaces the Eiffel Tower example. Its circuit was found on GPT-2 small itself (Wang et al. 2022); ROME-style factual recall tracing is clean only on much larger GPT-2s.

## Features: `gpt2-small-res-jb`, layer 7 (planned)

Joseph Bloom's residual SAEs (MIT, plain ReLU, so a subset of features gives exact activations). Ship about 2,000 curated features as static files (about 8 MB). Labels and examples come from Neuronpedia's exports, whose licence is unstated: ask before launch.

## Head jobs are measured, not quoted (2026-10-10)

The Machine view marks heads with known jobs. Previous-token, duplicate-token and induction heads are measured in the learner's browser at load, on 20 random tokens shown twice, so a mark always reflects this model. The measured heads match the literature: 4.11 is a previous-token head (score 1.00), and 5.1, 5.5, 6.9, 7.2 and 7.10 are induction heads (0.82–0.94). Roles that need more than attention patterns to find (the IOI name movers, negative name movers and S-inhibition heads) are labelled with their source, Wang et al. 2022.

Head activity in the Machine view is the length of each head's output vector at the last token, after the output projection. That needs each layer's `hook_z`, which the engine now keeps and the GPU check compares with TransformerLens.

## Wordmark: "Hop" (2026-10-10)

The wordmark is "featurama" in IBM Plex Mono with the last "a" sending attention arcs back to the earlier two, over a blue residual line: what an induction head does with a repeated token. It draws itself once when the page loads, replays on hover, and loops on the loading screen. Shu chose it from five concepts on the look board.
