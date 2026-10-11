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

## Interventions in the interface (2026-10-11)

- **Unplugging** a head sets its output to zero (zero ablation), on every token. With the three name movers unplugged, GPT-2 still answers "Mary" (68% → 50%): backup heads take over, which the IOI paper calls self-repair. Unplugging the eight backup name movers as well takes "Mary" to 2%.
- **Patching** copies A's residual stream at one (layer, token) into B, before that layer. A square's colour is how far B's answer moves toward A's, measured on the difference between the two answers' log-probabilities: 0 means unchanged, 1 means as sure as A. The default pair reproduces the published picture: the deciding information sits on the second name until layer 8 and on the last token from layer 9.
- **Let it write** samples from the top 40 tokens at three temperatures (0, 0.8, 1.3). At 0 it shows GPT-2's habit of repeating itself.
- **You vs GPT-2** uses 31 everyday sentences whose answers are single GPT-2 tokens, so both players are scored on the same token.

## Explaining every head (2026-10-11)

Each head's card combines two kinds of evidence, kept apart so readers can tell them apart:

- **What researchers found**, reworded in plain English with a link to the source. Most labels come from a study that inspected every head with sparse autoencoders (Krzyzanowski, Kissane, Conmy & Nanda 2024; republished CC BY 4.0 in arXiv 2406.17759). Six heads have no description there and two have no row, and the card says so. Task-specific roles come from the IOI, greater-than, copy-suppression, sequence-continuation, acronym and gendered-pronoun studies; the last is a hackathon report and is labelled as one.
- **What we measured**, from `tools/reference/heads.py` on 300 paragraphs of WikiText-103 (CC BY-SA): how much attention rests on the start token, goes to the previous token or to itself, how far back it reaches, and which whole words draw it. A copying score from the head's weights says whether it raises or lowers the odds of the tokens it looks at, using GPT-2's effective token embedding (the embedding plus layer 0's MLP). With the raw embedding the score called copy-suppression head 10.7 a copier, so it isn't used. It is only shown for layers 6 and up.

Each head also shows the passages where it works hardest (`public/head-examples.json`, fetched on demand), with its attention highlighted. "Work" is the size of what the head writes, leaving out what it copies from the start token, where idle heads rest. For long-range heads the passage includes the word it looks at most, with a gap before the nearby text. The cards link to the survey's per-head dashboards, and a few well-understood heads offer a prompt that should wake them up.

## Direct effect and token healing (2026-10-11)

- Head and MLP cards show what that part's output, on its own, pushes the next token toward and away from: its vector at the chosen token through the final LayerNorm (with that position's real scale) and the unembedding. Computed in the browser in about 40 ms; it ignores what later layers do with the vector. Lists show whole words and numbers only, leaving out word fragments and GPT-2's glitch tokens such as " RandomRedditor", which crowd the extremes. On the Mary-and-John sentence, name mover 9.9 pushes toward " Mary" and copy-suppression head 10.7 away from it; in "from 1732 to 17", MLP 10 pushes toward later years, as the greater-than study found.
- A prompt ending in spaces is run without them, and suggestions are limited to tokens that start with those spaces. Accepting one adds the rest of the token. GPT-2 attaches spaces to the following word, so a bare trailing space is text it rarely saw.
- The grids no longer carry letter badges. Highlight chips outline heads known to do one job, and pointing at a square shows its label.

## Attention views (2026-10-11)

- Arrows point from the selected token to what it looks at. Resting attention on the start token stays as a thin dashed arrow; pointing at another token previews its arrows faintly.
- The head picker is a heat map of how much each head adds at the selected token, with the current head boxed.
- "All heads at once" combines every head's attention, leaving out the start token and weighting each head by how much it adds at that token. It shows where information was gathered from, not what caused the prediction; the page says so.

## Addresses (2026-10-11)

- Pages have real paths (/featurama/attention/9.9) instead of #/ routes, and old #/ links redirect. GitHub Pages has no server routing, so the build copies index.html to 404.html. A deep link opened fresh therefore arrives with a 404 status, which browsers log but people don't see.
- An edited prompt goes in the address as ?p=, so a link shares it. A shared link skips the opening questions on Attention and Layers and goes straight to the shared prompt.

## Features and steering (2026-10-11)

- Features come from Joseph Bloom's sparse autoencoder on the stream entering layer 7 (24,576 features, MIT, pinned revision). The centring TransformerLens applies and the decoder bias are folded into the encoder, so the browser applies it to GPT-2's raw stream; the GPU result matches TransformerLens to 0.3%.
- The encoder (38 MB) loads when the Features page first opens. Steering directions load 256 at a time, when used.
- Each feature's frequency, strongest passages and pushed words were measured on 3 million WikiText-103 tokens by tools/reference/features.py (needs a GPU or Apple silicon; output committed, not rerun in CI). Neuronpedia's explanations aren't used: their licence isn't stated.
- Most features are named by the tokens they fire on, plus the words they push toward. A handful are hand-labelled after reading their passages.
- Steering adds the feature's direction at every token but the start token, including newly written ones. Strengths of 30–60 change the topic; above about 90, GPT-2 loses grammar. The two versions share their dice rolls, so differences come from the steering.
