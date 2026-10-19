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

## Three sections (2026-10-11)

- The header has three sections: Explore (the machine and a page per part: Machine, Heads, MLPs, Layers, Features), Lab (every experiment that changes a run: Unplug, Swap, Steer, Trace) and Play. Sections and pages share one header row: the open section is a white group in the section pill, with its pages unfolded to its right, and the other sections fold theirs away (inert, so Tab skips them). Below 940 px the pill gets its own row; on phones the sections become three equal tabs with the open section's pages in a scrolling row beneath. ⌥1–⌥3 switch sections; ⌥ and a letter, underlined in the page's name, opens a page (Machine M, Heads H, MLPs P, Layers L, Features A; Unplug G, Swap S, Steer R, Trace T; Puzzles Z, You vs GPT-2 V). The letters avoid Mac accent keys (⌥E, ⌥U, ⌥I, ⌥N) and Alt+D, E and F, which Windows and Linux browsers use. Pages no longer repeat the section and page name above their title. The Guide switch is gone: it only hid one short hint on three pages (Heads, Layers' guess view, Swap), so the hints are always shown.
- Before this, pages were split two ways at once, by technique (Attention, Layers, Patching, Features) and by zoom level (a separate Zoom tab), so most parts had two homes. Now each part has one page that goes from overview to single numbers: a head's page shows where it looks, then inside it, what it copies and its wiring; Layers reads the same grid as a guess, as numbers, or on the token map.
- The Machine shades parts by how much they add or by how they vote for the answer. Unplugging moved to the Lab; Machine cards link there with the part already unplugged (?off=).
- Old addresses (/attention, /patching, /zoom/...) redirect to their new places.

## Looking inside parts (2026-10-11)

- The engine keeps every layer's queries, keys, values and MLP neurons (about 65 MB more GPU memory at 256 tokens), and can unplug a whole MLP. Both match TransformerLens.
- Who voted: each part's output through the final LayerNorm, at that position's real scale, against the difference between two tokens' embeddings. The votes add up exactly to the score gap, because the LayerNorm's scale is fixed from the real run.
- Tracing back unplugs parts one at a time (zero ablation), 156 runs for the answer and up to 155 for each later step, a few seconds each. Deeper steps measure how much a part's output changes, not the answer, so a step stays meaningful however far back it goes. Zero ablation overstates MLP 0, which GPT-2 uses as part of reading tokens in; the page says so.
- Neuron passages come from 2 million WikiText-103 tokens (tools/reference/neurons.py, output committed, 11 MB). What a neuron pushes toward is computed in the browser from its output weights.
- Wiring is composition scores from TransformerLens (Elhage et al. 2021). A head's wires are its three strongest sources of each kind, if at least twice the typical score. The copying table runs each token through layer 0 for real before applying the head's value and output weights, since the raw embedding gets copy suppression wrong.
- The token map is a UMAP of the token embedding, computed once (tools/reference/weights.py; umap-learn isn't in requirements.txt because CI doesn't run it). Data lives in public/zoom-data, not public/zoom, so the folder can't shadow the /zoom page.

## Weights tower (2026-10-16)

- Explore › Weights (⌥W) draws all 124M weights in 3D: the token table on the ground, one floor per layer with its matrices lying flat (inputs front to back, outputs left to right), and the same table again on the roof as the unembedding. Biases, LayerNorm parameters and the vectors flowing in (running summary, normalised input, blended values, neurons) are thin strips beside the matrices.
- It renders with WebGPU on the engine's own device, reading the weight buffers the forward pass uses; nothing is copied. Far away, a pyramid of 4×4, 16×16 and 64×64 block averages (mean and RMS) stands in for the weights; it's built on the GPU once, and again for the forward pass whenever the prompt or token changes.
- "The forward pass" colours each weight by weight × its input for the chosen token; the token and position tables light only the row that's used, and the roof uses the normalised final summary. "Run it up the tower" reveals floors in order.
- Positions are sent to the GPU relative to the eye, computed in double precision, with reversed-Z depth, so you can zoom from the whole tower to one weight (its value and fp16 bits) without jitter.
- Colours are scaled per matrix by a typical value (80th percentile), not RMS, because GPT-2's few huge activations would otherwise darken everything else.
- Pointing and zooming only count matrices and vectors, so empty floor space doesn't hide the floors below.
- Navigation is structured so it's hard to get lost: drag moves like a map (the point under the pointer stays under it), shift-drag only turns about the vertical, and the camera tilts by zoom level, from an angled view of the tower to almost straight down on a floor. The view can't leave the tower. Clicking a part flies to it facing straight on; a breadcrumb (Tower › Layer › matrix), Esc, a floor rail and Page Up/Down move between levels.
- Close to a floor, it's in focus: other floors fade, and floors above it are hidden, since they'd sit between the camera and the floor.
- "Animate the pass" (on by default) plays the forward pass as a short film, paced to be followed rather than in real time: a 1.6 s establishing shot of the whole tower, a 10 s climb (eased, a little longer on the ground and roof), then the answer. The camera keeps the whole tower in view, turning slowly and rising with the pass; a glowing pulse climbs a line beside the floors, and each floor lights in two soft sweeps (attention, then MLP). Floating callouts tie the input to its row in the token table and the answer to its row on the roof.
- A new token to follow (generated, typed or picked) replays the pass once things settle, skipping the establishing shot. Typing updates the colours in place.
- Zoom eases toward its goal; floors fade continuously by distance from the focus, and floors above it dissolve rather than vanish. Floor labels sit at the right end of each floor, out of the line's way; matrix names appear only once a floor is in focus.
- The view can go full screen (button or F).
- Weights are also coloured against a typical value (80th percentile): some matrices, like the MLP output, have a few huge weights that would otherwise make the rest look black.

## Overview home page (2026-10-19)

- A friend found it hard to see how the parts fit together, so the site now opens on an Overview (⌥O): one diagram of the whole path on your prompt. Tokens are lanes running down through the 12 layers; in each layer attention arcs bring information into the last token and an MLP square sits on every lane; the last lane is scored against all 50,257 tokens at the bottom, and a column on the right shows the logit lens guess after each layer. The Machine moved to /machine.
- Seven states: the whole path, then six steps in the order a token meets the parts: tokens, numbers, attention, MLPs, adding up (the residual stream), the guess. The stepper above the diagram is itself the mental model, with attention and MLPs boxed as "× 12 layers". Each step lights its part, dims the rest, quotes real numbers from the run, says why the part matters, and links to the page that goes deeper; adding up offers to unplug the head it named, and the guess offers to watch GPT-2 write.
- The diagram plays once when the first run lands (tokens, numbers, then each layer in order, then the guess) and again on request; parts that change while typing don't animate. On narrow screens the step card docks at the bottom so the diagram stays in view.
- Each layer's arc is its head that brings the most from one earlier token (attention share times the size of that token's value after the output projection, against the stream's size there). The bold head in the attention step, and the MLP named in the MLP step, are the ones whose direct vote for GPT-2's answer over its second guess is largest, so on the IOI prompt it's 9.9 looking at "Mary". Raw output size alone always picked layer-11 parts.
- Prompts longer than 20 tokens show the start token and the last 19. Tight lanes get their token names at an angle.
