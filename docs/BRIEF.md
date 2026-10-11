# Featurama: project brief

Revised 2026-10-10. Technical choices and their reasons are in [DECISIONS.md](../DECISIONS.md).

## What it is

Featurama teaches how large language models work by letting you run interpretability experiments on a real model in your browser. No code, no install, no account. The name is "feature" plus "-rama", and a nod to Futurama, the ride at the 1939 World's Fair that flew visitors over a model of the future.

It fills the gap between architecture explainers (Transformer Explainer, LLM Visualization, AnimatedLLM), which show the parts but stop there, and research tools (Neuronpedia, ARENA, TransformerLens), which go deep but assume code. No existing tool lets a non-coder patch activations or read a logit lens.

## Audience and principles

Technically curious people with no machine-learning background. It is a public site.

- **Real model, real results.** Every experiment runs GPT-2 small on the learner's own text. Nothing is faked or scripted.
- **Do, then explain.** Poke first, read second.
- **Correct above all.** The engine is tested against TransformerLens, and the drawing of the model is true to its architecture.
- **Honest about limits.** Where results are messy or contested, say so.
- **Instant.** Interactions answer in well under a second once the model has loaded.

## How it feels

Free roaming with an optional guide, like a machine you can open and tinker with, rather than a linear course. The home screen is the whole model, drawn abstractly and accurately:

- **Signal** (the core view): one line per token, attention arcs between lines, MLP marks on each line, the logit lens underneath. Click a layer to open it.
- **Table** (the map): all 144 attention heads and 12 MLPs as a periodic table. Heads with jobs found in published research are marked; the rest invite exploring.
- **Field** (the microscope): one activation's 768 numbers, which later split into features.

Exhibits hang off the parts they explain. Each follows four beats: **predict** (lock in a guess), **poke** (the live experiment), **explain** (two or three short paragraphs, each term defined once), **challenge** (a checkable task; completing it fills in your progress).

Visual language: blue for the residual stream, orange for attention, teal for MLPs, one colour each everywhere. Motion only to show cause and effect.

## Exhibits

| # | Exhibit | Experiment | Release |
|---|---|---|---|
| 1 | Tokens | Watch text split into tokens (" Eiffel" is three) | v1 |
| 2 | Attention | See what each token reads; find the head that links a name back | v1 |
| 3 | Layers | Logit lens: watch the best guess change layer by layer | v1, first |
| 4 | Patching | "When Mary and John went to the store, John gave a drink to": swap activations until the answer flips | v1 |
| 5 | Features | Sparse-autoencoder features that fire on your text | v1 |
| 6 | Steering | Turn a feature up and watch the writing bend | v1 |
| 7 | Embeddings | Nearest neighbours of a word | later |
| 8 | Superposition | Train a toy model live | later |
| 9 | Circuits | Find the induction heads, switch them off | later |
| 10 | Attribution graphs | Trace one answer through features | later |

## Scope of v1

Exhibits 1–6, the three views, a landing page with model-loading progress, progress kept in the browser, light and dark themes, desktop Chrome, Edge and Safari. Not in v1: accounts, other models, a phone-first layout, exhibits 7–10, shareable links, translations.

## First milestone

1. Engine: GPT-2 small on WebGPU with readable and writable activations. **Done**: see DECISIONS.md for timings.
2. Verify against TransformerLens. **Done**: `bun test`, `bun run test:gpu`.
3. Exhibit 3, the logit lens, in the Signal view with all four beats. **Done.**
4. Review with Shu. **Done**; next were the Table home (Machine view) and Exhibit 2 (Attention). **Done.**
