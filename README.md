# Featurama

Run interpretability experiments on a real language model (GPT-2 small) in your browser. See [docs/BRIEF.md](docs/BRIEF.md) for what it is and [DECISIONS.md](DECISIONS.md) for why it's built the way it is.

## Develop

Needs Bun, Chrome, and (to rebuild the model files and test fixtures) Python 3.12 with uv.

```sh
bun install
uv venv tools/reference/.venv --python 3.12
VIRTUAL_ENV=tools/reference/.venv uv pip install -r tools/reference/requirements.txt
bun run model       # GPT-2 weights into public/model/gpt2 (249 MB)
bun run fixtures    # TransformerLens reference activations into fixtures/
bun run dev         # http://localhost:5190
```

Checks: `bun run check` (types, tokenizer, CPU reference) and, with the dev server running, `bun run test:gpu` (the WebGPU engine against TransformerLens, plus timings). `node scripts/look.mjs` saves screenshots of the exhibit to `shots/`.
