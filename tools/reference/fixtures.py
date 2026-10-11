"""Reference activations from TransformerLens, for checking the browser engine.

Two sets, so engine bugs can be told apart from weight rounding:
  fixtures/fp32/  the original weights (ground truth)
  fixtures/fp16/  the same weights rounded to fp16, as the browser stores them

Each prompt is a JSON file of arrays, base64 little-endian, with shapes.
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[2]
out = root / "fixtures"
BOS = 50256

PROMPTS = {
    "eiffel": "The Eiffel Tower is in the city of",
    "ioi": "When Mary and John went to the store, John gave a drink to",
    "years": "The war lasted from the year 1732 to the year 17",
    "fair": (
        "In 1939 the New York World's Fair opened in Flushing Meadows. Its theme was the "
        "world of tomorrow, and its most popular exhibit was Futurama, a ride over a model "
        "of America in 1960."
    ),
}

TOKENIZER_CASES = [
    "Hello world",
    " leading space",
    "Featurama",
    "The Eiffel Tower is in the city of",
    "don't you'll we've I'm she's THEY'RE",
    "  multiple   spaces\n\nand\ttabs  ",
    "numbers 1234567 and 3.14159",
    "emoji 🎡🎢 and accents café naïve",
    "日本語のテキスト",
    " SolidGoldMagikarp",
    "trailing space ",
    "a\r\nb",
    "",
]


def arr(x, dtype):
    a = np.ascontiguousarray(np.asarray(x, dtype=dtype))
    return {"dtype": dtype, "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def record(model, tokens):
    t = torch.tensor([tokens])
    with torch.no_grad():
        logits, cache = model.run_with_cache(t)
        n = model.cfg.n_layers
        resid = torch.stack([cache["blocks.0.hook_resid_pre"][0]] + [cache[f"blocks.{i}.hook_resid_post"][0] for i in range(n)])
        mid = torch.stack([cache[f"blocks.{i}.hook_resid_mid"][0] for i in range(n)])
        pattern = torch.stack([cache[f"blocks.{i}.attn.hook_pattern"][0] for i in range(n)])
        z = torch.stack([cache[f"blocks.{i}.attn.hook_z"][0] for i in range(n)])
        qkv = torch.stack([torch.cat([cache[f"blocks.{i}.attn.hook_{x}"][0].flatten(1) for x in "qkv"], dim=-1) for i in range(n)])
        neurons = torch.stack([cache[f"blocks.{i}.mlp.hook_post"][0] for i in range(n)])
        # Logit lens: the final LayerNorm recomputed on each layer's residual, then unembed.
        lens = model.unembed(model.ln_final(resid)).log_softmax(-1)
        lens_top = lens.topk(5, dim=-1)
        final = logits[0].log_softmax(-1).topk(10, dim=-1)
    return {
        "tokens": arr(tokens, "int32"),
        "resid": arr(resid, "float32"),
        "resid_mid": arr(mid, "float32"),
        "pattern": arr(pattern, "float32"),
        "z": arr(z, "float32"),
        "qkv": arr(qkv, "float32"),
        "neurons": arr(neurons, "float32"),
        "logits_last": arr(logits[0, -1], "float32"),
        "top_ids": arr(final.indices, "int32"),
        "top_logprobs": arr(final.values, "float32"),
        "lens_ids": arr(lens_top.indices, "int32"),
        "lens_logprobs": arr(lens_top.values, "float32"),
    }


def interventions(model, prompts):
    """Runs with activations changed, in the engine's intervention format."""
    cases = []

    def run(tokens, hooks):
        with torch.no_grad():
            return model.run_with_hooks(torch.tensor([tokens]), fwd_hooks=hooks)[0, -1]

    # Switch off the name-mover heads (Wang et al. 2022) on the IOI prompt.
    heads = {9: [6, 9], 10: [0]}

    def zero(hs):
        def hook(z, hook):
            z[:, :, hs, :] = 0
            return z
        return hook

    logits = run(prompts["ioi"], [(f"blocks.{l}.attn.hook_z", zero(hs)) for l, hs in heads.items()])
    cases.append({
        "name": "ablate name movers", "tokens": prompts["ioi"], "logits_last": arr(logits, "float32"),
        "interventions": [{"kind": "head", "layer": l, "head": h, "scale": 0} for l, hs in heads.items() for h in hs],
    })

    # Switch off two MLPs.
    def zero_out(x, hook):
        return torch.zeros_like(x)

    logits = run(prompts["ioi"], [(f"blocks.{l}.hook_mlp_out", zero_out) for l in (3, 9)])
    cases.append({
        "name": "unplug two MLPs", "tokens": prompts["ioi"], "logits_last": arr(logits, "float32"),
        "interventions": [{"kind": "mlp", "layer": l, "scale": 0} for l in (3, 9)],
    })

    # Add a fixed random direction to the residual stream at every position.
    v = np.random.default_rng(7).normal(size=model.cfg.d_model).astype(np.float32) * 4
    logits = run(prompts["eiffel"], [("blocks.6.hook_resid_pre", lambda r, hook: r + torch.tensor(v))])
    cases.append({
        "name": "steer every position", "tokens": prompts["eiffel"], "logits_last": arr(logits, "float32"),
        "interventions": [{"kind": "resid", "hook": "resid_pre", "layer": 6, "position": "all", "mode": "add", "value": arr(v, "float32")}],
    })

    # Patch one position's residual from another prompt's run.
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.tensor([prompts["years"]]))
    donor = cache["blocks.5.hook_resid_mid"][0, -1].clone()
    last = len(prompts["eiffel"]) - 1

    def patch(r, hook):
        r[0, last] = donor
        return r

    logits = run(prompts["eiffel"], [("blocks.5.hook_resid_mid", patch)])
    cases.append({
        "name": "patch one position", "tokens": prompts["eiffel"], "logits_last": arr(logits, "float32"),
        "interventions": [{"kind": "resid", "hook": "resid_mid", "layer": 5, "position": last, "mode": "set", "value": arr(donor, "float32")}],
    })
    return cases


def next_token(rec):
    ids = np.frombuffer(base64.b64decode(rec["top_ids"]["b64"]), dtype=np.int32).reshape(rec["top_ids"]["shape"])
    return int(ids[-1, 0])


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
    model.eval()

    prompts = {name: [BOS] + tokenizer.encode(text) for name, text in PROMPTS.items()}
    rng = np.random.default_rng(1939)
    half = rng.integers(1000, 20000, size=12).tolist()
    prompts["repeat"] = [BOS] + half + half

    for variant in ["fp32", "fp16"]:
        if variant == "fp16":
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(p.half().float())
        d = out / variant
        d.mkdir(parents=True, exist_ok=True)
        for name, tokens in prompts.items():
            rec = record(model, tokens)
            (d / f"{name}.json").write_text(json.dumps({"name": name, "text": PROMPTS.get(name), **rec}))
            print(f"{variant} {name}: {len(tokens)} tokens, next token {tokenizer.decode([next_token(rec)])!r}")
        (d / "interventions.json").write_text(json.dumps(interventions(model, prompts)))

    cases = [{"text": s, "ids": tokenizer.encode(s)} for s in TOKENIZER_CASES]
    (out / "tokenizer.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f"{len(cases)} tokenizer cases")


if __name__ == "__main__":
    main()
