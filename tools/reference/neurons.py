"""Measure every MLP neuron of GPT-2 small on Wikipedia text.

Runs GPT-2 small over WikiText-103 (CC BY-SA) and, for each of the 12 x 3,072
neurons, records how often it fires, its strongest passages and the tokens it
fires on most. Which next tokens a neuron pushes toward is computed in the
browser from the weights. Writes public/neurons/LL.json, one file per layer.

Needs a GPU or Apple silicon for a reasonable run time. The output is
committed; CI does not rerun it.
"""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))
from heads import clean  # noqa: E402

root = Path(__file__).resolve().parents[2]
CONTEXT = 128
TOKENS = 2_000_000
BATCH = 32
KEEP = 3  # passages kept per neuron
BEFORE, AFTER = 10, 2  # tokens shown around the strongest position
FIRE = 0.5  # a neuron counts as firing above this (GELU output)


def main():
    torch.set_grad_enabled(False)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device=dev)
    tok = model.tokenizer
    L, N = model.cfg.n_layers, model.cfg.d_mlp

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    contexts = []
    total = 0
    for text in ds["text"]:
        if len(text) < 400 or text.strip().startswith("="):
            continue
        ids = tok(clean(text), add_special_tokens=False)["input_ids"]
        if ids and ids[0] == tok.bos_token_id:
            ids = ids[1:]
        ids = ids[: CONTEXT - 1]
        if len(ids) < 64:
            continue
        contexts.append([tok.bos_token_id] + ids)
        total += len(ids)
        if total >= TOKENS:
            break
    print(f"{len(contexts)} passages, {total} tokens", flush=True)

    names = [f"blocks.{l}.mlp.hook_post" for l in range(L)]
    fires = torch.zeros(L, N, device=dev)
    seen = 0
    best_v = torch.full((KEEP, L, N), -1e9, device=dev)
    best_at = torch.zeros((KEEP, L, N), dtype=torch.long, device=dev)
    for start in range(0, len(contexts), BATCH):
        chunk = contexts[start : start + BATCH]
        T = max(len(c) for c in chunk)
        toks = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.bool)
        for i, c in enumerate(chunk):
            toks[i, : len(c)] = torch.tensor(c)
            mask[i, 1 : len(c)] = True  # the start token is left out
        _, cache = model.run_with_cache(toks.to(dev), names_filter=lambda n: n in names)
        acts = torch.stack([cache[n] for n in names], dim=2)  # [B, T, L, N]
        m = mask.to(dev)[..., None, None]
        fires += ((acts > FIRE) & m).sum(dim=(0, 1))
        seen += int(mask.sum())
        acts = torch.where(m, acts, torch.tensor(-1e9, device=dev))
        per_ctx, pos = acts.max(dim=1)  # [B, L, N]
        at = torch.arange(start, start + len(chunk), device=dev)[:, None, None] * CONTEXT + pos
        v = torch.cat([best_v, per_ctx])
        a = torch.cat([best_at, at])
        top = v.topk(KEEP, dim=0)
        best_v, best_at = top.values, a.gather(0, top.indices)
        if start % (BATCH * 50) == 0:
            print(f"  {start}/{len(contexts)}", flush=True)

    freq = (fires / seen).cpu()
    best_v, best_at = best_v.cpu(), best_at.cpu()

    # Each neuron's activation across its passages, rerunning each source passage once.
    wanted = {}
    for l in range(L):
        for n in range(N):
            for k in range(KEEP):
                if best_v[k, l, n] > 0:
                    c, p = divmod(int(best_at[k, l, n]), CONTEXT)
                    wanted.setdefault(c, []).append((l, n, k, p))
    windows = {}
    order = sorted(wanted)
    for start in range(0, len(order), BATCH):
        cs = order[start : start + BATCH]
        T = max(len(contexts[c]) for c in cs)
        toks = torch.full((len(cs), T), tok.eos_token_id, dtype=torch.long)
        for i, c in enumerate(cs):
            toks[i, : len(contexts[c])] = torch.tensor(contexts[c])
        _, cache = model.run_with_cache(toks.to(dev), names_filter=lambda n: n in names)
        acts = torch.stack([cache[n] for n in names], dim=2).cpu()  # [B, T, L, N]
        for i, c in enumerate(cs):
            for l, n, k, p in wanted[c]:
                lo, hi = max(1, p - BEFORE), min(len(contexts[c]), p + AFTER + 1)
                windows[l, n, k] = (c, p, lo, hi, acts[i, lo:hi, l, n].tolist())
        if start % (BATCH * 50) == 0:
            print(f"  passages {start}/{len(order)}", flush=True)

    out = root / "public" / "neurons"
    out.mkdir(parents=True, exist_ok=True)
    for l in range(L):
        rows = []
        for n in range(N):
            passages = []
            firing = []
            scale = float(best_v[0, l, n])
            for k in range(KEEP):
                if (l, n, k) not in windows:
                    continue
                c, p, lo, hi, a = windows[l, n, k]
                passages.append([contexts[c][lo:hi], [min(255, max(0, round(255 * x / scale))) for x in a], p - lo])
                firing.append(contexts[c][p])
            rows.append([round(float(freq[l, n]) * 1e6), round(max(scale, 0), 2), list(dict.fromkeys(firing)), passages])
        meta = {"source": "WikiText-103 (CC BY-SA 3.0)", "tokens": seen, "fires_above": FIRE,
                "fields": ["fires per million tokens", "strongest activation", "tokens it fires on most", "passages: [token ids, activation 0-255, strongest position]"]}
        (out / f"{l:02d}.json").write_text(json.dumps({"meta": meta, "neurons": rows}, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out.glob("*.json"))
    print(f"wrote {out} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
