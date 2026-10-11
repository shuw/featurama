"""Measure every feature of the layer-7 sparse autoencoder on Wikipedia text.

Runs GPT-2 small over WikiText-103 (CC BY-SA) and, for each of the 24,576
features, records how often it fires, its strongest passages, the tokens it
fires on most, and which next tokens its direction pushes toward and away
from. Writes public/features/index.json (small, for search and lists) and
public/features/NN.json (the passages, 256 features per file), plus
fixtures/sae.json for the browser's parity check.

Needs a GPU or Apple silicon for a reasonable run time (a few minutes on an
M-series Mac). The output is committed; CI does not rerun it.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))
from heads import clean  # noqa: E402
from sae import load as load_sae, folded  # noqa: E402

root = Path(__file__).resolve().parents[2]
CONTEXT = 128
TOKENS = 3_000_000
BATCH = 32
KEEP = 4  # passages kept per feature
BEFORE, AFTER = 20, 4  # tokens shown around the strongest position
PER_FILE = 256
WORD = re.compile(r"^( [^\W\d_][^\W\d_]+| ?\d+)$")


def main():
    torch.set_grad_enabled(False)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2", device=dev)  # folded and centred, as the SAE was trained
    tok = model.tokenizer
    w = load_sae()
    W_enc = torch.tensor(w["W_enc"], device=dev)
    b_enc = torch.tensor(w["b_enc"], device=dev)
    b_dec = torch.tensor(w["b_dec"], device=dev)
    W_dec = torch.tensor(w["W_dec"], device=dev)
    F = W_enc.shape[1]
    encode = lambda x: torch.relu((x - b_dec) @ W_enc + b_enc)

    check_folding(model, w, dev)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    contexts = []
    total = 0
    for text in ds["text"]:
        if len(text) < 400 or text.strip().startswith("="):
            continue
        ids = tok(clean(text), add_special_tokens=False)["input_ids"]
        if ids and ids[0] == tok.bos_token_id:  # TransformerLens's tokenizer adds its own start token
            ids = ids[1:]
        ids = ids[: CONTEXT - 1]
        if len(ids) < 64:
            continue
        contexts.append([tok.bos_token_id] + ids)
        total += len(ids)
        if total >= TOKENS:
            break
    print(f"{len(contexts)} passages, {total} tokens", flush=True)

    fires = torch.zeros(F, device=dev)
    seen = 0
    best_v = torch.full((KEEP, F), -1.0, device=dev)
    best_at = torch.zeros((KEEP, F), dtype=torch.long, device=dev)  # context * CONTEXT + position
    for start in range(0, len(contexts), BATCH):
        chunk = contexts[start : start + BATCH]
        T = max(len(c) for c in chunk)
        toks = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.bool)
        for i, c in enumerate(chunk):
            toks[i, : len(c)] = torch.tensor(c)
            mask[i, 1 : len(c)] = True  # the start token is left out
        resid = model(toks.to(dev), stop_at_layer=7)  # resid_pre of layer 7
        acts = encode(resid)  # [B, T, F]
        acts = acts * mask.to(dev)[..., None]
        fires += (acts > 0).sum(dim=(0, 1))
        seen += int(mask.sum())
        # One passage per context per feature, then merge with the best so far.
        per_ctx, pos = acts.max(dim=1)  # [B, F]
        at = (torch.arange(start, start + len(chunk), device=dev)[:, None] * CONTEXT + pos)
        v = torch.cat([best_v, per_ctx])
        a = torch.cat([best_at, at])
        top = v.topk(KEEP, dim=0)
        best_v, best_at = top.values, a.gather(0, top.indices)
        if start % (BATCH * 50) == 0:
            print(f"  {start}/{len(contexts)}", flush=True)

    freq = (fires / seen).cpu().numpy()
    best_v, best_at = best_v.cpu().numpy(), best_at.cpu().numpy()

    # Direct effect of each feature's direction on the next-token scores.
    effect = W_dec @ model.W_U  # [F, vocab]
    effect = effect - effect.mean(dim=1, keepdim=True)
    vocab = [tok.decode([i]) for i in range(model.cfg.d_vocab)]
    allowed = torch.tensor([bool(WORD.match(t)) for t in vocab], device=dev)
    effect[:, ~allowed] = float("nan")
    up = torch.nan_to_num(effect, nan=-1e9).topk(6, dim=1).indices.cpu().numpy()
    down = torch.nan_to_num(-effect, nan=-1e9).topk(6, dim=1).indices.cpu().numpy()

    # Each feature's activation across its passages, rerunning each source passage once.
    wanted = {}
    for f in range(F):
        for k in range(KEEP):
            if best_v[k, f] > 0:
                c, p = divmod(int(best_at[k, f]), CONTEXT)
                wanted.setdefault(c, []).append((f, k, p))
    windows = {}
    order = sorted(wanted)
    for start in range(0, len(order), BATCH):
        cs = order[start : start + BATCH]
        T = max(len(contexts[c]) for c in cs)
        toks = torch.full((len(cs), T), tok.eos_token_id, dtype=torch.long)
        for i, c in enumerate(cs):
            toks[i, : len(contexts[c])] = torch.tensor(contexts[c])
        acts = encode(model(toks.to(dev), stop_at_layer=7))
        for i, c in enumerate(cs):
            for f, k, p in wanted[c]:
                lo, hi = max(1, p - BEFORE), min(len(contexts[c]), p + AFTER + 1)
                windows[f, k] = (c, p, lo, hi, acts[i, lo:hi, f].tolist())
        if start % (BATCH * 50) == 0:
            print(f"  passages {start}/{len(order)}", flush=True)

    index = []
    files = {}
    for f in range(F):
        passages = []
        firing = []
        for k in range(KEEP):
            if (f, k) not in windows:
                continue
            c, p, lo, hi, a = windows[f, k]
            scale = float(best_v[0, f])
            passages.append({"ids": contexts[c][lo:hi], "acts": [min(255, max(0, round(255 * x / scale))) for x in a], "at": p - lo})
            firing.append(contexts[c][p])
        top_act = float(max(best_v[0, f], 0))
        index.append([round(float(freq[f]) * 1e6), round(top_act, 2), list(dict.fromkeys(firing))[:3], up[f][:3].tolist()])
        files.setdefault(f // PER_FILE, []).append({"up": up[f].tolist(), "down": down[f].tolist(), "passages": passages})

    out = root / "public" / "features"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": "WikiText-103 (CC BY-SA 3.0)", "tokens": seen, "per_file": PER_FILE, "fields": ["fires per million tokens", "strongest activation", "tokens it fires on most", "tokens it pushes toward most"]}
    (out / "index.json").write_text(json.dumps({"meta": meta, "features": index}, separators=(",", ":")))
    for n, rows in files.items():
        (out / f"{n:02d}.json").write_text(json.dumps(rows, separators=(",", ":")))
    alive = sum(1 for r in index if r[0] > 0)
    print(f"wrote {out}: {alive} of {F} features fire on this text")


def check_folding(model, w, dev):
    """The browser applies the folded encoder to the raw residual stream; check
    it matches the original encoder on TransformerLens's centred stream, and
    write the expected top features for the browser's parity test."""
    raw = HookedTransformer.from_pretrained_no_processing("gpt2", device=dev)
    prompt = "When Mary and John went to the store, John gave a drink to"
    toks = model.to_tokens(prompt)
    centred = model(toks, stop_at_layer=7)
    x = raw(toks, stop_at_layer=7)
    W, b = folded(w)
    mine = torch.relu(x.cpu().double() @ torch.tensor(W) + torch.tensor(b))
    ref = torch.relu((centred.cpu().double() - torch.tensor(w["b_dec"]).double()) @ torch.tensor(w["W_enc"]).double() + torch.tensor(w["b_enc"]).double())
    err = (mine[0, 1:] - ref[0, 1:]).abs().max().item()
    print(f"folded encoder vs original: max difference {err:.2e} (largest activation {ref[0, 1:].max().item():.1f})")
    assert err < 1e-2, err
    top = ref[0].topk(8, dim=-1)
    fixture = {"prompt": prompt, "tokens": toks[0].tolist(), "top": [[[int(i), round(float(v), 4)] for i, v in zip(top.indices[t], top.values[t])] for t in range(toks.shape[1])]}
    (root / "fixtures" / "sae.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
