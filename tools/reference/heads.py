"""Measure what each of GPT-2 small's 144 attention heads tends to do.

Also writes public/head-examples.json: for each head, the passages where it
does the most work (how much it writes, leaving out attention resting on the
start token), with its attention over the words before.

Runs TransformerLens over a sample of WikiText-103 (CC BY-SA) and writes
src/app/head-atlas.json: for each head, where it looks (start token, previous
token, itself, how far back, punctuation, which tokens draw it) and a copying
score from its weights (does it make the tokens it looks at more or less
likely to come next). The app turns these numbers into plain sentences.
"""

import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

root = Path(__file__).resolve().parents[2]
SNIPPETS = 300
MAX_TOKENS = 128
EXAMPLES = 5  # passages per head
WINDOW = 20  # tokens shown before the head's position


def clean(text):
    # WikiText-103 "raw" still spaces out punctuation and joins hyphens with @-@.
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([,.;:!?)\]'])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    text = text.replace(" n't", "n't").replace(" 's", "'s")
    return text.strip()


def main():
    torch.set_grad_enabled(False)
    # LayerNorm folded into the weights, as the copy-suppression work does.
    # Attention patterns are the same either way.
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    L, H = model.cfg.n_layers, model.cfg.n_heads
    tok = model.tokenizer

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [clean(t) for t in ds["text"] if len(t) > 300 and not t.strip().startswith("=")][:SNIPPETS]

    punct_ids = {i for t, i in tok.get_vocab().items() if re.fullmatch(r"Ġ?[\.,;:!?\"'()\-–—]+", t)}

    rest = torch.zeros(L, H)
    selfa = torch.zeros(L, H)
    prev = torch.zeros(L, H)
    dist = torch.zeros(L, H)
    punct = torch.zeros(L, H)
    n = 0
    vocab = model.cfg.d_vocab
    received = torch.zeros(L, H, vocab)  # attention mass each key token receives (start token excluded)
    freq = torch.zeros(vocab)
    best = [[] for _ in range(L * H)]  # per head: (work, snippet, position, attention row)
    all_tokens = []

    for text in texts:
        tokens = model.to_tokens(text)[:, :MAX_TOKENS]  # starts with <|endoftext|>
        T = tokens.shape[1]
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith(("hook_pattern", "hook_v", "hook_z")))
        pat = torch.stack([cache[f"blocks.{l}.attn.hook_pattern"][0] for l in range(L)])  # [L, H, T, T]
        all_tokens.append(tokens[0])
        # Work: the size of what each head writes at each position, leaving
        # out what it copies from the start token (where idle heads rest).
        v = torch.stack([cache[f"blocks.{l}.attn.hook_v"][0] for l in range(L)])  # [L, T, H, dh]
        z = torch.stack([cache[f"blocks.{l}.attn.hook_z"][0] for l in range(L)])  # [L, T, H, dh]
        z = z - pat.permute(0, 2, 1, 3)[..., 0:1] * v[:, 0:1]  # remove the start token's share
        work = torch.einsum("lthd,lhdm->lhtm", z, model.W_O).norm(dim=-1)  # [L, H, T]
        work[..., :2] = 0
        top = work.max(dim=-1)
        for l in range(L):
            for h in range(H):
                i = int(top.indices[l, h])
                best[l * H + h].append((float(top.values[l, h]), len(all_tokens) - 1, i, pat[l, h, i, : i + 1].clone()))
        q = pat[:, :, 2:, :]  # queries from position 2 on, so "previous" never means the start token
        idx = torch.arange(2, T)  # query positions; row r of q is position idx[r]
        rows = idx - 2
        rest += q[..., 0].sum(-1)
        selfa += q[..., rows, idx].sum(-1)
        prev += q[..., rows, idx - 1].sum(-1)
        offsets = (idx[:, None] - torch.arange(T)[None, :]).clamp(min=0).float()
        offsets[:, 0] = 0  # ignore the start token for distance
        active = q[..., 1:].sum(-1).clamp(min=1e-6)
        dist += ((q * offsets).sum(-1) / active).sum(-1)
        is_p = torch.tensor([int(t) in punct_ids for t in tokens[0]])
        punct += ((q * is_p).sum(-1) / active).sum(-1)
        ids = tokens[0, 1:]
        received.index_add_(2, ids, q[..., 1:].sum(2))
        freq.index_add_(0, ids, torch.ones(len(ids)))
        n += T - 2

    rest, selfa, prev, dist, punct = (x / n for x in (rest, selfa, prev, dist, punct))

    # Copying score (Elhage et al. 2021): eigenvalues of the head's OV circuit
    # from token to logit, E W_V W_O W_U, whose nonzero eigenvalues match the
    # 64 x 64 matrix W_O W_U E W_V. Positive: what the head looks at, it makes
    # more likely next. Negative: less likely. E is GPT-2's effective token
    # embedding, the embedding plus layer 0's MLP (McDougall et al. 2023);
    # with the raw embedding, known copiers and suppressors come out wrong.
    WE = model.W_E
    E = WE + model.blocks[0].mlp(model.blocks[0].ln2(WE.unsqueeze(0)))[0]
    E = E / E.norm(dim=-1, keepdim=True) * WE.norm(dim=-1, keepdim=True).mean()
    UE = model.W_U @ E  # [d, V] @ [V, d] -> [d, d]
    copying = torch.zeros(L, H)
    for l in range(L):
        for h in range(H):
            M = model.W_O[l, h] @ UE @ model.W_V[l, h]  # [d_head, d_head]
            ev = torch.linalg.eigvals(M)
            copying[l, h] = (ev.real.sum() / ev.abs().sum()).item()

    # Tokens that draw a head's attention far more than their frequency would
    # explain, among tokens seen at least 20 times.
    common = freq >= 20
    share = freq / freq.sum()
    heads = {}
    for l in range(L):
        for h in range(H):
            mass = received[l, h]
            total = mass.sum().clamp(min=1e-9)
            lift = torch.where(common, (mass / total) / share.clamp(min=1e-9), torch.zeros_like(share))
            top = torch.topk(lift, 5)
            heads[f"{l}.{h}"] = {
                "rest": round(rest[l, h].item(), 3),
                "self": round(selfa[l, h].item(), 3),
                "prev": round(prev[l, h].item(), 3),
                "dist": round(dist[l, h].item(), 2),
                "punct": round(punct[l, h].item(), 3),
                "copying": round(copying[l, h].item(), 3),
                "drawn": [[tok.decode([int(i)]), round(v.item(), 1)] for i, v in zip(top.indices, top.values) if v > 3],
            }

    examples = {}
    for l in range(L):
        for h in range(H):
            chosen = sorted(best[l * H + h], key=lambda x: -x[0])[:EXAMPLES]
            items = []
            for w, si, i, row in chosen:
                toks = all_tokens[si]
                start = max(1, i - WINDOW)
                span = list(range(start, i + 1))
                # Long-range heads often look further back than the window:
                # include the word they look at most, with a gap (None) before the window.
                focus = int(row[1:].argmax()) + 1
                if focus < start:
                    near = list(range(max(1, focus - 3), min(start, focus + 4)))
                    span = near + ([None] if near[-1] + 1 < start else []) + span
                items.append({
                    "pieces": [None if t is None else tok.decode([int(toks[t])]) for t in span],
                    "attn": [None if t is None else round(float(row[t]), 2) for t in span],
                    "start": round(float(row[0]), 2),
                    "next": tok.decode([int(toks[i + 1])]) if i + 1 < len(toks) else None,
                })
            examples[f"{l}.{h}"] = items
    ex = root / "public" / "head-examples.json"
    ex.write_text(json.dumps({"source": "WikiText-103 (CC BY-SA 4.0), validation set", "heads": examples}, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {ex} ({ex.stat().st_size // 1024} KB)")

    out = root / "src" / "app" / "head-atlas.json"
    out.write_text(json.dumps({"source": "WikiText-103 validation, first 300 paragraphs, up to 128 tokens each", "heads": heads}, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(heads)} heads, {n} query positions)")
    for hid in ["4.11", "5.5", "9.9", "10.7", "0.1", "11.0", "1.4"]:
        print(hid, heads[hid])


if __name__ == "__main__":
    main()
