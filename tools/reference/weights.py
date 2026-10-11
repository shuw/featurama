"""Pictures of GPT-2 small's weights that are too slow to compute in a browser.

- public/zoom-data/wiring.json: how strongly each head's output can feed each later
  head's query, key or value (composition scores, Elhage et al. 2021), with
  the typical score as a baseline.
- public/zoom-data/token-map.bin: every token's place on a 2D map of the token
  embedding (UMAP with cosine distance), as int16 x, y pairs.

The output is committed; CI does not rerun it. Needs umap-learn, which is not
in requirements.txt.
"""

import json
from pathlib import Path

import numpy as np
import torch
import umap
from transformer_lens import HookedTransformer

root = Path(__file__).resolve().parents[2]
TOP = 6  # strongest sources kept per head and kind


def main():
    torch.set_grad_enabled(False)
    out = root / "public" / "zoom-data"
    out.mkdir(parents=True, exist_ok=True)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")  # LayerNorm folded in, as composition scores assume
    L, H = model.cfg.n_layers, model.cfg.n_heads

    wiring = {"source": "Composition scores (Elhage et al. 2021) from GPT-2 small's weights", "kinds": {}}
    for kind in "QKV":
        s = model.all_composition_scores(kind)  # [L, H, L, H]: earlier head -> later head
        s = s.reshape(L * H, L * H).numpy()
        earlier = np.arange(L * H)[:, None] // H < np.arange(L * H)[None, :] // H
        valid = s[earlier]
        heads = {}
        for j in range(H, L * H):
            col = np.where(earlier[:, j], s[:, j], -1)
            top = np.argsort(-col)[:TOP]
            heads[f"{j // H}.{j % H}"] = [[f"{i // H}.{i % H}", round(float(col[i]), 4)] for i in top]
        wiring["kinds"][kind] = {"median": round(float(np.median(valid)), 4), "p99": round(float(np.percentile(valid, 99)), 4), "heads": heads}
        print(kind, wiring["kinds"][kind]["median"], wiring["kinds"][kind]["p99"], heads["5.5"][:3])
    (out / "wiring.json").write_text(json.dumps(wiring, separators=(",", ":")))

    E = model.W_E.numpy()
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=1939, low_memory=True).fit_transform(E)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    q = np.round(xy * 32767).astype(np.int16)
    (out / "token-map.bin").write_bytes(q.tobytes())
    print(f"wrote {out}: {len(q)} tokens on the map")


if __name__ == "__main__":
    main()
