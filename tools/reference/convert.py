"""Convert GPT-2 small from Hugging Face into the browser's weight format.

Writes public/model/gpt2/: manifest.json, weights-N.bin shards (fp16, little
endian, every tensor in its Hugging Face layout), vocab.json and merges.txt.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SHARD_BYTES = 64 * 1024 * 1024

root = Path(__file__).resolve().parents[2]
out = root / "public" / "model" / "gpt2"


def tensor_names(n_layers):
    names = ["wte.weight", "wpe.weight"]
    for i in range(n_layers):
        for part in [
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]:
            names.append(f"h.{i}.{part}")
    return names + ["ln_f.weight", "ln_f.bias"]


def main():
    config = json.loads(Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text())
    weights = load_file(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
    out.mkdir(parents=True, exist_ok=True)

    blob = bytearray()
    tensors = []
    worst = 0.0
    for name in tensor_names(config["n_layer"]):
        w = weights[name].astype(np.float32)
        h = w.astype(np.float16)
        if not np.isfinite(h).all():
            sys.exit(f"{name} overflows fp16")
        worst = max(worst, float(np.abs(h.astype(np.float32) - w).max()))
        tensors.append({"name": name, "shape": list(w.shape), "offset": len(blob)})
        blob += h.tobytes()

    shards = []
    for i in range(0, len(blob), SHARD_BYTES):
        chunk = bytes(blob[i:i + SHARD_BYTES])
        path = out / f"weights-{len(shards)}.bin"
        path.write_bytes(chunk)
        shards.append({"file": path.name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})

    manifest = {
        "model": "gpt2",
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "dtype": "f16",
        "config": {
            "n_layers": config["n_layer"],
            "n_heads": config["n_head"],
            "d_model": config["n_embd"],
            "d_mlp": 4 * config["n_embd"],
            "n_ctx": config["n_positions"],
            "d_vocab": config["vocab_size"],
            "ln_eps": config["layer_norm_epsilon"],
        },
        "bytes": len(blob),
        "shards": shards,
        "tensors": tensors,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ["vocab.json", "merges.txt"]:
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION), out / f)
    print(f"{len(blob) / 1e6:.1f} MB in {len(shards)} shards; worst fp16 rounding error {worst:.2e}")


if __name__ == "__main__":
    main()
