"""Convert Joseph Bloom's layer-7 sparse autoencoder for GPT-2 small (MIT)
into the browser's format.

The autoencoder was trained on TransformerLens's residual stream, which has
each token's mean across the 768 dimensions removed. The browser runs GPT-2's
original weights, so that centring and the decoder bias are folded into the
encoder here: features = ReLU(x @ W_enc' + b_enc') on the raw stream.

Writes public/model/sae-l7/: manifest.json, encoder.bin ([768, 24576] fp16),
bias.bin ([24576] fp16) and decoder-NN.bin (256 feature directions each,
[256, 768] fp16), fetched only when a feature is steered.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
REVISION = "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9"
HOOK = "blocks.7.hook_resid_pre"
ROWS_PER_SHARD = 256

root = Path(__file__).resolve().parents[2]
out = root / "public" / "model" / "sae-l7"


def load():
    path = hf_hub_download(REPO, f"{HOOK}/sae_weights.safetensors", revision=REVISION)
    return load_file(path)


def folded(w):
    W_enc = w["W_enc"].astype(np.float64)
    b = w["b_enc"].astype(np.float64) - w["b_dec"].astype(np.float64) @ W_enc
    return W_enc - W_enc.mean(axis=0, keepdims=True), b


def main():
    w = load()
    W, b = folded(w)
    d_in, d_sae = W.shape
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    files = {}

    def write(name, array):
        data = array.astype("<f2").tobytes()
        (out / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    write("encoder.bin", W)
    write("bias.bin", b)
    dec = w["W_dec"]
    shards = 0
    for i in range(0, d_sae, ROWS_PER_SHARD):
        write(f"decoder-{shards:02d}.bin", dec[i : i + ROWS_PER_SHARD])
        shards += 1
    manifest = {
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}/{HOOK}",
        "license": "MIT",
        "layer": 7,
        "hook": "resid_pre",
        "d_in": int(d_in),
        "d_sae": int(d_sae),
        "rows_per_shard": ROWS_PER_SHARD,
        "decoder_shards": shards,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out} ({d_sae} features, {shards} decoder shards)")


if __name__ == "__main__":
    main()
