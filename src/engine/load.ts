// Downloads the model once, checks it, and keeps it in the browser's cache.

import { f16ToF32, Weights, type Manifest } from "./model.ts";
import { Tokenizer } from "./tokenizer.ts";

export interface LoadProgress {
  loaded: number;
  total: number;
  fromCache: boolean;
}

const CACHE = "featurama-model-v1";

async function openCache() {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // private windows and some embedded browsers have no Cache API
  }
}

async function sha256(bytes: Uint8Array<ArrayBuffer>) {
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(hash, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function loadModel(base: string, onProgress?: (p: LoadProgress) => void) {
  const manifest: Manifest = await (await fetch(new URL("manifest.json", base))).json();
  const cache = await openCache();
  const data = new Uint8Array(manifest.bytes);
  let loaded = 0;
  let fromCache = true;

  for (const shard of manifest.shards) {
    const url = new URL(shard.file, base).href;
    const key = `${url}?sha256=${shard.sha256}`;
    const start = loaded;
    let res = await cache?.match(key);
    const hit = !!res;
    if (!res) {
      fromCache = false;
      res = await fetch(url);
      if (!res.ok || !res.body) throw new Error(`Couldn't download ${shard.file} (${res.status}).`);
    }
    const reader = res.body!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      data.set(value, loaded);
      loaded += value.length;
      onProgress?.({ loaded, total: manifest.bytes, fromCache });
    }
    const bytes = data.subarray(start, loaded);
    if (bytes.length !== shard.bytes || (await sha256(bytes)) !== shard.sha256) {
      if (hit) await cache?.delete(key);
      throw new Error(`${shard.file} arrived damaged. Reload to try again.`);
    }
    if (!hit) await cache?.put(key, new Response(bytes.slice()));
  }

  const [vocab, merges] = await Promise.all([
    fetch(new URL("vocab.json", base)).then((r) => r.json()),
    fetch(new URL("merges.txt", base)).then((r) => r.text()),
  ]);
  return {
    weights: new Weights(manifest, new Uint16Array(data.buffer)),
    tokenizer: new Tokenizer(vocab, merges),
    fromCache,
  };
}

export interface SaeManifest {
  layer: number;
  hook: "resid_pre";
  d_in: number;
  d_sae: number;
  rows_per_shard: number;
  decoder_shards: number;
  files: Record<string, { bytes: number; sha256: string }>;
}

// One checked file, from the cache when we have it.
async function fetchChecked(base: string, name: string, info: { bytes: number; sha256: string }) {
  const cache = await openCache();
  const url = new URL(name, base).href;
  const key = `${url}?sha256=${info.sha256}`;
  const hit = await cache?.match(key);
  const res = hit ?? (await fetch(url));
  if (!res.ok) throw new Error(`Couldn't download ${name} (${res.status}).`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (bytes.length !== info.bytes || (await sha256(bytes)) !== info.sha256) {
    if (hit) await cache?.delete(key);
    throw new Error(`${name} arrived damaged. Reload to try again.`);
  }
  if (!hit) await cache?.put(key, new Response(bytes.slice()));
  return new Uint16Array(bytes.buffer);
}

// The sparse autoencoder's encoder; its decoder rows come later, one shard at a time.
export async function loadSae(base: string) {
  const manifest: SaeManifest = await (await fetch(new URL("manifest.json", base))).json();
  const [encoder, bias] = await Promise.all([
    fetchChecked(base, "encoder.bin", manifest.files["encoder.bin"]),
    fetchChecked(base, "bias.bin", manifest.files["bias.bin"]),
  ]);
  const shards = new Map<number, Promise<Uint16Array>>();
  // The direction a feature writes into the residual stream (its decoder row).
  const direction = async (feature: number) => {
    const n = Math.floor(feature / manifest.rows_per_shard);
    const name = `decoder-${String(n).padStart(2, "0")}.bin`;
    if (!shards.has(n)) shards.set(n, fetchChecked(base, name, manifest.files[name]).catch((e) => (shards.delete(n), Promise.reject(e))));
    const rows = await shards.get(n)!;
    const r = feature % manifest.rows_per_shard;
    return f16ToF32(rows.subarray(r * manifest.d_in, (r + 1) * manifest.d_in));
  };
  return { manifest, encoder, bias, direction };
}
