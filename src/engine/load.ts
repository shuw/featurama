// Downloads the model once, checks it, and keeps it in the browser's cache.

import { Weights, type Manifest } from "./model.ts";
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
