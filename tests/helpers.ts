import { join } from "node:path";
import { Weights, type Manifest } from "../src/engine/model.ts";
import { Tokenizer } from "../src/engine/tokenizer.ts";

const root = join(import.meta.dir, "..");
const modelDir = join(root, "public/model/gpt2");

type Arr = { dtype: "float32" | "int32"; shape: number[]; b64: string };

function decode(a: Arr) {
  const bytes = Buffer.from(a.b64, "base64");
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return a.dtype === "int32" ? new Int32Array(buf) : new Float32Array(buf);
}

export interface Fixture {
  name: string;
  text: string | null;
  tokens: number[];
  shape: Record<string, number[]>;
  resid: Float32Array;
  resid_mid: Float32Array;
  pattern: Float32Array;
  logits_last: Float32Array;
  top_ids: Int32Array;
  lens_ids: Int32Array;
  lens_logprobs: Float32Array;
}

export async function fixture(variant: "fp16" | "fp32", name: string): Promise<Fixture> {
  const raw = await Bun.file(join(root, "fixtures", variant, `${name}.json`)).json();
  const out: any = { name: raw.name, text: raw.text, shape: {} };
  for (const [k, v] of Object.entries(raw)) {
    if (v && typeof v === "object" && "b64" in (v as object)) {
      out[k] = decode(v as Arr);
      out.shape[k] = (v as Arr).shape;
    }
  }
  out.tokens = Array.from(out.tokens);
  return out;
}

export const FIXTURES = ["eiffel", "ioi", "years", "fair", "repeat"];

export async function weights() {
  const manifest: Manifest = await Bun.file(join(modelDir, "manifest.json")).json();
  const data = new Uint16Array(manifest.bytes / 2);
  let offset = 0;
  for (const s of manifest.shards) {
    const bytes = new Uint8Array(await Bun.file(join(modelDir, s.file)).arrayBuffer());
    new Uint8Array(data.buffer).set(bytes, offset);
    offset += bytes.length;
  }
  return new Weights(manifest, data);
}

export async function tokenizer() {
  const vocab = await Bun.file(join(modelDir, "vocab.json")).json();
  const merges = await Bun.file(join(modelDir, "merges.txt")).text();
  return new Tokenizer(vocab, merges);
}

export async function tokenizerCases(): Promise<{ text: string; ids: number[] }[]> {
  return Bun.file(join(root, "fixtures/tokenizer.json")).json();
}

// Worst relative error over rows: |a - b| / |b| for each row of length n.
export function rowError(a: Float32Array, b: Float32Array, n: number) {
  let worst = 0;
  for (let r = 0; r < b.length / n; r++) {
    let diff = 0;
    let norm = 0;
    for (let i = r * n; i < (r + 1) * n; i++) {
      diff += (a[i] - b[i]) ** 2;
      norm += b[i] ** 2;
    }
    worst = Math.max(worst, Math.sqrt(diff / Math.max(norm, 1e-12)));
  }
  return worst;
}

export function maxAbs(a: Float32Array, b: Float32Array) {
  let worst = 0;
  for (let i = 0; i < b.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

export function concat(arrays: Float32Array[]) {
  const out = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrays) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}
