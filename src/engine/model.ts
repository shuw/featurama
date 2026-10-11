// The shape of GPT-2 small and of the weight files the converter writes.

export interface ModelConfig {
  n_layers: number;
  n_heads: number;
  d_model: number;
  d_mlp: number;
  n_ctx: number;
  d_vocab: number;
  ln_eps: number;
}

export interface Manifest {
  model: string;
  source: string;
  dtype: "f16";
  config: ModelConfig;
  bytes: number;
  shards: { file: string; bytes: number; sha256: string }[];
  tensors: { name: string; shape: number[]; offset: number }[];
}

// <|endoftext|>. Featurama starts every prompt with it, as TransformerLens does.
export const BOS = 50256;

const F16 = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) table[h] = sign * 2 ** -14 * (frac / 1024);
    else if (exp === 31) table[h] = frac ? NaN : sign * Infinity;
    else table[h] = sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return table;
})();

export function f16ToF32(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = F16[h[i]];
  return out;
}

// All weights as one fp16 array, with views by tensor name.
export class Weights {
  private index: Map<string, { shape: number[]; offset: number; length: number }>;
  private cache = new Map<string, Float32Array>();

  constructor(readonly manifest: Manifest, readonly data: Uint16Array) {
    this.index = new Map(
      manifest.tensors.map((t) => [t.name, { shape: t.shape, offset: t.offset / 2, length: t.shape.reduce((a, b) => a * b, 1) }]),
    );
  }

  get config() {
    return this.manifest.config;
  }

  names() {
    return [...this.index.keys()];
  }

  shape(name: string) {
    return this.entry(name).shape;
  }

  raw(name: string) {
    const t = this.entry(name);
    return this.data.subarray(t.offset, t.offset + t.length);
  }

  f32(name: string) {
    let a = this.cache.get(name);
    if (!a) this.cache.set(name, (a = f16ToF32(this.raw(name))));
    return a;
  }

  private entry(name: string) {
    const t = this.index.get(name);
    if (!t) throw new Error(`No tensor named ${name}`);
    return t;
  }
}
