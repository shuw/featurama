// Where every weight sits in the tower. One unit is one weight. Floors are
// horizontal: the embedding table on the ground, one floor per layer, and the
// same table again on the roof, where it turns the result into next-token
// scores. On each floor a matrix lies flat, its inputs along z (rows) and its
// outputs along x (columns), in the order a forward pass uses them.

export const FLOOR = 1800; // height between floors
export const STRIP = 24; // vectors are one weight wide; drawn this wide so they show from afar
const GAP = 120;
const FOLD = { rows: 3200, gap: 48, count: 16 }; // the 50,257-row table, wrapped into 16 columns

export type Forward =
  | { mode: "none" }
  | { mode: "row" | "col"; act: string } // contribution: weight × the input along rows (or columns)
  | { mode: "pick"; pick: "token" | "position" }; // only one row is used: the token's (or position's)

export interface Tile {
  id: string;
  floor: number;
  part: string; // for the readout, e.g. "Layer 9 · attention · queries"
  name: string; // short label
  x: number; // world position of the corner (smallest x and z)
  z: number;
  w: number; // world size along x and z
  d: number;
  rows: number; // in the tensor (or vector)
  cols: number;
  source: { kind: "weight"; tensor: string; stride: number; colStart: number } | { kind: "act"; act: string };
  fold?: typeof FOLD;
  forward: Forward;
  heads?: boolean; // mark every 64 columns: one head each
}

export interface Floor {
  index: number;
  label: string;
  y: number;
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

export function buildTower(L = 12, d = 768, dMlp = 3072, vocab = 50257, nCtx = 1024) {
  const tiles: Tile[] = [];
  const floors: Floor[] = [];

  const place = (floor: number, label: string, items: Omit<Tile, "floor" | "x" | "z">[][], z0: number) => {
    // Groups go left to right; inside a group, items sit side by side.
    // A bias (one row) sits just in front of the matrix before it.
    let x = 0;
    let last = 0;
    const placed: Tile[] = [];
    for (const group of items) {
      for (const it of group) {
        const isBias = it.rows === 1 && it.source.kind === "weight";
        placed.push({ ...it, floor, x: isBias ? last : x, z: z0 + (isBias ? -STRIP - 16 : 0) });
        if (!isBias) {
          last = x;
          x += it.w + 16;
        }
      }
      x += GAP;
    }
    // Centre the floor on x = 0.
    const width = x - GAP;
    for (const t of placed) t.x -= width / 2;
    tiles.push(...placed);
    const zs = placed.map((t) => [t.z, t.z + t.d]).flat();
    floors.push({ index: floor, label, y: floor * FLOOR, x0: -width / 2 - 200, x1: width / 2 + 200, z0: Math.min(...zs) - 200, z1: Math.max(...zs) + 200 });
  };

  const vec = (id: string, part: string, name: string, act: string, n: number): Omit<Tile, "floor" | "x" | "z"> => ({
    id, part, name, w: STRIP, d: n, rows: n, cols: 1, source: { kind: "act", act }, forward: { mode: "none" },
  });
  const gain = (id: string, part: string, name: string, tensor: string, n: number): Omit<Tile, "floor" | "x" | "z"> => ({
    id, part, name, w: STRIP, d: n, rows: n, cols: 1, source: { kind: "weight", tensor, stride: 1, colStart: 0 }, forward: { mode: "none" },
  });
  const mat = (id: string, part: string, name: string, tensor: string, rows: number, cols: number, stride: number, colStart: number, act: string, heads = false): Omit<Tile, "floor" | "x" | "z"> => ({
    id, part, name, w: cols, d: rows, rows, cols, source: { kind: "weight", tensor, stride, colStart }, forward: { mode: "row", act }, heads,
  });
  const bias = (id: string, part: string, tensor: string, cols: number, stride: number, colStart: number): Omit<Tile, "floor" | "x" | "z"> => ({
    id, part, name: "bias", w: cols, d: STRIP, rows: 1, cols, source: { kind: "weight", tensor, stride, colStart }, forward: { mode: "none" },
  });

  const z0 = -1600;
  const table = (id: string, part: string, forward: Forward): Omit<Tile, "floor" | "x" | "z"> => ({
    id, part, name: "token table", w: FOLD.count * (d + FOLD.gap) - FOLD.gap, d: FOLD.rows, rows: vocab, cols: d,
    source: { kind: "weight", tensor: "wte.weight", stride: d, colStart: 0 }, fold: FOLD, forward,
  });

  place(0, "Embedding", [
    [table("wte", "Embedding · token table: one row of 768 numbers per token", { mode: "pick", pick: "token" })],
    [{ id: "wpe", part: "Embedding · position table: one row per position", name: "position table", w: d, d: nCtx, rows: nCtx, cols: d,
      source: { kind: "weight", tensor: "wpe.weight", stride: d, colStart: 0 }, forward: { mode: "pick", pick: "position" } }],
  ], z0);

  for (let l = 0; l < L; l++) {
    const p = `h.${l}`;
    const at = `Layer ${l}`;
    place(l + 1, at, [
      [vec(`r${l}`, `${at} · running summary coming in`, "stream", `resid${l}`, d)],
      [gain(`${l}g1`, `${at} · LayerNorm gain`, "norm", `${p}.ln_1.weight`, d), gain(`${l}b1`, `${at} · LayerNorm bias`, "norm", `${p}.ln_1.bias`, d), vec(`${l}x1`, `${at} · attention input`, "input", `ln1_${l}`, d)],
      [
        mat(`${l}q`, `${at} · attention · queries (12 heads × 64)`, "queries", `${p}.attn.c_attn.weight`, d, d, 3 * d, 0, `ln1_${l}`, true),
        bias(`${l}qb`, `${at} · attention · query bias`, `${p}.attn.c_attn.bias`, d, 3 * d, 0),
        mat(`${l}k`, `${at} · attention · keys (12 heads × 64)`, "keys", `${p}.attn.c_attn.weight`, d, d, 3 * d, d, `ln1_${l}`, true),
        bias(`${l}kb`, `${at} · attention · key bias`, `${p}.attn.c_attn.bias`, d, 3 * d, d),
        mat(`${l}v`, `${at} · attention · values (12 heads × 64)`, "values", `${p}.attn.c_attn.weight`, d, d, 3 * d, 2 * d, `ln1_${l}`, true),
        bias(`${l}vb`, `${at} · attention · value bias`, `${p}.attn.c_attn.bias`, d, 3 * d, 2 * d),
      ],
      [
        vec(`${l}z`, `${at} · the heads' blended values`, "blend", `z${l}`, d),
        mat(`${l}o`, `${at} · attention · output`, "output", `${p}.attn.c_proj.weight`, d, d, d, 0, `z${l}`),
        bias(`${l}ob`, `${at} · attention · output bias`, `${p}.attn.c_proj.bias`, d, d, 0),
      ],
      [gain(`${l}g2`, `${at} · LayerNorm gain`, "norm", `${p}.ln_2.weight`, d), gain(`${l}b2`, `${at} · LayerNorm bias`, "norm", `${p}.ln_2.bias`, d), vec(`${l}x2`, `${at} · MLP input`, "input", `ln2_${l}`, d)],
      [
        mat(`${l}fc`, `${at} · MLP · into 3,072 neurons`, "MLP in", `${p}.mlp.c_fc.weight`, d, dMlp, dMlp, 0, `ln2_${l}`),
        bias(`${l}fcb`, `${at} · MLP · neuron bias`, `${p}.mlp.c_fc.bias`, dMlp, dMlp, 0),
      ],
      [
        vec(`${l}n`, `${at} · the 3,072 neurons`, "neurons", `post${l}`, dMlp),
        mat(`${l}pr`, `${at} · MLP · out of the neurons`, "MLP out", `${p}.mlp.c_proj.weight`, dMlp, d, d, 0, `post${l}`),
        bias(`${l}prb`, `${at} · MLP · output bias`, `${p}.mlp.c_proj.bias`, d, d, 0),
      ],
    ], z0);
  }

  place(L + 1, "Unembedding", [
    [vec("rf", "Final · running summary", "stream", `resid${L}`, d)],
    [gain("gf", "Final LayerNorm gain", "norm", "ln_f.weight", d), gain("bf", "Final LayerNorm bias", "norm", "ln_f.bias", d), vec("xf", "Final · normalised summary", "input", "lnf", d)],
    [table("wte2", "Unembedding · the same token table, scoring every token", { mode: "col", act: "lnf" })],
  ], z0);

  return { tiles, floors };
}

// A display row and column on a tile to the tensor's row and column, or null for a gap.
export function toTensor(t: Tile, displayRow: number, displayCol: number): [number, number] | null {
  if (!t.fold) return [Math.floor((displayRow / t.d) * t.rows), Math.floor((displayCol / t.w) * t.cols)];
  const span = t.cols + t.fold.gap;
  const fold = Math.floor(displayCol / span);
  const c = Math.floor(displayCol - fold * span);
  const r = fold * t.fold.rows + Math.floor(displayRow);
  if (c >= t.cols || r >= t.rows || fold >= t.fold.count) return null;
  return [r, c];
}
