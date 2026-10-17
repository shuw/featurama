// The forward pass as a sequence of steps, in the order GPT-2 computes them,
// each lighting the tiles it uses. Captions are filled in from the real run.

export interface Stage {
  tiles: string[]; // tile ids lit by this step
  floor: number;
  title: string;
  kind: "token" | "position" | "stream" | "norm" | "qkv" | "mix" | "out" | "fc" | "fire" | "proj" | "final" | "scores";
  layer: number; // -1 for the embedding and roof
}

export function buildStages(L: number): Stage[] {
  const out: Stage[] = [
    { tiles: ["wte"], floor: 0, title: "Look up the token's row in the token table", kind: "token", layer: -1 },
    { tiles: ["wpe"], floor: 0, title: "Add its position's row from the position table", kind: "position", layer: -1 },
  ];
  for (let l = 0; l < L; l++) {
    const f = l + 1;
    const at = `Layer ${l}`;
    out.push(
      { tiles: [`r${l}`], floor: f, title: `${at}: the running summary arrives`, kind: "stream", layer: l },
      { tiles: [`${l}g1`, `${l}b1`, `${l}x1`], floor: f, title: `${at}: LayerNorm rescales it`, kind: "norm", layer: l },
      { tiles: [`${l}q`, `${l}qb`, `${l}k`, `${l}kb`, `${l}v`, `${l}vb`], floor: f, title: `${at}: queries, keys and values for 12 heads`, kind: "qkv", layer: l },
      { tiles: [`${l}z`], floor: f, title: `${at}: each head blends values from earlier tokens`, kind: "mix", layer: l },
      { tiles: [`${l}o`, `${l}ob`], floor: f, title: `${at}: the heads' output joins the running summary`, kind: "out", layer: l },
      { tiles: [`${l}g2`, `${l}b2`, `${l}x2`], floor: f, title: `${at}: LayerNorm again`, kind: "norm", layer: l },
      { tiles: [`${l}fc`, `${l}fcb`], floor: f, title: `${at}: into the MLP's 3,072 neurons`, kind: "fc", layer: l },
      { tiles: [`${l}n`], floor: f, title: `${at}: the neurons fire`, kind: "fire", layer: l },
      { tiles: [`${l}pr`, `${l}prb`], floor: f, title: `${at}: the MLP's output joins the running summary`, kind: "proj", layer: l },
    );
  }
  out.push(
    { tiles: ["rf"], floor: L + 1, title: "The finished summary reaches the roof", kind: "stream", layer: L },
    { tiles: ["gf", "bf", "xf"], floor: L + 1, title: "A final LayerNorm", kind: "final", layer: L },
    { tiles: ["wte2"], floor: L + 1, title: "Score every token with the same token table", kind: "scores", layer: L },
  );
  return out;
}
