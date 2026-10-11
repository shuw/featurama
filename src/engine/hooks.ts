// What a run can read and change. Names follow TransformerLens, so results can
// be checked against it and learners who move on to it find the same words.

// resid_pre of layer l is the stream entering layer l; resid_mid is after its
// attention; resid_post is after its MLP (the same as resid_pre of l + 1).
export type ResidHook = "resid_pre" | "resid_mid" | "resid_post";

export type Intervention =
  // Replace or add to the residual stream. value is one vector of d_model
  // (applied at `position`, or at every position for "all"), or T * d_model
  // to set the whole activation.
  | { kind: "resid"; hook: ResidHook; layer: number; position: number | "all"; mode: "set" | "add"; value: Float32Array }
  // Scale one attention head's output: 0 switches it off.
  | { kind: "head"; layer: number; head: number; scale: number };

export interface RunRequest {
  tokens: number[];
  interventions?: Intervention[];
  read?: { resid?: boolean; residMid?: boolean; pattern?: boolean };
  // The logit lens: final LayerNorm and unembedding applied to the residual
  // stream before layer 0 and after every layer, at the chosen positions.
  // The row for the last layer is the model's real prediction.
  lens?: { positions?: number[]; topK?: number; targets?: number[] };
}

export interface Lens {
  positions: number[];
  topK: number;
  targets: number[];
  // Rows are layer-major: row = layer * positions.length + positionIndex,
  // layer 0 being the embedding and layer n_layers the final output.
  ids: Uint32Array; // [rows, topK]
  logprobs: Float32Array; // [rows, topK]
  targetLogprobs: Float32Array; // [rows, targets.length]
}

export interface Trace {
  tokens: number[];
  resid?: Float32Array; // [n_layers + 1, T, d_model]
  residMid?: Float32Array; // [n_layers, T, d_model]
  pattern?: Float32Array; // [n_layers, n_heads, T, T]
  lens?: Lens;
  ms: number; // wall time for the run, including reading results back
}
