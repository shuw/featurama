import { expect, test } from "bun:test";
import { runCpu, unembedCpu } from "../src/engine/cpu.ts";
import { concat, fixture, rowError, weights } from "./helpers.ts";

// The CPU pass computes in float64/float32 from the fp16 weights, so it should
// agree closely with TransformerLens run on the same rounded weights.
for (const name of ["eiffel", "ioi"]) {
  test(`CPU forward pass matches TransformerLens on "${name}"`, async () => {
    const w = await weights();
    const ref = await fixture("fp16", name);
    const d = w.config.d_model;
    const T = ref.tokens.length;
    const trace = runCpu(w, ref.tokens);

    expect(rowError(concat(trace.resid), ref.resid, d)).toBeLessThan(1e-4);
    expect(rowError(concat(trace.residMid), ref.resid_mid, d)).toBeLessThan(1e-4);
    expect(rowError(concat(trace.pattern), ref.pattern, T)).toBeLessThan(1e-4);

    const last = trace.resid[w.config.n_layers].subarray((T - 1) * d, T * d);
    const logits = unembedCpu(w, last);
    expect(rowError(logits, ref.logits_last, logits.length)).toBeLessThan(1e-4);
    const top = logits.indexOf(Math.max(...logits));
    expect(top).toBe(ref.top_ids[(T - 1) * 10]);
  }, 120_000);
}
