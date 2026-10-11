// Checks the WebGPU engine against TransformerLens fixtures and times it.
// scripts/gpu-test.mjs opens this page in Chrome and reads window.results.

import { GpuEngine } from "../../src/engine/gpu/engine.ts";
import type { Intervention } from "../../src/engine/hooks.ts";
import { loadModel, loadSae } from "../../src/engine/load.ts";

type Arr = { dtype: "float32" | "int32"; shape: number[]; b64: string };
const decode = (a: Arr) => {
  const bytes = Uint8Array.from(atob(a.b64), (c) => c.charCodeAt(0));
  return a.dtype === "int32" ? new Int32Array(bytes.buffer) : new Float32Array(bytes.buffer);
};

const logEl = document.getElementById("log")!;
const lines: string[] = [];
const log = (s: string) => {
  lines.push(s);
  logEl.textContent = lines.join("\n");
};

function rowError(a: Float32Array, b: Float32Array, n: number) {
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

const median = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

async function main() {
  const results: any = { cases: [], interventions: [], perf: [], failures: [] };
  const fail = (s: string) => {
    results.failures.push(s);
    log("FAIL " + s);
  };

  let t = performance.now();
  const { weights, fromCache } = await loadModel(new URL("/model/gpt2/", location.href).href);
  results.loadMs = performance.now() - t;
  results.fromCache = fromCache;
  t = performance.now();
  const engine = await GpuEngine.create(weights);
  results.createMs = performance.now() - t;
  results.adapter = engine.adapterInfo;
  log(`model loaded in ${results.loadMs.toFixed(0)} ms (${fromCache ? "cache" : "network"}), GPU ready in ${results.createMs.toFixed(0)} ms on ${engine.adapterInfo || "unknown GPU"}`);

  const { n_layers: L, d_model: d } = engine.config;
  for (const variant of ["fp16", "fp32"] as const) {
    for (const name of ["eiffel", "ioi", "years", "fair", "repeat"]) {
      const raw = await (await fetch(`/fixtures/${variant}/${name}.json`)).json();
      const tokens = Array.from(decode(raw.tokens));
      const T = tokens.length;
      const trace = await engine.run({
        tokens,
        read: { resid: true, residMid: true, pattern: true, z: true },
        lens: { positions: tokens.map((_, i) => i), topK: 5 },
      });
      const lensIds = decode(raw.lens_ids) as Int32Array;
      const lensLp = decode(raw.lens_logprobs) as Float32Array;
      const rows = (L + 1) * T;
      let top1 = 0;
      let top5 = 0;
      let lpErr = 0;
      for (let r = 0; r < rows; r++) {
        // fixture lens is [layer, position, k]; engine rows are layer-major too.
        if (trace.lens!.ids[r * 5] === lensIds[r * 5]) top1++;
        let same = true;
        for (let k = 0; k < 5; k++) {
          if (trace.lens!.ids[r * 5 + k] !== lensIds[r * 5 + k]) same = false;
          lpErr = Math.max(lpErr, Math.abs(trace.lens!.logprobs[r * 5 + k] - lensLp[r * 5 + k]));
        }
        if (same) top5++;
      }
      const c = {
        variant, name, T,
        resid: rowError(trace.resid!, decode(raw.resid) as Float32Array, d),
        residMid: rowError(trace.residMid!, decode(raw.resid_mid) as Float32Array, d),
        pattern: rowError(trace.pattern!, decode(raw.pattern) as Float32Array, T),
        z: rowError(trace.z!, decode(raw.z) as Float32Array, d / engine.config.n_heads),
        lensTop1: top1 / rows,
        lensTop5: top5 / rows,
        lensLogprobErr: lpErr,
        finalTop1: trace.lens!.ids[(L * T + T - 1) * 5] === lensIds[(L * T + T - 1) * 5],
        ms: trace.ms,
      };
      results.cases.push(c);
      log(`${variant} ${name.padEnd(7)} T=${String(T).padStart(2)}  resid ${c.resid.toExponential(1)}  mid ${c.residMid.toExponential(1)}  pattern ${c.pattern.toExponential(1)}  z ${c.z.toExponential(1)}  lens top-1 ${(c.lensTop1 * 100).toFixed(1)}%  top-5 ${(c.lensTop5 * 100).toFixed(1)}%  logprob err ${c.lensLogprobErr.toExponential(1)}`);
      if (variant === "fp16") {
        if (c.resid > 1e-3 || c.residMid > 1e-3 || c.pattern > 1e-3 || c.z > 1e-3) fail(`${name}: activations differ from TransformerLens beyond 1e-3`);
        if (c.lensTop1 < 0.99 || c.lensLogprobErr > 1e-2) fail(`${name}: logit lens differs from TransformerLens`);
      } else if (!c.finalTop1) fail(`${name}: final prediction differs from the full-precision model`);
    }
  }

  const ivs = await (await fetch("/fixtures/fp16/interventions.json")).json();
  for (const iv of ivs) {
    const interventions: Intervention[] = iv.interventions.map((x: any) => (x.value ? { ...x, value: decode(x.value) } : x));
    const trace = await engine.run({ tokens: iv.tokens, interventions, lens: { positions: [iv.tokens.length - 1], topK: 1 } });
    const ref = decode(iv.logits_last) as Float32Array;
    const refTop = ref.indexOf(Math.max(...ref));
    const lp = trace.lens!.logprobs[L];
    const refLse = Math.log(ref.reduce((s, x) => s + Math.exp(x - ref[refTop]), 0)) + ref[refTop];
    const c = { name: iv.name, top: trace.lens!.ids[L] === refTop, logprobErr: Math.abs(lp - (ref[refTop] - refLse)) };
    results.interventions.push(c);
    log(`intervention: ${iv.name.padEnd(22)} top token ${c.top ? "matches" : "DIFFERS"}, logprob err ${c.logprobErr.toExponential(1)}`);
    if (!c.top || c.logprobErr > 1e-2) fail(`intervention "${iv.name}" differs from TransformerLens`);
  }

  // The layer-7 sparse autoencoder, against TransformerLens with the original encoder.
  const fx = await (await fetch("/fixtures/sae.json")).json();
  const sae = await loadSae(new URL("/model/sae-l7/", location.href).href);
  engine.attachSae(sae.manifest.layer, sae.manifest.d_sae, sae.encoder, sae.bias);
  const ft = await engine.run({ tokens: fx.tokens, read: { features: true } });
  const F = sae.manifest.d_sae;
  let saeErr = 0;
  let saeTop = 0;
  for (let t = 1; t < fx.tokens.length; t++) {
    const row = ft.features!.subarray(t * F, (t + 1) * F);
    for (const [i, v] of fx.top[t] as [number, number][]) saeErr = Math.max(saeErr, Math.abs(row[i] - v) / Math.max(1, v));
    let best = 0;
    for (let i = 1; i < F; i++) if (row[i] > row[best]) best = i;
    if (best === fx.top[t][0][0]) saeTop++;
  }
  results.sae = { relErr: saeErr, topMatch: saeTop / (fx.tokens.length - 1), ms: ft.ms };
  log(`sparse autoencoder: strongest feature matches at ${saeTop}/${fx.tokens.length - 1} tokens, worst relative error ${saeErr.toExponential(1)}, ${ft.ms.toFixed(0)} ms`);
  if (saeErr > 2e-2 || saeTop < fx.tokens.length - 2) fail("sparse autoencoder differs from TransformerLens");

  for (const T of [16, 64, 128, 256]) {
    const tokens = Array.from({ length: T }, (_, i) => (i * 7919 + 13) % 50000);
    await engine.run({ tokens });
    const plain: number[] = [];
    const lensLast: number[] = [];
    for (let i = 0; i < 5; i++) plain.push((await engine.run({ tokens })).ms);
    for (let i = 0; i < 5; i++) lensLast.push((await engine.run({ tokens, lens: { positions: [T - 1] } })).ms);
    const p = { T, forwardMs: median(plain), withLensMs: median(lensLast) };
    if (T <= 16) {
      const all: number[] = [];
      for (let i = 0; i < 5; i++) all.push((await engine.run({ tokens, lens: { positions: tokens.map((_, k) => k) } })).ms);
      (p as any).lensAllPositionsMs = median(all);
    }
    results.perf.push(p);
    log(`perf T=${String(T).padStart(3)}: forward ${p.forwardMs.toFixed(1)} ms, + lens on last token ${p.withLensMs.toFixed(1)} ms${"lensAllPositionsMs" in p ? `, lens on every token ${(p as any).lensAllPositionsMs.toFixed(1)} ms` : ""}`);
  }

  results.ok = results.failures.length === 0;
  log(results.ok ? "ALL CHECKS PASSED" : `${results.failures.length} FAILURES`);
  (window as any).results = results;
}

main().catch((e) => {
  log("ERROR " + (e?.stack ?? e));
  (window as any).results = { ok: false, error: String(e?.message ?? e) };
});
