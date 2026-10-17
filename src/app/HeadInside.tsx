// Inside one attention head: the question it asks, the labels it compares
// with, the values it fetches and what it writes, down to single numbers.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show } from "./analysis.ts";
import { DirectEffect } from "./DirectEffect.tsx";
import { headOutput } from "./heads.ts";
import { pct } from "./PromptBox.tsx";
import { Term } from "./Tip.tsx";
import { headSources, insideHead } from "./inside.ts";
import { fmt, Numbers, Section } from "./inside-ui.tsx";
import { Copying, WiringMap } from "./WeightViews.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  trace: Trace; // needs qkv, pattern, z and resid
  id: string;
  query: number;
  onHead: (id: string) => void;
}

export function HeadInside({ engine, tokenizer, weights, trace, id, query: q, onHead }: Props) {
  const [l, h] = id.split(".").map(Number);
  const [picked, setPicked] = useState<number | null>(null);
  const [hoverDim, setHoverDim] = useState<number | null>(null);
  useEffect(() => setPicked(null), [q, id]);

  const inside = useMemo(() => insideHead(trace, engine.config, l, h, q), [trace, l, h, q, engine]);
  const sources = useMemo(() => headSources(trace, weights, l, h, q), [trace, weights, l, h, q]);
  const outVec = useMemo(() => headOutput(trace, weights, l, h, q), [trace, weights, l, h, q]);
  const out = useCallback(() => outVec, [outVec]);
  const pieces = trace.tokens.map((t, i) => (i === 0 ? "⟨start⟩" : show(tokenizer.piece(t))));
  const at = pieces[q];
  const j = picked !== null && picked < inside.scores.length ? picked : inside.pattern.indexOf(Math.max(...inside.pattern));

  const kMax = Math.max(...inside.keys.flatMap((k) => Array.from(k, Math.abs)));
  const vMax = Math.max(...inside.values.flatMap((v) => Array.from(v, Math.abs)));
  const qMax = Math.max(...Array.from(inside.q, Math.abs));
  const sMax = Math.max(...inside.scores.map(Math.abs));

  return (
    <>
      <header className="deeper-head">
        <h2>Inside head {id}</h2>
        <p className="muted">Where a head looks is only half its story. Here is how it decides, what it carries and what it writes, at “{at}”, down to single numbers. Click a token above to follow another.</p>
      </header>
      <div className="part-steps">
          <Section n="1" title="Ask: a question and a label for each token">
            <p>At “{at}”, the head turns the running summary into a question: 64 numbers called its <Term tip="Made by multiplying the token's 768 numbers by this head's query weights. Every head has its own.">query</Term>. Every token up to here holds up a label, 64 numbers called its <Term tip="Made by multiplying each token's 768 numbers by this head's key weights.">key</Term>. A label matches when its numbers line up with the question's.</p>
            <div className="qk-grid">
              <div className="qk-row qk-head">
                <span className="qk-tok qk-query">{at}</span>
                <Numbers values={inside.q} max={qMax} label={`The query at ${at}`} onHover={setHoverDim} mark={hoverDim} />
                <span className="qk-num muted">score</span>
                <span className="qk-num muted">share</span>
              </div>
              {inside.keys.map((k, t) => (
                <button key={t} className={`qk-row${t === j ? " qk-on" : ""}`} onClick={() => setPicked(t)} aria-pressed={t === j}>
                  <span className="qk-tok">{pieces[t]}</span>
                  <Numbers values={k} max={kMax} label={`The key of ${pieces[t]}`} onHover={setHoverDim} mark={hoverDim} />
                  <span className="qk-num" style={{ color: inside.scores[t] >= 0 ? undefined : "var(--muted)" }}>{fmt(inside.scores[t], 1)}</span>
                  <span className="qk-share"><span className="qk-bar" style={{ width: `${inside.pattern[t] * 100}%` }} /><span className="qk-pct">{pct(inside.pattern[t])}</span></span>
                </button>
              ))}
            </div>
            <p className="muted small">The middle column is each label's match score. Scores are squashed into shares that add up to 100% (a softmax), so the best match takes most of the attention. Click a row to see its score worked out.{hoverDim !== null && <> Number {hoverDim + 1} of 64: query {fmt(inside.q[hoverDim])}.</>}</p>
          </Section>

          <OneScore q={inside.q} k={inside.keys[j]} score={inside.scores[j]} scores={inside.scores} share={inside.pattern[j]} from={at} to={pieces[j]} sMax={sMax} />

          <Section n="2" title="Fetch: a blend of values">
            <p>Each token also offers a <Term tip="64 numbers made from the token's running summary by this head's value weights: what the head would carry away if it looked there.">value</Term>. The head takes a blend, weighted by the shares from step 1.</p>
            <div className="qk-grid">
              {inside.values.map((v, t) => (
                <div key={t} className={`qk-row${inside.pattern[t] >= 0.05 ? " qk-strong" : ""}`}>
                  <span className="qk-tok">{pieces[t]}</span>
                  <Numbers values={v} max={vMax} label={`The value of ${pieces[t]}`} />
                  <span className="qk-num">× {pct(inside.pattern[t])}</span>
                  <span className="qk-share" title="How much this token adds to what the head writes"><span className="qk-bar qk-bar-src" style={{ width: `${sources[t] * 100}%` }} /><span className="qk-pct">{pct(sources[t])}</span></span>
                </div>
              ))}
              <div className="qk-row qk-sum">
                <span className="qk-tok">blend</span>
                <Numbers values={inside.z} max={vMax} label="The blend of values" />
                <span className="qk-num" />
                <span className="qk-num muted">of what it writes</span>
              </div>
            </div>
            <p className="muted small">The last column is how much each token ends up adding to what the head writes, counting both its share of attention and the size of its value. A token can take most of the attention and still add little, like ⟨start⟩, whose value is tiny.</p>
          </Section>

          <Section n="3" title="Write: 64 numbers become 768">
            <p>The head's output weights turn the blend into 768 numbers, which are added to the running summary at “{at}”.</p>
            <Numbers values={outVec} label="What the head writes" height={22} />
            <DirectEffect trace={trace} weights={weights} tokenizer={tokenizer} vector={out} position={q} />
          </Section>

          <Copying engine={engine} tokenizer={tokenizer} weights={weights} head={id} />
          <WiringMap sel={id} onSelect={onHead} />
        </div>
    </>
  );
}

// One match score, all the way down to multiplications.
function OneScore({ q, k, score, scores, share, from, to, sMax }: { q: Float32Array; k: Float32Array; score: number; scores: number[]; share: number; from: string; to: string; sMax: number }) {
  const products = useMemo(() => Float32Array.from(q, (x, i) => x * k[i]), [q, k]);
  const sum = products.reduce((a, b) => a + b, 0);
  const biggest = useMemo(() => Array.from(products, (v, i) => [v, i] as const).sort((a, b) => Math.abs(b[0]) - Math.abs(a[0])).slice(0, 4), [products]);
  const pMax = Math.max(...Array.from(products, Math.abs));
  return (
    <Section n="1½" title={<>One score, down to single numbers: “{from}” asking “{to}”</>} className="one-score">
      <div className="one-grid">
        <span className="card-label">query</span><Numbers values={q} label="query" />
        <span className="card-label">key</span><Numbers values={k} label="key" />
        <span className="card-label">× each</span><Numbers values={products} max={pMax} label="Products" height={22} />
      </div>
      <p className="one-sum">
        {biggest.map(([, i], n) => <span key={i}>{n ? " + " : ""}{fmt(q[i])} × {fmt(k[i])}</span>)} + … (64 in all) = <strong>{fmt(sum)}</strong>
      </p>
      <p className="one-sum">{fmt(sum)} ÷ 8 = <strong>{fmt(score)}</strong> <span className="muted">(8 is the square root of 64, to keep scores in a workable range)</span></p>
      <p className="one-sum">share = e<sup>{fmt(score, 1)}</sup> ÷ (e<sup>score</sup> added up over all {scores.length} tokens) = <strong>{pct(share)}</strong></p>
      <p className="muted small">That's all attention is: multiplying, adding and one squash. The largest products above are where this question and label agree most.{Math.abs(score) === sMax ? " This is the strongest match here." : ""}</p>
    </Section>
  );
}

