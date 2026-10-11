// The side card for one layer's MLP.

import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { DirectEffect } from "./DirectEffect.tsx";
import { GLOSSARY } from "./glossary.ts";
import { mlpOutput } from "./heads.ts";
import { MLP_NOTES, SOURCES } from "./research.ts";
import { href } from "./route.ts";
import { Term } from "./Tip.tsx";
import { useCallback } from "react";

export function MlpCard({ layer, trace, weights, tokenizer, position, activity }: {
  layer: number; trace: Trace | null; weights: Weights; tokenizer: Tokenizer; position: number; activity: string;
}) {
  const d = weights.config.d_model;
  const vector = useCallback(() => mlpOutput(trace!, d, layer, position), [trace, d, layer, position]);
  const notes = MLP_NOTES[layer] ?? [];
  return (
    <div className="card head-card">
      <div className="head-name">MLP {layer}</div>
      <p className="muted">The <Term tip={GLOSSARY.mlp}>MLP</Term> of layer {layer}. It works on each token by itself: 768 numbers in, 3,072 inside, 768 out. Here it adds {activity} as much as the busiest MLP.</p>
      {trace && <DirectEffect trace={trace} weights={weights} tokenizer={tokenizer} vector={vector} position={position} />}
      <div className="research">
        <div className="card-label">What researchers found</div>
        {notes.length ? (
          <ul>{notes.map((n, i) => <li key={i}><strong>{n.role}.</strong> {n.text} <a className="src" href={SOURCES[n.source].url} target="_blank" rel="noreferrer" title={SOURCES[n.source].name}>source</a></li>)}</ul>
        ) : (
          <p className="muted">No published study singles out this MLP. Most of what MLPs know is spread across thousands of neurons. The <a href={href({ view: "features" })}>Features page</a> splits layer 7's numbers into readable features instead.</p>
        )}
      </div>
      <div className="card-buttons">
        <a className="secondary" href={href({ view: "zoom", part: "mlp", arg: String(layer) })}>Look inside its neurons →</a>
        <a className="secondary" href={href({ view: "layers" })}>See it in Layers →</a>
      </div>
    </div>
  );
}
