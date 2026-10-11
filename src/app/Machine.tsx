// The home screen: every part of GPT-2 laid out like a periodic table,
// shaded by how much it writes at the last word of your prompt.

import { useCallback, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, word } from "./analysis.ts";
import { headActivity, IOI_ROLES, IOI_SOURCE, JOB_TEXT, jobsOf, mlpActivity, type HeadJobs } from "./heads.ts";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { href } from "./route.ts";
import { useRun } from "./useRun.ts";

const DEFAULT = "When Mary and John went to the store, John gave a drink to";

const EXHIBITS = [
  { n: 1, name: "Tokens", text: "How text is cut into pieces" },
  { n: 2, name: "Attention", text: "Who looks at whom", route: href({ view: "attention" }) },
  { n: 3, name: "Layers", text: "Watch the guess take shape", route: href({ view: "layers" }) },
  { n: 4, name: "Patching", text: "Swap activations until the answer flips" },
  { n: 5, name: "Features", text: "Concepts inside the numbers" },
  { n: 6, name: "Steering", text: "Turn a concept up and watch the writing bend" },
];

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  jobs: HeadJobs;
}

export function Machine({ engine, tokenizer, weights, jobs }: Props) {
  const { n_layers: L, n_heads: H, d_model: d } = engine.config;
  const [text, setText] = useState(DEFAULT);
  const [tokens, setTokens] = useState(() => tokenize(tokenizer, DEFAULT));
  const [focus, setFocus] = useState<string | null>(null);
  const run = useCallback(async (t: number[]) => {
    const trace = await engine.run({ tokens: t, read: { z: true, resid: true, residMid: true }, lens: { positions: [t.length - 1], topK: 5 } });
    const last = t.length - 1;
    return { trace, heads: headActivity(trace, weights, last), mlps: mlpActivity(trace, d, L, last) };
  }, [engine, weights, d, L]);
  const { result } = useRun(tokens, run);

  const current = !!result && result.trace.tokens === tokens;
  const shown = result?.trace.tokens ?? tokens;
  const lastWord = word(tokenizer.piece(shown[shown.length - 1]));
  const headMax = result ? Math.max(...result.heads) : 1;
  const mlpMax = result ? Math.max(...result.mlps) : 1;
  // Square root, so the busiest head doesn't wash out the rest.
  const shade = (v: number, max: number) => (0.05 + 0.95 * Math.sqrt(v / max)).toFixed(2);

  const focused = focus ? describeHead(focus, jobs, H, result?.heads ?? null, headMax) : null;

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">The machine · GPT-2 small</div>
        <h1>Every part of the model, working on your prompt</h1>
        <p className="dek">Twelve layers, each with twelve attention heads that move information between words and one MLP that works on each word alone. Shading shows how much each part writes at the last word, “{lastWord}”.</p>
      </header>

      <PromptBox value={text} onChange={(v) => { setText(v); setTokens(tokenize(tokenizer, v)); }} tokens={tokens} tokenizer={tokenizer}
        next={result ? predictions(result.trace, L, tokenizer) : null} stale={!current} />

      <div className="machine-body">
        <div className={!current && result ? "stage table-wrap stale" : "stage table-wrap"}>
          <div className="table" role="grid" aria-label="Attention heads by layer" onMouseLeave={() => setFocus(null)}>
            <div />
            {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
            {Array.from({ length: H }, (_, h) => (
              <Row key={h} label={`${h}`}>
                {Array.from({ length: L }, (_, l) => {
                  const id = `${l}.${h}`;
                  const measured = jobsOf(jobs, l, h, H);
                  return (
                    <a key={id} href={href({ view: "attention", head: id })} className={focus === id ? "head head-on" : "head"}
                      style={{ background: `rgba(var(--attn-rgb), ${result ? shade(result.heads[l * H + h], headMax) : "0.05"})` }}
                      onMouseEnter={() => setFocus(id)} onFocus={() => setFocus(id)}
                      aria-label={`Head ${id}${measured.length ? `, ${measured.map((j) => JOB_TEXT[j].name).join(", ")}` : ""}${IOI_ROLES[id] ? ", IOI role" : ""}`}>
                      {measured.length > 0 && <span className="mark-job" />}
                      {IOI_ROLES[id] && <span className="mark-ioi" />}
                    </a>
                  );
                })}
              </Row>
            ))}
            <div className="table-row-label table-mlp-label">MLP</div>
            {Array.from({ length: L }, (_, l) => (
              <a key={l} href={href({ view: "layers" })} className="mlp-cell" aria-label={`MLP of layer ${l}`}
                style={{ background: `rgba(var(--mlp-rgb), ${result ? shade(result.mlps[l], mlpMax) : "0.05"})` }} />
            ))}
          </div>
          <div className="table-foot">
            <span>layer →</span>
            <span className="scale"><span>less</span><span className="scale-bar" /><span>more written at “{lastWord}”</span></span>
          </div>
        </div>

        <aside className="machine-side">
          {focused ? (
            <div className="card head-card">
              <div className="head-name">Head {focus}</div>
              <p className="muted">Layer {focus!.split(".")[0]}, head {focus!.split(".")[1]} · writes {focused.activity} as much as the busiest head at “{lastWord}”.</p>
              {focused.jobs.map((j) => <p key={j.name}><strong>{j.name}.</strong> {j.text} <span className="muted">Measured just now: {j.score}.</span></p>)}
              {focused.ioi && <p><strong>IOI role.</strong> {focused.ioi} <span className="muted">({IOI_SOURCE})</span></p>}
              {!focused.jobs.length && !focused.ioi && <p className="muted">No known job yet. Open it and see if you can tell what it's for.</p>}
              <a className="primary link-button" href={href({ view: "attention", head: focus! })}>Open head {focus} →</a>
            </div>
          ) : (
            <div className="card">
              <div className="card-label">How to read this</div>
              <p>Columns are layers, from the first (0) to the last (11). Each square is one attention head; the teal row is each layer's MLP.</p>
              <p className="legend"><span><span className="mark-job static" /> job measured in your browser just now</span><span><span className="mark-ioi static" /> role found in published research</span></p>
              <p className="muted">Point at a head to learn about it; click to open it.</p>
            </div>
          )}
          <nav className="exhibit-list" aria-label="Exhibits">
            <div className="card-label">Exhibits</div>
            {EXHIBITS.map((e) =>
              e.route ? (
                <a key={e.n} href={e.route} className="exhibit-link"><span className="n">{e.n}</span><span><strong>{e.name}</strong><span className="muted"> · {e.text}</span></span></a>
              ) : (
                <div key={e.n} className="exhibit-link soon"><span className="n">{e.n}</span><span><strong>{e.name}</strong> · {e.text}</span><span className="soon-tag">soon</span></div>
              ),
            )}
          </nav>
        </aside>
      </div>
    </article>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <div className="table-row-label">{label}</div>
      {children}
    </>
  );
}

function describeHead(id: string, jobs: HeadJobs, H: number, heads: Float32Array | null, max: number) {
  const [l, h] = id.split(".").map(Number);
  return {
    activity: heads ? `${Math.round((heads[l * H + h] / max) * 100)}%` : "…",
    jobs: jobsOf(jobs, l, h, H).map((j) => ({ ...JOB_TEXT[j], score: jobs.scores[j][l * H + h].toFixed(2) })),
    ioi: IOI_ROLES[id],
  };
}
