// Start at the answer: how every part voted for it, then trace it back by
// unplugging parts one at a time.

import { useCallback, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Trace } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { tokenize } from "./model.ts";
import { pct, PromptBox } from "./PromptBox.tsx";
import { href } from "./route.ts";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { headSources, partsBefore, scanAnswer, scanPart, votes, type PartId } from "./zoom.ts";
import { cssRgb, diverging, fmt, partGist, partName, Section, useThemeVersion } from "./zoom-ui.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

const IOI = "When Mary and John went to the store, John gave a drink to";

export function ZoomAnswer({ engine, tokenizer, weights, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, n_heads: H } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { z: true, resid: true, residMid: true, qkv: true, pattern: true }, lens: { positions: [t.length - 1], topK: 5 } }), [engine]);
  const { result: trace } = useRun(tokens, run);
  const current = !!trace && trace.tokens === tokens;
  const next = trace ? predictions(trace, L, tokenizer) : null;
  const ids = trace ? Array.from(trace.lens!.ids.subarray(L * 5, L * 5 + 5)) : [];
  const [pick, setPick] = useState<[number, number]>([0, 1]);
  const target = ids[pick[0]];
  const rival = ids[pick[1]];
  const pos = (trace?.tokens.length ?? 1) - 1;
  const v = useMemo(() => (trace && target !== undefined ? votes(trace, weights, pos, target, rival) : null), [trace, weights, pos, target, rival]);
  const byId = useMemo(() => new Map(v?.parts.map((p) => [p.id, p.value]) ?? []), [v]);
  const [selected, setSelected] = useState<PartId | null>(null);
  const theme = useThemeVersion();
  const colors = useMemo(() => ({ pos: cssRgb("--pos-rgb"), neg: cssRgb("--neg-rgb") }), [theme]);
  const max = v ? Math.max(...v.parts.map((p) => Math.abs(p.value))) : 1;
  const ranked = v ? [...v.parts].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : [];
  const shownParts = ranked.slice(0, 10);
  const rest = ranked.slice(10).reduce((s, p) => s + p.value, 0);
  const tw = target !== undefined ? word(tokenizer.piece(target)) : "";
  const rw = rival !== undefined ? word(tokenizer.piece(rival)) : "";
  const runKey = `${tokens.join(",")}|${target}|${rival}`;

  const cell = (id: PartId) => {
    const value = byId.get(id) ?? 0;
    return { background: diverging(value, max, colors.pos, colors.neg) };
  };

  return (
    <>
      <PromptBox writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer} next={current ? next : null} stale={!current} />
      {prompt !== IOI && <button className="link-small" onClick={() => { setPrompt(IOI); setPick([0, 3]); }}>Try the classic: “{IOI}”</button>}

      {trace && next && v && (
        <div className="zoom-steps">
          <Section n="1" title="Who voted?">
            <div className="vote-pick">
              <span className="card-label">Why</span>
              <select value={pick[0]} onChange={(e) => setPick([Number(e.target.value), Number(e.target.value) === pick[1] ? pick[0] : pick[1]])} aria-label="The token to explain">
                {next.map((p, i) => <option key={i} value={i}>{show(p.piece)} ({pct(p.p)})</option>)}
              </select>
              <span className="card-label">rather than</span>
              <select value={pick[1]} onChange={(e) => setPick([Number(e.target.value) === pick[0] ? pick[1] : pick[0], Number(e.target.value)])} aria-label="The token to compare with">
                {next.map((p, i) => <option key={i} value={i}>{show(p.piece)} ({pct(p.p)})</option>)}
              </select>
            </div>
            <p>GPT-2 scores every possible next token. “{tw}” beats “{rw}” by <strong>{fmt(v.total)}</strong> points. Every head and MLP writes into the running summary, and each part's writing on its own adds to that gap or takes from it: its <Term tip="A part's direct vote: what its output adds to the score gap through the final LayerNorm and the token table, ignoring any effect on later parts. The votes add up exactly to the real gap.">vote</Term>. Blue votes for “{tw}”, red for “{rw}”.</p>
            <div className="vote-body">
              <div className="table vote-table" role="grid" aria-label="Each part's vote">
                <div className="table-corner"><span className="muted small">head</span></div>
                {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
                {Array.from({ length: H }, (_, h) => (
                  <Row key={h} label={`${h}`}>
                    {Array.from({ length: L }, (_, l) => {
                      const id = `${l}.${h}`;
                      return <button key={id} className={`head${selected === id ? " head-picked" : ""}`} style={cell(id)} onClick={() => setSelected(id)} aria-label={`Head ${id}: ${fmt(byId.get(id) ?? 0)}`} title={`${id}: ${fmt(byId.get(id) ?? 0)}`} />;
                    })}
                  </Row>
                ))}
                <div className="table-row-label table-mlp-label">MLP</div>
                {Array.from({ length: L }, (_, l) => (
                  <button key={l} className={`mlp-cell${selected === `m${l}` ? " head-picked" : ""}`} style={cell(`m${l}`)} onClick={() => setSelected(`m${l}`)} aria-label={`MLP ${l}: ${fmt(byId.get(`m${l}`) ?? 0)}`} title={`MLP ${l}: ${fmt(byId.get(`m${l}`) ?? 0)}`} />
                ))}
              </div>
              <ul className="vote-list">
                {shownParts.map((p) => (
                  <li key={p.id}>
                    <button className={`vote-row${selected === p.id ? " vote-on" : ""}`} onClick={() => setSelected(p.id)}>
                      <span className="vote-name">{partName(p.id)}</span>
                      <VoteBar value={p.value} max={max} />
                      <span className="vote-num">{p.value > 0 ? "+" : ""}{fmt(p.value)}</span>
                    </button>
                  </li>
                ))}
                <li className="vote-row vote-rest"><span className="vote-name" title={`The other ${ranked.length - 10} parts`}>{ranked.length - 10} more</span><VoteBar value={rest} max={max} /><span className="vote-num">{rest > 0 ? "+" : ""}{fmt(rest)}</span></li>
                <li className="vote-row vote-total"><span className="vote-name">All together</span><span /><span className="vote-num">{fmt(v.total)}</span></li>
              </ul>
            </div>
            {selected && <PartNote id={selected} value={byId.get(selected) ?? 0} tw={tw} />}
          </Section>

          <TraceBack engine={engine} weights={weights} tokenizer={tokenizer} trace={trace} target={target} rival={rival} tw={tw} rw={rw} votes={byId} runKey={runKey} />
        </div>
      )}
    </>
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

function VoteBar({ value, max }: { value: number; max: number }) {
  const w = Math.min(50, (50 * Math.abs(value)) / (max || 1));
  return (
    <span className="vote-bar">
      <span className={value >= 0 ? "vote-fill vote-pos" : "vote-fill vote-neg"} style={value >= 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
    </span>
  );
}

function PartNote({ id, value, tw }: { id: PartId; value: number; tw: string }) {
  const zoomable = id !== "embed" && id !== "bias";
  return (
    <div className="part-note">
      <strong>{partName(id)}</strong> <span className="muted">· {partGist(id)}</span>
      <p className="small">Its vote: {value > 0 ? "+" : ""}{fmt(value)} {value >= 0 ? `for “${tw}”` : `against “${tw}”`}.</p>
      {zoomable && <a className="link-small" href={href(id.startsWith("m") ? { view: "zoom", part: "mlp", arg: id.slice(1) } : { view: "zoom", part: "head", arg: id })}>Look inside {partName(id)} →</a>}
    </div>
  );
}

// Unplug every part in turn, then follow the biggest effects back.
function TraceBack({ engine, weights, tokenizer, trace, target, rival, tw, rw, votes: direct, runKey }: {
  engine: GpuEngine; weights: Weights; tokenizer: Tokenizer; trace: Trace; target: number; rival: number; tw: string; rw: string; votes: Map<PartId, number>; runKey: string;
}) {
  const [scan, setScan] = useState<{ key: string; base: number; change: Map<PartId, number> } | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [chain, setChain] = useState<PartId[]>([]);
  const [deps, setDeps] = useState<Map<string, Map<PartId, number>>>(new Map());
  const tokens = trace.tokens;
  const pos = tokens.length - 1;
  const fresh = scan?.key === runKey;

  const start = async () => {
    const parts = partsBefore("answer", engine.config);
    setChain([]);
    setDeps(new Map());
    setProgress({ done: 0, total: parts.length });
    const r = await scanAnswer(engine, tokens, pos, target, rival, parts, (done) => setProgress({ done, total: parts.length }));
    setScan({ key: runKey, ...r });
    setProgress(null);
  };
  const depend = async (id: PartId) => {
    const key = `${runKey}|${id}`;
    if (deps.has(key)) return;
    const parts = partsBefore(id, engine.config);
    setProgress({ done: 0, total: parts.length });
    const r = await scanPart(engine, weights, tokens, pos, id, parts, (done) => setProgress({ done, total: parts.length }));
    setDeps((m) => new Map(m).set(key, r));
    setProgress(null);
  };
  const open = (id: PartId, depth: number) => {
    setChain((c) => [...c.slice(0, depth), id]);
  };

  const ranked = fresh ? [...scan!.change.entries()].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8) : [];
  const max = ranked.length ? Math.max(...ranked.map(([, c]) => Math.abs(c)), ...ranked.map(([id]) => Math.abs(direct.get(id) ?? 0))) : 1;
  const pieces = tokens.map((t, i) => (i === 0 ? "⟨start⟩" : show(tokenizer.piece(t))));

  return (
    <Section n="2" title="Trace it back">
      <p>Votes count only what a part writes straight into the answer. Parts also work through later parts: a head can tell another head where to look. To catch that, unplug each part in turn and see how the gap between “{tw}” and “{rw}” changes.</p>
      {!fresh && (
        <button className="primary" onClick={start} disabled={!!progress}>{progress ? `Unplugging… ${progress.done} of ${progress.total}` : "Unplug each part in turn (156 runs)"}</button>
      )}
      {fresh && (
        <>
          <p className="small">With everything plugged in, the gap is {fmt(scan!.base)}. The parts that matter most:</p>
          <ul className="vote-list trace-list">
            <li className="trace-head"><span /><span className="muted small">vote</span><span className="muted small">total effect</span><span /></li>
            {ranked.map(([id, c]) => (
              <li key={id}>
                <button className={`vote-row trace-row${chain[0] === id ? " vote-on" : ""}`} onClick={() => { open(id, 0); depend(id); }}>
                  <span className="vote-name">{partName(id)}</span>
                  <VoteBar value={direct.get(id) ?? 0} max={max} />
                  <VoteBar value={-c} max={max} />
                  <span className="vote-num">{-c > 0 ? "+" : ""}{fmt(-c)}</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="muted small">Total effect is how much the gap shrinks when the part is unplugged; negative means the gap grows. When it's far bigger than the vote, the part works mostly through later parts. When it's smaller, others step in when it's gone (self-repair). Click a part to trace what it depends on.</p>
        </>
      )}
      {progress && fresh && <p className="muted small">Unplugging… {progress.done} of {progress.total}</p>}
      {fresh && chain.length > 0 && (
        <div className="trace-chain">
          <div className="trace-crumbs">
            <span className="crumb">“{tw}”</span>
            {chain.map((id, i) => (
              <span key={i}><span className="crumb-arrow">←</span><button className="crumb" onClick={() => setChain(chain.slice(0, i + 1))}>{partName(id)}</button></span>
            ))}
          </div>
          {(() => {
            const id = chain[chain.length - 1];
            const dep = deps.get(`${runKey}|${id}`);
            const reads = id.startsWith("m") ? null : (() => {
              const [l, h] = id.split(".").map(Number);
              return headSources(trace, weights, l, h, pos).map((s, j) => [s, j] as const).sort((a, b) => b[0] - a[0]).slice(0, 3).filter(([s]) => s > 0.05);
            })();
            const top = dep ? [...dep.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6) : [];
            return (
              <div className="trace-node">
                <div className="head-name">{partName(id)}</div>
                <p className="muted small">{partGist(id)}</p>
                {reads ? (
                  <p className="small">At the last token it reads mostly from {reads.map(([s, j], k) => <span key={j}>{k ? ", " : ""}<span className="mono">{pieces[j]}</span> ({pct(s)})</span>)}.</p>
                ) : <p className="small">An MLP reads only the token it's on.</p>}
                <div className="card-label">What it depends on</div>
                {!dep ? <p className="muted small">{progress ? `Unplugging each earlier part… ${progress.done} of ${progress.total}` : "Working…"}</p> : dep.size === 0 ? (
                  <p className="small trace-end">Nothing runs before it: it reads the tokens and their positions straight from the embedding. This is as far back as the trace goes.</p>
                ) : (
                  <ul className="vote-list">
                    {top.map(([d, s]) => (
                      <li key={d}>
                        <button className="vote-row dep-row" onClick={() => { open(d, chain.length); depend(d); }}>
                          <span className="vote-name">{partName(d)}</span>
                          <span className="dep-bar"><span style={{ width: `${Math.min(100, s * 100)}%` }} /></span>
                          <span className="vote-num">{pct(Math.min(s, 9.99))}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                {dep && dep.size > 0 && <p className="muted small">How much {partName(id)}'s output changes when each earlier part is unplugged, as a share of its size.{top[0]?.[0] === "m0" ? " MLP 0 shows up almost everywhere: GPT-2 uses it to finish reading each token in, so without it little downstream works." : ""}</p>}
                <a className="link-small" href={href(id.startsWith("m") ? { view: "zoom", part: "mlp", arg: id.slice(1) } : { view: "zoom", part: "head", arg: id })}>Look inside {partName(id)} →</a>
              </div>
            );
          })()}
        </div>
      )}
    </Section>
  );
}
