// Lab: steer GPT-2's writing by adding a feature's direction as it writes.

import { useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { featureHint, featureName, searchFeatures, useSae, write } from "./features.ts";
import { STEER_PICKS } from "./feature-labels.ts";
import { GLOSSARY } from "./glossary.ts";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { href, setUrlParam, urlParam } from "./route.ts";
import { Term } from "./Tip.tsx";
import type { Writer } from "./useWriter.ts";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function LabSteer({ engine, tokenizer, prompt, setPrompt, shuffle, writer }: Props) {
  const sae = useSae(engine);
  const ready = sae.status === "ready" ? sae : null;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  // The feature is in the address (?f=), shared with the Features page.
  const [picked, setPicked] = useState<number>(() => {
    const f = Number(urlParam("f"));
    return urlParam("f") !== null && Number.isInteger(f) && f >= 0 && f < 24576 ? f : STEER_PICKS[0];
  });
  useEffect(() => {
    setUrlParam("f", String(picked));
    return () => setUrlParam("f", null);
  }, [picked]);
  const [search, setSearch] = useState("");
  const results = useMemo(() => (ready ? searchFeatures(ready.index, tokenizer, search) : []), [search, ready, tokenizer]);
  const name = (id: number) => featureName(id, ready?.index[id], tokenizer);
  const hint = (id: number) => featureHint(ready?.index[id], tokenizer);

  const [strength, setStrength] = useState(50);
  const [out, setOut] = useState<{ plain: string; steered: string; at: number; id: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const steer = async () => {
    if (!ready) return;
    setBusy(true);
    const seed = Math.floor(Math.random() * 1e9);
    const dir = await ready.sae.direction(picked);
    setOut({ plain: "", steered: "", at: strength, id: picked });
    const n = 24;
    await write(engine, tokenizer, tokens, n, seed, undefined, (plain) => setOut((o) => o && { ...o, plain }));
    await write(engine, tokenizer, tokens, n, seed, { direction: dir, strength }, (steered) => setOut((o) => o && { ...o, steered }));
    setBusy(false);
  };

  return (
    <>
      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer} next={null} nextNote="Steering writes on from here" stale={false} />
      {sae.status === "loading" && <p className="card muted">Fetching the feature dictionary (38 MB, only the first time)…</p>}
      {sae.status === "error" && <p className="card">Couldn't load the features: {sae.message}</p>}
      {ready && (
        <div className="features">
          <div className="features-main">
            <section className="card">
              <div className="card-label">1 · Pick a <Term tip={GLOSSARY.feature}>feature</Term></div>
              <div className="hl-chips compact feat-picks">
                {STEER_PICKS.map((id) => <button key={id} className={picked === id ? "hl-chip hl-on" : "hl-chip"} onClick={() => setPicked(id)}>{name(id)}</button>)}
              </div>
              <input className="feat-search" placeholder="Or find one: a word, like “ocean” or “court”" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a feature" />
              {results.length > 0 && (
                <ul className="feat-list">
                  {results.map((id) => <li key={id}><button className={picked === id ? "feat-row feat-row-on" : "feat-row"} onClick={() => setPicked(id)}><span className="feat-name">{name(id)} <span className="feat-hint">{hint(id)}</span></span><span className="feat-id">#{id}</span></button></li>)}
                </ul>
              )}
              {search.trim().length >= 2 && results.length === 0 && <p className="muted small">No feature fires mostly on that. Try another word.</p>}
              <p className="muted small">Picked: <strong>{name(picked)}</strong> <span className="feat-hint">{hint(picked)}</span> · <a href={href({ view: "features" })}>where it fires →</a></p>
            </section>
            <section className="card steer">
              <div className="card-label">2 · Choose how hard to push</div>
              <label className="steer-row">
                <span className="push-label">strength</span>
                <input type="range" min={-100} max={150} step={5} value={strength} onChange={(e) => setStrength(Number(e.target.value))} aria-label="Steering strength" />
                <span className="mono">{strength > 0 ? "+" : ""}{strength}</span>
              </label>
              <button className="primary" onClick={steer} disabled={busy}>{busy ? "Writing…" : "Write with and without it"}</button>
              <p className="muted small">GPT-2 writes 24 tokens twice from your prompt, the second time with the feature's direction added to every token at layer 7. Around +50 changes the topic; past +90 it falls apart; negative pushes away.</p>
            </section>
          </div>
          <aside className="features-side">
            <section className="card">
              <div className="card-label">3 · Compare</div>
              {!out ? <p className="muted">Press “Write with and without it”.</p> : (
                <div className="steer-out steer-stack">
                  <div><div className="card-label">Without</div><p className="steer-text">{out.plain}</p></div>
                  <div><div className="card-label">With {name(out.id)} at {out.at > 0 ? "+" : ""}{out.at}</div><p className="steer-text steer-on">{out.steered}</p></div>
                </div>
              )}
              <p className="muted small">Both use the same dice rolls, so differences come from the steering.</p>
            </section>
          </aside>
        </div>
      )}
    </>
  );
}
