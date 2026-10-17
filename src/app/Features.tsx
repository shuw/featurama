// The features a sparse autoencoder finds at layer 7. Steering with them is in the Lab.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { featureDetail, featureHint, featureName, SAE_LAYER, searchFeatures, topAt, useSae, type FeatureDetail, type FeatureInfo } from "./features.ts";
import { STEER_PICKS } from "./feature-labels.ts";
import { GLOSSARY } from "./glossary.ts";
import { tokenize } from "./model.ts";
import { PromptBox } from "./PromptBox.tsx";
import { href, setUrlParam, urlParam } from "./route.ts";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

// Stable stand-ins while the feature dictionary loads, so nothing reruns.
const NOTHING: number[] = [];
const idle = async () => null;

export function Features({ engine, tokenizer, prompt, setPrompt, shuffle, writer }: Props) {
  const L = engine.config.n_layers;
  const sae = useSae(engine);
  const ready = sae.status === "ready" ? sae : null;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  const run = useCallback((t: number[]) => engine.run({ tokens: t, read: { features: true }, lens: { positions: [t.length - 1], topK: 10 } }), [engine]);
  const { result: trace } = useRun(ready ? tokens : NOTHING, ready ? run : idle, 150, ready ? "ready" : "");
  const [query, setQuery] = useState<number | null>(null);
  // The picked feature is in the address (?f=), so a link opens it.
  const [picked, setPicked] = useState<number | null>(() => {
    const f = Number(urlParam("f"));
    return Number.isInteger(f) && f >= 0 && f < 24576 && urlParam("f") !== null ? f : null;
  });
  useEffect(() => {
    setUrlParam("f", picked === null ? null : String(picked));
    return () => setUrlParam("f", null);
  }, [picked]);
  // On narrow screens the card sits below the lists, so bring it into view.
  const side = useRef<HTMLElement>(null);
  const pick = (id: number) => {
    setPicked(id);
    if (innerWidth <= 960) requestAnimationFrame(() => side.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  const [search, setSearch] = useState("");

  const current = !!trace && trace.tokens === tokens;
  const shown = trace?.tokens ?? tokens;
  const T = shown.length;
  const q = query === null ? T - 1 : Math.min(query, T - 1);
  const F = engine.saeSize;
  const top = trace?.features && q > 0 ? topAt(trace.features, F, q, 10) : [];
  const info = (id: number) => ready?.index[id];
  const name = (id: number) => featureName(id, info(id), tokenizer);
  const hint = (id: number) => featureHint(info(id), tokenizer);
  const pieces = shown.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));
  // The picked feature's activation at each token, for tinting the strip.
  const along = picked !== null && trace?.features ? Array.from({ length: T }, (_, t) => (t === 0 ? 0 : trace.features![t * F + picked])) : null;
  const alongMax = along ? Math.max(info(picked!)?.top ?? 0, ...along, 1e-6) : 1;

  const results = useMemo(() => (ready ? searchFeatures(ready.index, tokenizer, search) : []), [search, ready, tokenizer]);

  return (
    <article className="page">
      <header className="page-head">
        <h1>Concepts inside the numbers</h1>
        <p className="dek">Halfway through, at layer {SAE_LAYER}, each token is 768 numbers. A <Term tip={GLOSSARY.sae}>sparse autoencoder</Term> splits them into 24,576 <Term tip={GLOSSARY.feature}>features</Term>, each firing on one kind of thing. In the Lab you can turn one up and watch GPT-2's writing bend toward it.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current && trace?.lens ? predictions(trace, L, tokenizer) : null} stale={!current} />

      {sae.status === "loading" && <p className="card muted">Fetching the feature dictionary (38 MB, only the first time)…</p>}
      {sae.status === "error" && <p className="card">Couldn't load the features: {sae.message}</p>}

      {ready && (
        <div className="features">
          <div className="features-main">
            <div className="feat-strip" role="group" aria-label="Tokens: pick one to see its features">
              {pieces.map((p, t) => (
                <button key={t} className={`feat-tok${t === q ? " feat-tok-on" : ""}`} disabled={t === 0} onClick={() => setQuery(t)}
                  style={along && along[t] > 0 ? { background: `rgba(var(--feat-rgb), ${(0.12 + 0.88 * Math.min(1, along[t] / alongMax)).toFixed(2)})` } : undefined}
                  title={along ? `${name(picked!)}: ${along[t].toFixed(1)}` : undefined}>{show(p)}</button>
              ))}
            </div>
            {along && <p className="muted small">Tinted by how strongly “{name(picked!)}” fires on each token.</p>}

            <section className="card">
              <div className="card-label">Strongest features at “{word(pieces[q] ?? "")}”</div>
              {!trace ? <p className="muted">Reading…</p> : top.length === 0 ? <p className="muted">Nothing fires here.</p> : (
                <ul className="feat-list">
                  {top.map(({ id, value }) => (
                    <li key={id}>
                      <button className={picked === id ? "feat-row feat-row-on" : "feat-row"} onClick={() => pick(id)}>
                        <span className="feat-bar"><span style={{ width: `${Math.min(100, (100 * value) / Math.max(info(id)?.top ?? value, value))}%` }} /></span>
                        <span className="feat-name">{name(id)} <span className="feat-hint">{hint(id)}</span></span>
                        <span className="feat-id">#{id}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <p className="muted small">Bars compare each feature with its strongest firing on Wikipedia.</p>
            </section>

            <section className="card">
              <div className="card-label">Find a feature</div>
              <input className="feat-search" placeholder="A word, like “ocean” or “court”" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a feature" />
              {results.length > 0 && (
                <ul className="feat-list">
                  {results.map((id) => <li key={id}><button className={picked === id ? "feat-row feat-row-on" : "feat-row"} onClick={() => pick(id)}><span className="feat-name">{name(id)} <span className="feat-hint">{hint(id)}</span></span><span className="feat-id">#{id}</span></button></li>)}
                </ul>
              )}
              {search.trim().length >= 2 && results.length === 0 && <p className="muted small">No feature fires mostly on that. Try another word.</p>}
              {STEER_PICKS.length > 0 && (
                <div className="hl-chips compact feat-picks">
                  {STEER_PICKS.map((id) => <button key={id} className={picked === id ? "hl-chip hl-on" : "hl-chip"} onClick={() => pick(id)}>{name(id)}</button>)}
                </div>
              )}
            </section>
          </div>

          <aside className="features-side" ref={side}>
            {picked === null ? (
              <section className="card">
                <div className="card-label">How to read this</div>
                <p>Click a token to see which features fire on it, then click a feature to see where else it fires and to steer GPT-2 with it.</p>
                <p className="muted">Most features have no name: GPT-2 was never told about concepts. We name each by the tokens it fires on most, and hand-label a few.</p>
              </section>
            ) : (
              <FeatureCard key={picked} id={picked} info={info(picked)} name={name(picked)} hint={hint(picked)} tokenizer={tokenizer} />
            )}
          </aside>
        </div>
      )}
    </article>
  );
}

function FeatureCard({ id, info, name, hint, tokenizer }: {
  id: number; info: FeatureInfo | undefined; name: string; hint: string; tokenizer: Tokenizer;
}) {
  const [detail, setDetail] = useState<FeatureDetail | null>(null);
  useEffect(() => {
    let live = true;
    featureDetail(id).then((d) => live && setDetail(d));
    return () => {
      live = false;
    };
  }, [id]);

  const often = info && info.perMillion > 0 ? `fires on about 1 in ${Math.max(1, Math.round(1e6 / info.perMillion)).toLocaleString()} tokens of Wikipedia` : "never fired on our Wikipedia sample";
  return (
    <section className="card head-card">
      <div className="head-name">Feature #{id}</div>
      <p className="head-gist">{name} <span className="feat-hint">{hint}</span></p>
      <p className="muted small">It {often}.</p>

      <div className="effect">
        <Term tip={GLOSSARY.featurePush}><span className="card-label">Pushes the next token</span></Term>
        {!detail ? <p className="muted small">Loading…</p> : (
          <>
            <div className="push-row"><span className="push-label">toward</span>{detail.up.slice(0, 4).map((t) => <span key={t} className="push push-up">{show(tokenizer.piece(t))}</span>)}</div>
            <div className="push-row"><span className="push-label">away from</span>{detail.down.slice(0, 4).map((t) => <span key={t} className="push push-down">{show(tokenizer.piece(t))}</span>)}</div>
          </>
        )}
      </div>

      <div className="card-buttons">
        <a className="primary" href={href({ view: "lab", tool: "steer" })}>Steer GPT-2 with it →</a>
      </div>

      {detail && detail.passages.length > 0 && (
        <div className="examples">
          <div className="card-label">Where it fires hardest on Wikipedia</div>
          {detail.passages.slice(0, 3).map((p, i) => (
            <p key={i} className="example-text">
              {p.ids.map((t, j) => (
                <span key={j} className={j === p.at ? "ex-tok ex-here" : "ex-tok"} style={p.acts[j] > 12 ? { background: `rgba(var(--feat-rgb), ${(0.1 + 0.8 * (p.acts[j] / 255)).toFixed(2)})` } : undefined}>{tokenizer.piece(t)}</span>
              ))}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
