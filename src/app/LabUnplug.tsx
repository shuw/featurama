// Lab: switch heads and MLPs off and see what GPT-2 says without them.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Intervention } from "../engine/hooks.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { predictions, show, word } from "./analysis.ts";
import { headActivity, mlpActivity } from "./heads.ts";
import { tokenize } from "./model.ts";
import { pct, PromptBox } from "./PromptBox.tsx";
import { setUrlParam, urlParam } from "./route.ts";
import { useRun } from "./useRun.ts";
import type { Writer } from "./useWriter.ts";
import { unplug } from "./inside.ts";
import { partGist, partName } from "./inside-ui.tsx";

export const IOI_PROMPT = "When Mary and John went to the store, John gave a drink to";
export const NAME_MOVERS = ["9.6", "9.9", "10.0"];
const BACKUPS = ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"];
const PART = /^(m([0-9]|1[01])|([0-9]|1[01])\.([0-9]|1[01]))$/;

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function LabUnplug({ engine, tokenizer, weights, prompt, setPrompt, shuffle, writer }: Props) {
  const { n_layers: L, n_heads: H, d_model: d } = engine.config;
  const tokens = useMemo(() => tokenize(tokenizer, prompt), [tokenizer, prompt]);
  // What's unplugged is in the address (?off=9.9,m3), so links can set it up.
  const [off, setOff] = useState<string[]>(() => (urlParam("off") ?? "").split(",").filter((x) => PART.test(x)));
  useEffect(() => {
    setUrlParam("off", off.length ? off.join(",") : null);
    return () => setUrlParam("off", null);
  }, [off]);
  const key = off.join(",");
  const [hover, setHover] = useState<string | null>(null);

  const run = useCallback(async (t: number[]) => {
    const last = t.length - 1;
    const interventions: Intervention[] = off.map(unplug);
    const intact = await engine.run({ tokens: t, read: { z: true, resid: true, residMid: true }, lens: { positions: [last], topK: 5 } });
    const targets = Array.from(intact.lens!.ids.subarray(L * 5, L * 5 + 5));
    const changed = interventions.length ? await engine.run({ tokens: t, interventions, lens: { positions: [last], topK: 5, targets } }) : intact;
    const now = tokenizer.piece(changed.lens!.ids[L * 5]);
    const items = targets.map((id, k) => ({
      piece: tokenizer.piece(id),
      before: Math.exp(intact.lens!.logprobs[L * 5 + k]),
      after: interventions.length ? Math.exp(changed.lens!.targetLogprobs[L * 5 + k]) : Math.exp(intact.lens!.logprobs[L * 5 + k]),
    }));
    return { intact, items, now, next: predictions(changed, L, tokenizer) };
  }, [engine, tokenizer, L, key]);
  const { result, version } = useRun(tokens, run, 150, key);
  const current = !!result && result.intact.tokens === tokens;
  const last = (result?.intact.tokens.length ?? tokens.length) - 1;
  const activity = useMemo(() => (result ? { heads: headActivity(result.intact, weights, last), mlps: mlpActivity(result.intact, d, L, last) } : null), [result, weights, last, d, L]);
  const headMax = activity ? Math.max(...activity.heads) : 1;
  const mlpMax = activity ? Math.max(...activity.mlps) : 1;
  const shade = (v: number, max: number) => (0.05 + 0.95 * Math.sqrt(v / max)).toFixed(2);
  const toggle = (id: string) => setOff((u) => (u.includes(id) ? u.filter((x) => x !== id) : [...u, id]));
  const isOff = (id: string) => off.includes(id);
  const lastWord = word(tokenizer.piece(tokens[tokens.length - 1] ?? 0));

  return (
    <>
      <PromptBox autoFocus writer={writer} onShuffle={shuffle} value={prompt} onChange={setPrompt} tokens={tokens} tokenizer={tokenizer}
        next={current ? result!.next : null} stale={!current} />
      <div className="machine-body">
        <div className={!current && result ? "stage table-wrap stale" : "stage table-wrap"}>
          <div className="table" role="grid" aria-label="Heads and MLPs: click to unplug" onMouseLeave={() => setHover(null)}>
            <div className="table-corner">head</div>
            {Array.from({ length: L }, (_, l) => <div key={l} className="table-col">{l}</div>)}
            {Array.from({ length: H }, (_, h) => (
              <Row key={h} label={`${h}`}>
                {Array.from({ length: L }, (_, l) => {
                  const id = `${l}.${h}`;
                  return (
                    <button key={id} className={`head${isOff(id) ? " head-off" : ""}`} aria-pressed={isOff(id)} aria-label={`Head ${id}${isOff(id) ? ", unplugged" : ""}`}
                      onClick={() => toggle(id)} onMouseEnter={() => setHover(id)} onFocus={() => setHover(id)}
                      style={{ ...(isOff(id) ? {} : { background: `rgba(var(--attn-rgb), ${activity ? shade(activity.heads[l * H + h], headMax) : "0.05"})` }), animationName: version % 2 ? "ripple-a" : "ripple-b", animationDelay: `${l * 35}ms` }} />
                  );
                })}
              </Row>
            ))}
            <div className="table-row-label table-mlp-label">MLP</div>
            {Array.from({ length: L }, (_, l) => {
              const id = `m${l}`;
              return (
                <button key={id} className={`mlp-cell${isOff(id) ? " head-off" : ""}`} aria-pressed={isOff(id)} aria-label={`MLP ${l}${isOff(id) ? ", unplugged" : ""}`}
                  onClick={() => toggle(id)} onMouseEnter={() => setHover(id)} onFocus={() => setHover(id)}
                  style={isOff(id) ? undefined : { background: `rgba(var(--mlp-rgb), ${activity ? shade(activity.mlps[l], mlpMax) : "0.05"})` }} />
              );
            })}
          </div>
          {hover && <div className="grid-hint" aria-hidden="true"><strong>{partName(hover)}</strong> {partGist(hover)}</div>}
          <p className="muted small">Click any square to unplug it; click again to plug it back in. Brighter parts add more at “{lastWord}”.</p>
        </div>

        <aside className="machine-side">
          <section className="card card-unplugged" aria-live="polite">
            <div className="card-label">Unplugged</div>
            {off.length === 0 ? (
              <>
                <p>Nothing yet. Click squares in the grid, or try the classic experiment from the paper that mapped how GPT-2 picks names.</p>
                <button className="secondary" onClick={() => { setPrompt(IOI_PROMPT); setOff(NAME_MOVERS); }}>Unplug the name movers</button>
              </>
            ) : (
              <div className="plug-list">
                {off.map((id) => <button key={id} className="plug-chip" onClick={() => toggle(id)} title={`Plug ${partName(id)} back in`}>{id.startsWith("m") ? `MLP ${id.slice(1)}` : id} <span aria-hidden="true">×</span></button>)}
                <button className="plug-reset" onClick={() => setOff([])}>Plug all back in</button>
              </div>
            )}
            {result && current && off.length > 0 && (
              <>
                <p className="effect-headline">{headline(result.items, result.now)}</p>
                <ul className="effect-list">
                  {result.items.map((c, i) => (
                    <li key={i}>
                      <span className="effect-piece">{show(c.piece)}</span>
                      <span className="effect-bar"><span className="effect-before" style={{ width: `${c.before * 100}%` }} /><span className="effect-after" style={{ width: `${c.after * 100}%` }} /></span>
                      <span className="effect-nums">{pct(c.before)} → <strong className={c.after < c.before ? "down" : "up"}>{pct(c.after)}</strong></span>
                    </li>
                  ))}
                </ul>
                <p className="muted small">Small changes are normal: when a part is switched off, others often step in. Researchers call this self-repair.</p>
                {[...off].sort().join(",") === [...NAME_MOVERS].sort().join(",") && prompt === IOI_PROMPT && (
                  <button className="secondary" onClick={() => setOff([...NAME_MOVERS, ...BACKUPS])}>Now unplug the backup name movers too</button>
                )}
              </>
            )}
          </section>
        </aside>
      </div>
    </>
  );
}

function headline(items: { piece: string; before: number; after: number }[], now: string) {
  const was = items[0].piece;
  return now === was
    ? `Top guess is still “${word(was)}”, at ${pct(items[0].after)} instead of ${pct(items[0].before)}.`
    : `Top guess changed from “${word(was)}” to “${word(now)}”.`;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <div className="table-row-label">{label}</div>
      {children}
    </>
  );
}
