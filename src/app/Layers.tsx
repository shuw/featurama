// Exhibit 3: the logit lens. Predict, poke, explain, challenge.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { analyse, predictions, readingLabel, word, type Analysis } from "./analysis.ts";
import { Beats } from "./Beats.tsx";
import { runPrompt, tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { promptFromUrl } from "./route.ts";
import { PromptBox } from "./PromptBox.tsx";
import type { Writer } from "./useWriter.ts";
import { Ring } from "./Ring.tsx";
import { Signal } from "./Signal.tsx";
import { GLOSSARY } from "./glossary.ts";
import { Term } from "./Tip.tsx";
import { useRun } from "./useRun.ts";

// The predict beat asks about this sentence; afterwards the page uses the shared prompt.
const PREDICT_PROMPT = "The first president of the United States was George";

const CHALLENGES = [
  {
    id: "layers.flipflop",
    title: "Indecisive",
    text: "Find a prompt where the top guess changes at least 4 times, among at least 3 different words.",
    check: (a: Analysis) => a.changes >= 4 && a.distinct >= 3,
  },
  {
    id: "layers.late",
    title: "Last-minute answer",
    text: "Find a prompt whose final answer only takes the lead for good in the last two layers.",
    check: (a: Analysis) => a.settle >= a.lens.length - 2,
  },
];

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  guide: boolean;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

export function Layers({ engine, tokenizer, guide, prompt, setPrompt, shuffle, writer }: Props) {
  const L = engine.config.n_layers;
  const [override, setOverride] = useState<string | null>(() => (done("layers.predict") || promptFromUrl() !== null ? null : PREDICT_PROMPT));
  const text = override ?? prompt;
  const tokens = useMemo(() => tokenize(tokenizer, text), [tokenizer, text]);
  const [position, setPosition] = useState<number | null>(null); // null follows the last token
  const [predicting, setPredicting] = useState(() => !done("layers.predict") && promptFromUrl() === null); // a shared link skips the opening question
  const [guess, setGuess] = useState<number | null>(null);
  const [defaultAnswer, setDefaultAnswer] = useState<Analysis | null>(null);
  const [, rerender] = useState(0);
  const run = useCallback((t: number[]) => runPrompt(engine, t), [engine]);
  const { result, error, version } = useRun(tokens, run);

  const shown = result?.trace.tokens ?? tokens;
  const last = shown.length - 1;
  const pos = position === null ? last : Math.min(position, last);
  const analysis = result ? analyse(result.trace, result.finals, engine.config, tokenizer, pos) : null;
  const current = !!result && result.trace.tokens === tokens;

  useEffect(() => {
    if (!analysis || !current) return;
    if (text === PREDICT_PROMPT && pos === last && !defaultAnswer) setDefaultAnswer(analysis);
    if (text !== PREDICT_PROMPT && pos === last && !predicting) {
      for (const c of CHALLENGES) if (c.check(analysis) && !done(c.id)) {
        markDone(c.id);
        rerender((n) => n + 1);
      }
    }
  });

  const edit = (value: string) => {
    setOverride(null);
    setPrompt(value);
    setPosition(null);
  };

  const reveal = () => {
    markDone("layers.predict");
    setPredicting(false);
  };

  const pieces = shown.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Exhibit 3 · Layers</div>
        <h1>Watch the guess take shape</h1>
        <p className="dek">GPT-2 builds its answer one layer at a time. The <Term tip={GLOSSARY.lens}>logit lens</Term> lets you read its best guess part-way through.</p>
      </header>

      <PromptBox autoFocus writer={writer} onShuffle={() => { setOverride(null); shuffle(); }} value={text} onChange={edit} tokens={tokens} tokenizer={tokenizer} next={result ? predictions(result.trace, L, tokenizer) : null}
        stale={!current} disabled={predicting} disabledNote="Make your guess below first" />

      <div className={!current && result ? "stage stale" : "stage"}>
        <Signal layers={L} pieces={pieces} analysis={predicting ? null : analysis} onSelect={setPosition} version={version} />
      </div>
      {error && <p className="error">{error}</p>}

      <Beats
        guide={guide && (predicting
          ? "Guess first, then run it. A wrong guess is useful: it tells you what to look at."
          : "Type your own prompt and watch the lens update. Click a token on the left to read the lens there, and point at a lens square for its top five guesses.")}
        side={!predicting && (
          <section className="card">
            <div className="card-label">Challenges</div>
            {CHALLENGES.map((c) => (
              <div key={c.id} className="challenge">
                <Ring full={done(c.id)} />
                <div><strong>{c.title}.</strong> {c.text}</div>
              </div>
            ))}
          </section>
        )}
      >
        {predicting ? (
          <section className="card card-predict">
            <div className="card-label">Predict</div>
            {defaultAnswer ? (
              <>
                <p className="question">GPT-2 will finish this prompt with “{word(defaultAnswer.final)}”. If you read its best guess after each layer, after which layer does “{word(defaultAnswer.final)}” take the lead for good?</p>
                <div className="chips" role="group" aria-label="Your guess">
                  {Array.from({ length: L + 1 }, (_, k) => (
                    <button key={k} className={guess === k ? "chip chip-on" : "chip"} onClick={() => setGuess(k)} aria-pressed={guess === k}>
                      {k === 0 ? "start" : k - 1}
                    </button>
                  ))}
                </div>
                <div className="card-actions">
                  <span className="hint">{guess === null ? "Pick a layer. “start” means before any layer has run." : `Your guess: ${readingLabel(guess)}.`}</span>
                  <button className="primary" disabled={guess === null} onClick={reveal}>Run it</button>
                </div>
              </>
            ) : (
              <p className="question muted">Getting GPT-2 ready…</p>
            )}
          </section>
        ) : (
          <>
            {guess !== null && defaultAnswer && text === PREDICT_PROMPT && (
              <section className="card card-result">
                <div className="card-label">Your guess</div>
                <p>{verdict(guess, defaultAnswer)}</p>
              </section>
            )}
            {analysis && (
              <section className="card">
                <div className="card-label">What you're seeing</div>
                <p>Each token travels on its own line, carrying a running summary that researchers call the <Term tip={GLOSSARY.residual}><strong>residual stream</strong></Term>. Every layer reads the summary and adds to it. <Term tip={GLOSSARY.head}>Attention</Term> (orange arcs) brings in information from earlier tokens, and the <Term tip={GLOSSARY.mlp}>MLP</Term> (teal dots) works on each token by itself. Grey dashes are attention resting on the <Term tip={GLOSSARY.start}>start token</Term>.</p>
                <p>The <Term tip={GLOSSARY.lens}><strong>logit lens</strong></Term> asks what the model would say if it had to answer right now. It takes the summary part-way through and runs the model's final step on it early. Here, for “{word(analysis.pieces[analysis.position])}”, the final answer “{word(analysis.final)}” takes the lead for good {readingLabel(analysis.settle)}{analysis.changes > 0 ? `, and the top guess changes ${analysis.changes} time${analysis.changes === 1 ? "" : "s"} along the way` : ""}.</p>
                <p className="caveat">The lens is a rough window. Early layers were never trained to be read this way, so their guesses are mostly noise, and some knowledge stays hidden from it until late.</p>
              </section>
            )}
          </>
        )}
      </Beats>
    </article>
  );
}

function verdict(guess: number, a: Analysis) {
  const before = a.lens.slice(1, a.settle).map((r) => r.piece);
  const common = mostCommon(before);
  const actual = `“${word(a.final)}” took the lead for good ${readingLabel(a.settle)}`;
  const lead = guess === a.settle ? "Spot on. " : `You said ${readingLabel(guess)}. `;
  return lead + actual + (common && common !== a.final ? `; before that it mostly guessed “${word(common)}”.` : ".");
}

function mostCommon(xs: string[]) {
  const counts = new Map<string, number>();
  for (const x of xs) counts.set(x, (counts.get(x) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}
