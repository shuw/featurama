// Exhibit 3: the logit lens. Predict, poke, explain, challenge.

import { useCallback, useEffect, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { analyse, predictions, readingLabel, word, type Analysis } from "./analysis.ts";
import { Beats } from "./Beats.tsx";
import { runPrompt, tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { PromptBox } from "./PromptBox.tsx";
import { Ring } from "./Ring.tsx";
import { Signal } from "./Signal.tsx";
import { useRun } from "./useRun.ts";

const DEFAULT = "The first president of the United States was George";

const CHALLENGES = [
  {
    id: "layers.flipflop",
    title: "Indecisive",
    text: "Find a prompt where the top guess changes at least 4 times, through at least 3 different words.",
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
}

export function Layers({ engine, tokenizer, guide }: Props) {
  const L = engine.config.n_layers;
  const [text, setText] = useState(DEFAULT);
  const [tokens, setTokens] = useState(() => tokenize(tokenizer, DEFAULT));
  const [position, setPosition] = useState<number | null>(null); // null follows the last token
  const [predicting, setPredicting] = useState(!done("layers.predict"));
  const [guess, setGuess] = useState<number | null>(null);
  const [defaultAnswer, setDefaultAnswer] = useState<Analysis | null>(null);
  const [, rerender] = useState(0);
  const run = useCallback((t: number[]) => runPrompt(engine, t), [engine]);
  const { result, error } = useRun(tokens, run);

  const shown = result?.trace.tokens ?? tokens;
  const last = shown.length - 1;
  const pos = position === null ? last : Math.min(position, last);
  const analysis = result ? analyse(result.trace, result.finals, engine.config, tokenizer, pos) : null;
  const current = !!result && result.trace.tokens === tokens;

  useEffect(() => {
    if (!analysis || !current) return;
    if (text === DEFAULT && pos === last && !defaultAnswer) setDefaultAnswer(analysis);
    if (text !== DEFAULT && pos === last && !predicting) {
      for (const c of CHALLENGES) if (c.check(analysis) && !done(c.id)) {
        markDone(c.id);
        rerender((n) => n + 1);
      }
    }
  });

  const edit = (value: string) => {
    setText(value);
    setTokens(tokenize(tokenizer, value));
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
        <p className="dek">GPT-2 builds its answer one layer at a time. The logit lens lets you read its best guess part-way through.</p>
      </header>

      <PromptBox value={text} onChange={edit} tokens={tokens} tokenizer={tokenizer} next={predicting || !result ? null : predictions(result.trace, L, tokenizer)}
        stale={!current} disabled={predicting} disabledNote="Make your guess below first" />

      <div className={!current && result ? "stage stale" : "stage"}>
        <Signal layers={L} pieces={pieces} analysis={predicting ? null : analysis} onSelect={setPosition} />
      </div>
      {error && <p className="error">{error}</p>}

      <Beats
        guide={guide && (predicting
          ? "Guess first, then run it. Being wrong is half the fun: it tells you what to look at."
          : "Type your own prompt and watch the lens update. Click a word on the left to read the lens there; hover a lens square for its top five guesses.")}
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
                <p className="question">GPT-2 will finish this prompt with “{word(defaultAnswer.final)}”. Reading its mind layer by layer, after which layer does that guess take the lead for good?</p>
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
            {guess !== null && defaultAnswer && text === DEFAULT && (
              <section className="card card-result">
                <div className="card-label">Your guess</div>
                <p>{verdict(guess, defaultAnswer)}</p>
              </section>
            )}
            {analysis && (
              <section className="card">
                <div className="card-label">What you're seeing</div>
                <p>Each token rides its own line through the model, carrying a running summary that researchers call the <strong>residual stream</strong>. Every layer reads that summary and adds to it: attention (orange arcs) pulls in information from earlier tokens, and the MLP (teal dots) works on each token by itself. Grey dashes are attention parked on the start token, where heads rest when they have nothing to fetch.</p>
                <p>The <strong>logit lens</strong> asks what the model would say if it had to answer right now. It takes the summary part-way through and applies the model's own final step to it early. Here, for “{word(analysis.pieces[analysis.position])}”, the final answer “{word(analysis.final)}” takes the lead for good {readingLabel(analysis.settle)}{analysis.changes > 0 ? `, and the top guess changes ${analysis.changes} time${analysis.changes === 1 ? "" : "s"} along the way` : ""}.</p>
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
