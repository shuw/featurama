// Exhibit 3: the logit lens. Predict, poke, explain, challenge.

import { useEffect, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { analyse, readingLabel, word, type Analysis } from "./analysis.ts";
import { runPrompt, tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { Signal } from "./Signal.tsx";

const DEFAULT = "The first president of the United States was George";

const CHALLENGES = [
  {
    id: "layers.flipflop",
    title: "Indecisive",
    text: "Find a prompt where the top guess changes at least 4 times on the way through.",
    check: (a: Analysis) => a.changes >= 4,
  },
  {
    id: "layers.late",
    title: "Last-minute answer",
    text: "Find a prompt whose final answer only becomes the top guess in the last two layers.",
    check: (a: Analysis) => a.first >= a.lens.length - 2,
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
  const [position, setPosition] = useState(tokens.length - 1);
  const [result, setResult] = useState<Awaited<ReturnType<typeof runPrompt>> | null>(null);
  const [predicting, setPredicting] = useState(!done("layers.predict"));
  const [guess, setGuess] = useState<number | null>(null);
  const [defaultAnswer, setDefaultAnswer] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, rerender] = useState(0);
  const runId = useRef(0);

  useEffect(() => {
    const id = ++runId.current;
    const handle = setTimeout(async () => {
      try {
        const r = await runPrompt(engine, tokens);
        if (id !== runId.current) return;
        setResult(r);
        setError(null);
      } catch (e) {
        if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
      }
    }, 120);
    return () => clearTimeout(handle);
  }, [engine, tokens]);

  const analysis = result && result.trace.tokens === tokens ? analyse(result.trace, result.finals, engine.config, tokenizer, Math.min(position, tokens.length - 1)) : null;

  useEffect(() => {
    if (analysis && text === DEFAULT && position === tokens.length - 1 && !defaultAnswer) setDefaultAnswer(analysis);
    if (analysis && text !== DEFAULT && position === tokens.length - 1) {
      for (const c of CHALLENGES) if (c.check(analysis) && !done(c.id)) {
        markDone(c.id);
        rerender((n) => n + 1);
      }
    }
  });

  const edit = (value: string) => {
    setText(value);
    const t = tokenize(tokenizer, value);
    setTokens(t);
    setPosition(t.length - 1);
  };

  const reveal = () => {
    markDone("layers.predict");
    setPredicting(false);
  };

  const shown = predicting ? null : analysis;
  const pieces = tokens.map((id, i) => (i === 0 ? "⟨start⟩" : tokenizer.piece(id)));

  return (
    <article className="exhibit">
      <header className="exhibit-head">
        <div className="eyebrow">Exhibit 3 · Layers</div>
        <h1>Watch the guess take shape</h1>
      </header>

      <label className="prompt">
        <span className="prompt-label">Prompt</span>
        <input value={text} onChange={(e) => edit(e.target.value)} disabled={predicting} spellCheck={false} maxLength={300} />
        <span className="prompt-count">{tokens.length - 1} tokens</span>
      </label>

      <Signal layers={L} pieces={pieces} analysis={shown} onSelect={setPosition} />
      {error && <p className="error">{error}</p>}

      <div className="beats">
        {predicting ? (
          <section className="beat beat-predict">
            <div className="beat-label">Predict</div>
            {defaultAnswer ? (
              <>
                <p className="question">GPT-2 will finish this prompt with “{word(defaultAnswer.final)}”. Reading its mind layer by layer, when do you think that guess first comes out on top?</p>
                <div className="chips" role="group" aria-label="Your guess">
                  {Array.from({ length: L + 1 }, (_, k) => (
                    <button key={k} className={guess === k ? "chip chip-on" : "chip"} onClick={() => setGuess(k)} aria-pressed={guess === k}>
                      {k === 0 ? "start" : k - 1}
                    </button>
                  ))}
                </div>
                <div className="beat-actions">
                  <span className="hint">{guess === null ? "Pick a layer: “start” means before any layer has run." : `Your guess: ${readingLabel(guess)}.`}</span>
                  <button className="primary" disabled={guess === null} onClick={reveal}>Run it</button>
                </div>
              </>
            ) : (
              <p className="question">Getting GPT-2 ready…</p>
            )}
          </section>
        ) : (
          <>
            {guess !== null && defaultAnswer && text === DEFAULT && (
              <section className="beat">
                <div className="beat-label">Your guess</div>
                <p>{verdict(guess, defaultAnswer)}</p>
              </section>
            )}
            {analysis && (
              <section className="beat beat-explain">
                <div className="beat-label">What you're seeing</div>
                <p>Each token rides its own line through the model, carrying a running summary that researchers call the <strong>residual stream</strong>. Every layer reads that summary and adds to it: attention (orange arcs) pulls in information from earlier tokens (grey dashes are attention parked on the start token, where heads rest when they have nothing to fetch), and the MLP (teal dots) works on each token by itself.</p>
                <p>The <strong>logit lens</strong> asks what the model would say if it had to answer right now. It takes the summary part-way through and applies the model's own final step to it early. Here, for “{word(analysis.pieces[analysis.position])}”, the final answer “{word(analysis.final)}” first comes out on top {readingLabel(analysis.first)}{analysis.changes > 0 ? `, and the top guess changes ${analysis.changes} time${analysis.changes === 1 ? "" : "s"} along the way` : ""}.</p>
                <p className="caveat">The lens is a rough window. Early layers were never trained to be read this way, so their guesses are mostly noise, and some knowledge stays hidden from it until late.</p>
              </section>
            )}
            <section className="beat">
              <div className="beat-label">Challenges</div>
              {CHALLENGES.map((c) => (
                <div key={c.id} className="challenge">
                  <Ring full={done(c.id)} />
                  <div><strong>{c.title}.</strong> {c.text}</div>
                </div>
              ))}
            </section>
          </>
        )}
        {guide && (
          <aside className="guide">
            <div className="beat-label">Guide</div>
            <p>{predicting ? "Guess first, then run it. Being wrong is half the fun: it tells you what to look at." : "Type your own prompt. Click a token on the left to read the lens there, and hover a lens square for its top five guesses."}</p>
          </aside>
        )}
      </div>
    </article>
  );
}

function verdict(guess: number, a: Analysis) {
  const before = a.lens.slice(1, a.first).map((r) => r.piece);
  const common = mostCommon(before);
  const actual = `The lens first showed “${word(a.final)}” ${readingLabel(a.first)}`;
  const lead = guess === a.first ? "Spot on. " : `You said ${readingLabel(guess)}. `;
  return lead + actual + (common && common !== a.final ? `; before that it mostly guessed “${word(common)}”.` : ".");
}

function mostCommon(xs: string[]) {
  const counts = new Map<string, number>();
  for (const x of xs) counts.set(x, (counts.get(x) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

function Ring({ full }: { full: boolean }) {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-label={full ? "Done" : "Not yet"} role="img" className="ring">
      <circle cx="14" cy="14" r="11" className="ring-track" />
      {full && <circle cx="14" cy="14" r="11" className="ring-fill" />}
    </svg>
  );
}
