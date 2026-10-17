// You vs GPT-2: guess the next word of everyday sentences, then see how
// GPT-2 did on the same sentence.

import { useEffect, useRef, useState } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { show, word } from "./analysis.ts";
import { tokenize } from "./model.ts";
import { done, markDone } from "./progress.ts";
import { pct } from "./PromptBox.tsx";

// Everyday sentences with the word that really comes next. Every answer is a
// single GPT-2 token, so both players are scored on the same thing.
const SENTENCES: [string, string][] = [
  ["She opened the fridge and took out a carton of", "milk"],
  ["The train was delayed by twenty", "minutes"],
  ["He brushed his teeth and went to", "bed"],
  ["The museum is closed on", "Mondays"],
  ["I can't find my keys anywhere. I think I left them in the", "car"],
  ["The recipe calls for two cups of flour and one cup of", "sugar"],
  ["Please remember to turn off the", "lights"],
  ["The cat jumped onto the kitchen", "counter"],
  ["We watched the sun set over the", "ocean"],
  ["My phone battery is almost", "dead"],
  ["The doctor told him to drink more", "water"],
  ["She won first prize in the science", "fair"],
  ["The library asks visitors to keep their voices", "down"],
  ["He paid for the coffee with his credit", "card"],
  ["The children built a castle out of", "sand"],
  ["Can you pass me the salt and", "pepper"],
  ["The movie was so boring that I fell", "asleep"],
  ["Every morning she goes for a run in the", "park"],
  ["Don't forget to water the", "plants"],
  ["He knocked on the door, but nobody", "answered"],
  ["The concert was cancelled because of the", "weather"],
  ["She whispered so quietly that nobody could", "hear"],
  ["They adopted a puppy from the local animal", "shelter"],
  ["The teacher asked the students to open their", "books"],
  ["Turn left at the next traffic", "light"],
  ["He put on his coat because it was", "cold"],
  ["The bus was so crowded that we had to", "stand"],
  ["She took a deep breath and jumped into the", "water"],
  ["After dinner, we washed the", "dishes"],
  ["The old man walked slowly with a", "cane"],
  ["We need to buy some eggs, bread and", "milk"],
];
const ROUNDS = 10;

interface Reveal {
  top: { piece: string; p: number }[];
  pAnswer: number;
  pGuess: number;
  youRight: boolean;
  gptRight: boolean;
}

const norm = (s: string) => s.trim().toLowerCase().replace(/[^\p{L}\p{N}']/gu, "");

function deal() {
  return SENTENCES.map((s) => [Math.random(), s] as const).sort((a, b) => a[0] - b[0]).slice(0, ROUNDS).map(([, s]) => s);
}

export function Game({ engine, tokenizer }: { engine: GpuEngine; tokenizer: Tokenizer }) {
  const L = engine.config.n_layers;
  const [deck, setDeck] = useState(deal);
  const [round, setRound] = useState(0);
  const [guess, setGuess] = useState("");
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [score, setScore] = useState({ you: 0, gpt: 0 });
  const [thinking, setThinking] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const next = useRef<HTMLButtonElement>(null);
  const over = round >= ROUNDS;
  const [prefix, answer] = deck[Math.min(round, ROUNDS - 1)];

  useEffect(() => {
    (reveal ? next : input).current?.focus();
  }, [reveal, round]);

  const submit = async () => {
    if (!norm(guess) || reveal || thinking) return;
    setThinking(true);
    const tokens = tokenize(tokenizer, prefix);
    const answerId = tokenizer.encode(" " + answer)[0];
    const guessId = tokenizer.encode(" " + guess.trim())[0];
    const [trace] = await Promise.all([
      engine.run({ tokens, lens: { positions: [tokens.length - 1], topK: 5, targets: [answerId, guessId] } }),
      new Promise((r) => setTimeout(r, 450)), // a beat for GPT-2 to "think"
    ]);
    const lens = trace.lens!;
    const top = Array.from({ length: 5 }, (_, k) => ({ piece: tokenizer.piece(lens.ids[L * 5 + k]), p: Math.exp(lens.logprobs[L * 5 + k]) }));
    const r: Reveal = {
      top,
      pAnswer: Math.exp(lens.targetLogprobs[L * 2]),
      pGuess: Math.exp(lens.targetLogprobs[L * 2 + 1]),
      youRight: norm(guess) === norm(answer),
      gptRight: norm(top[0].piece) === norm(answer),
    };
    setScore((s) => ({ you: s.you + (r.youRight ? 1 : 0), gpt: s.gpt + (r.gptRight ? 1 : 0) }));
    setReveal(r);
    setThinking(false);
  };

  const advance = () => {
    setReveal(null);
    setGuess("");
    setRound((n) => n + 1);
    if (round + 1 >= ROUNDS && !done("play.finished")) markDone("play.finished");
  };

  const again = () => {
    setDeck(deal());
    setRound(0);
    setScore({ you: 0, gpt: 0 });
    setReveal(null);
    setGuess("");
  };

  return (
    <article className="page">
      <header className="page-head">
        <h1>You vs GPT-2</h1>
        <p className="dek">Guess the word that comes next. GPT-2 plays the same sentences. One point for each of you that gets the real word.</p>
      </header>

      <div className="scoreboard" aria-live="polite">
        <div className="score score-you"><span className="score-name">You</span><span className="score-num" key={`y${score.you}`}>{score.you}</span></div>
        <div className="score-round">{over ? "Final score" : `Round ${round + 1} of ${ROUNDS}`}</div>
        <div className="score score-gpt"><span className="score-num" key={`g${score.gpt}`}>{score.gpt}</span><span className="score-name">GPT-2</span></div>
      </div>

      {over ? (
        <section className="card game-card game-over">
          <p className="game-verdict">{score.you > score.gpt ? "You win!" : score.you < score.gpt ? "GPT-2 wins." : "A draw."}</p>
          <p>You {score.you}, GPT-2 {score.gpt}.</p>
          <p className="muted">These are everyday sentences, which suit people. On ordinary web text, where many words are hard to guess, studies find language models predict the next word better than people do (Shlegeris et al. 2022).</p>
          <button className="primary" onClick={again}>Play again</button>
        </section>
      ) : (
        <section className="card game-card">
          <p className="game-sentence">
            {prefix}{" "}
            {reveal ? <span className="game-answer">{answer}</span> : <span className="game-blank">{guess.trim() || "_____"}</span>}
          </p>
          {!reveal ? (
            <form className="game-form" onSubmit={(e) => { e.preventDefault(); submit(); }}>
              <input ref={input} value={guess} onChange={(e) => setGuess(e.target.value)} placeholder="Your guess" aria-label="Your guess for the next word" maxLength={30} autoComplete="off" spellCheck={false} />
              <button className="primary" disabled={!norm(guess) || thinking}>{thinking ? "GPT-2 is thinking…" : "Guess"}</button>
            </form>
          ) : (
            <div className="game-reveal">
              <div className={reveal.youRight ? "verdict-row right" : "verdict-row"}>
                <span className="who">You</span>
                <span>“{guess.trim()}” {reveal.youRight ? "is right! +1" : "isn't it."}</span>
                <span className="muted">GPT-2 thought your word was {pct(reveal.pGuess)} likely.</span>
              </div>
              <div className={reveal.gptRight ? "verdict-row right" : "verdict-row"}>
                <span className="who">GPT-2</span>
                <span>“{word(reveal.top[0].piece)}” {reveal.gptRight ? "is right! +1" : "isn't it."}</span>
                <span className="muted">It gave the real word {pct(reveal.pAnswer)}.</span>
              </div>
              <div className="game-top">
                <div className="card-label">GPT-2's top five</div>
                {reveal.top.map((t, i) => (
                  <div key={i} className={norm(t.piece) === norm(answer) ? "answer-bar answer-a" : "answer-bar answer-b"}>
                    <span className="answer-word">{show(t.piece)}</span>
                    <span className="answer-track"><span style={{ width: `${t.p * 100}%` }} /></span>
                    <span className="answer-p">{pct(t.p)}</span>
                  </div>
                ))}
              </div>
              <button ref={next} className="primary" onClick={advance}>{round + 1 >= ROUNDS ? "See the final score" : "Next sentence"}</button>
            </div>
          )}
        </section>
      )}
    </article>
  );
}
