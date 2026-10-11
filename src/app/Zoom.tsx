// Exhibit 6: zoom in, from the answer down to single numbers.

import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { href, navigate, type Route, type ZoomPart } from "./route.ts";
import type { Writer } from "./useWriter.ts";
import { ZoomAnswer } from "./ZoomAnswer.tsx";
import { ZoomHead } from "./ZoomHead.tsx";
import { ZoomMap } from "./ZoomMap.tsx";
import { ZoomMlp } from "./ZoomMlp.tsx";
import { ZoomStream } from "./ZoomStream.tsx";
import { ZoomWeights } from "./ZoomWeights.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  part?: ZoomPart;
  arg?: string;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

// Ordered from the whole model down to its smallest pieces, then the fixed parts.
const LEVELS: { part?: ZoomPart; arg?: string; name: string; scale: string; title: string; dek: string }[] = [
  { name: "The answer", scale: "whole model", title: "Start at the answer", dek: "Which of GPT-2's 156 parts pushed it toward its answer, and what those parts relied on." },
  { part: "stream", name: "The stream", scale: "768 numbers", title: "The running summary, number by number", dek: "Every token carries 768 numbers from layer to layer. Here they all are." },
  { part: "head", arg: "9.9", name: "A head", scale: "64 numbers", title: "Inside an attention head", dek: "A question, a label for every token, a blend of values: attention worked out step by step, down to the multiplications." },
  { part: "mlp", arg: "7", name: "An MLP", scale: "3,072 neurons", title: "Inside an MLP", dek: "Each MLP is 3,072 neurons. See which fire, what they fire on across Wikipedia, and one neuron worked out by hand." },
  { part: "weights", name: "The weights", scale: "fixed", title: "The weights themselves", dek: "What GPT-2 learned once and uses for every prompt: how it marks position, which heads copy, and how heads are wired together." },
  { part: "map", name: "Token map", scale: "50,257 tokens", title: "A map of every token", dek: "All the tokens GPT-2 knows, laid out by similarity, with your prompt and its guesses on top." },
];

export function Zoom({ engine, tokenizer, weights, part, arg, prompt, setPrompt, shuffle, writer }: Props) {
  const level = LEVELS.find((l) => l.part === part) ?? LEVELS[0];
  const shared = { engine, tokenizer, prompt, setPrompt, shuffle, writer };
  const [ml, mn] = (part === "mlp" && arg ? arg : "7").split(".").map(Number);
  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Exhibit 6 · Zoom</div>
        <h1>{level.title}</h1>
        <p className="dek">{level.dek}</p>
      </header>
      <nav className="zoom-nav" aria-label="Zoom levels">
        <span className="zoom-scale" aria-hidden="true"><span>zoomed out</span><span>zoomed in</span><span>fixed</span></span>
        <ol>
          {LEVELS.map((l) => (
            <li key={l.name}>
              <a href={href({ view: "zoom", part: l.part, arg: l.part === part ? arg : l.arg } as Route)} className={l === level ? "zoom-level zoom-level-on" : "zoom-level"} aria-current={l === level ? "page" : undefined}>
                <strong>{l.name}</strong><span>{l.scale}</span>
              </a>
            </li>
          ))}
        </ol>
      </nav>
      {part === "head" ? (
        <ZoomHead {...shared} weights={weights} id={arg ?? "9.9"} onHead={(id) => navigate(href({ view: "zoom", part: "head", arg: id }), true)} />
      ) : part === "mlp" ? (
        <ZoomMlp {...shared} weights={weights} layer={ml} neuron={Number.isInteger(mn) ? mn : null}
          onPick={(l, n) => navigate(href({ view: "zoom", part: "mlp", arg: n === null ? `${l}` : `${l}.${n}` }), true)} />
      ) : part === "stream" ? (
        <ZoomStream {...shared} />
      ) : part === "weights" ? (
        <ZoomWeights engine={engine} tokenizer={tokenizer} weights={weights} />
      ) : part === "map" ? (
        <ZoomMap {...shared} />
      ) : (
        <ZoomAnswer {...shared} weights={weights} />
      )}
    </article>
  );
}
