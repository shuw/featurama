// Layers: the same layers × tokens, read three ways.

import type { ReactNode } from "react";
import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { Layers } from "./Layers.tsx";
import { href, type LayersMode } from "./route.ts";
import { Stream } from "./Stream.tsx";
import { TokenMap } from "./TokenMap.tsx";
import type { Writer } from "./useWriter.ts";
import { Positions } from "./WeightViews.tsx";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  mode?: LayersMode;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

const MODES: { mode?: LayersMode; label: string }[] = [
  { label: "The guess" },
  { mode: "numbers", label: "The numbers" },
  { mode: "map", label: "The token map" },
];

export function LayersPage({ engine, tokenizer, weights, mode, prompt, setPrompt, shuffle, writer }: Props) {
  const shared = { engine, tokenizer, prompt, setPrompt, shuffle, writer };
  const modes: ReactNode = (
    <nav className="seg layer-modes" aria-label="Ways to read the layers">
      {MODES.map((m) => (
        <a key={m.label} href={href({ view: "layers", mode: m.mode })} className={m.mode === mode ? "seg-on" : ""} aria-current={m.mode === mode ? "page" : undefined}>{m.label}</a>
      ))}
    </nav>
  );
  if (!mode) return <Layers {...shared} modes={modes} />;
  return (
    <article className="page">
      <header className="page-head">
        <h1>{mode === "numbers" ? "The running summary, number by number" : "A map of every token"}</h1>
        <p className="dek">{mode === "numbers"
          ? "Every token carries 768 numbers from layer to layer. Here they all are, with the position vectors GPT-2 adds before the first layer."
          : "All the tokens GPT-2 knows, laid out by similarity, with your prompt and its guesses on top."}</p>
      </header>
      {modes}
      {mode === "numbers" ? (
        <>
          <Stream {...shared} />
          <div className="part-steps"><Positions weights={weights} /></div>
        </>
      ) : <TokenMap {...shared} />}
    </article>
  );
}
