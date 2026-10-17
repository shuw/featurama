// The Lab: every experiment that changes GPT-2 while it runs.

import type { GpuEngine } from "../engine/gpu/engine.ts";
import type { Weights } from "../engine/model.ts";
import type { Tokenizer } from "../engine/tokenizer.ts";
import { LabSteer } from "./LabSteer.tsx";
import { LabTrace } from "./LabTrace.tsx";
import { LabUnplug } from "./LabUnplug.tsx";
import { Patching } from "./Patching.tsx";
import type { LabTool } from "./route.ts";
import type { Writer } from "./useWriter.ts";

interface Props {
  engine: GpuEngine;
  tokenizer: Tokenizer;
  weights: Weights;
  tool: LabTool;
  prompt: string;
  setPrompt: (text: string) => void;
  shuffle: () => void;
  writer: Writer;
}

const HEADS: Record<Exclude<LabTool, "swap">, { title: string; dek: string }> = {
  unplug: { title: "Unplug parts", dek: "Switch off any heads and MLPs and see what GPT-2 says without them. It's the bluntest test of what a part is for, and GPT-2 often repairs itself." },
  steer: { title: "Steer with a feature", dek: "Add a feature's direction to every token as GPT-2 writes, and its writing bends toward that concept." },
  trace: { title: "Trace the answer back", dek: "How every part voted for GPT-2's answer, then which parts it really depended on, found by unplugging them one at a time." },
};

export function Lab({ engine, tokenizer, weights, tool, prompt, setPrompt, shuffle, writer }: Props) {
  if (tool === "swap") return <Patching engine={engine} tokenizer={tokenizer} />;
  const shared = { engine, tokenizer, prompt, setPrompt, shuffle, writer };
  return (
    <article className="page">
      <header className="page-head">
        <h1>{HEADS[tool].title}</h1>
        <p className="dek">{HEADS[tool].dek}</p>
      </header>
      {tool === "unplug" ? <LabUnplug {...shared} weights={weights} /> : tool === "steer" ? <LabSteer {...shared} /> : <LabTrace {...shared} weights={weights} />}
    </article>
  );
}
