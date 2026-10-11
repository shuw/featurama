// The first-visit download, used as a short tour: the machine fills in as its
// weights arrive (the word table first, then layers 0 to 11), with a card on
// each part as it lands.

import type { LoadProgress } from "../engine/load.ts";

// Share of the download that is the token and position tables (50,257 + 1,024
// rows of 768 fp16 numbers, out of 124 million parameters).
const TABLES = (50257 + 1024) * 768 / 124.4e6;
const LAYERS = 12;

const CARDS: { at: number; title: string; text: string }[] = [
  { at: 0, title: "Tokens", text: "GPT-2 reads text in pieces called tokens. “featurama” is three: fe · atur · ama." },
  { at: 0.12, title: "768 numbers per token", text: "Each token is looked up in a table of 50,257 and becomes 768 numbers. That table is what's arriving first." },
  { at: TABLES, title: "The residual stream", text: "Those numbers flow through 12 layers. Each layer reads them and adds something back." },
  { at: TABLES + (1 - TABLES) * 0.25, title: "Attention heads", text: "Each layer has 12 attention heads, the only parts that move information between tokens. One looks at the previous word; another copies names." },
  { at: TABLES + (1 - TABLES) * 0.55, title: "MLPs", text: "After its heads, each layer runs an MLP: it works on each token by itself, adding what it knows about that token in context." },
  { at: TABLES + (1 - TABLES) * 0.85, title: "The guess", text: "At the end, the 768 numbers are scored against all 50,257 tokens. The best score is GPT-2's next word, and you can watch it change, unplug parts, and steer it." },
];

export function Loading({ progress }: { progress: LoadProgress | null }) {
  const frac = progress ? progress.loaded / progress.total : 0;
  const cached = progress?.fromCache;
  const layersDone = frac <= TABLES ? 0 : ((frac - TABLES) / (1 - TABLES)) * LAYERS;
  const card = [...CARDS].reverse().find((c) => frac >= c.at) ?? CARDS[0];
  const label = !progress
    ? "Starting…"
    : cached
      ? "Loading GPT-2 from this browser's cache…"
      : `Downloading GPT-2: ${(progress.loaded / 1e6).toFixed(0)} of ${(progress.total / 1e6).toFixed(0)} MB. Only the first time; afterwards it loads from your browser.`;
  return (
    <div className="loading-tour">
      <svg className="tour-machine" viewBox="0 0 330 200" role="img" aria-label={`GPT-2 arriving: ${Math.round(frac * 100)}%`}>
        <rect x="0" y="6" width="34" height="164" rx="4" className="tour-table" />
        <rect x="0" y={6 + 164 * (1 - Math.min(1, frac / TABLES))} width="34" height={164 * Math.min(1, frac / TABLES)} rx="4" className="tour-table-fill" />
        <text x="17" y="190" textAnchor="middle" className="tour-axis">words</text>
        {Array.from({ length: LAYERS }, (_, l) => {
          const f = Math.max(0, Math.min(1, layersDone - l));
          return (
            <g key={l} transform={`translate(${46 + l * 23} 6)`}>
              {Array.from({ length: 12 }, (_, h) => (
                <rect key={h} y={h * 12} width="19" height="10" rx="2" className="tour-head" style={{ opacity: 0.12 + 0.88 * (f > h / 12 ? 1 : 0) }} />
              ))}
              <rect y={148} width="19" height="16" rx="2" className="tour-mlp" style={{ opacity: f >= 1 ? 1 : 0.15 }} />
            </g>
          );
        })}
        <text x="46" y="190" className="tour-axis">layer 0 →</text>
        <text x="321" y="190" textAnchor="end" className="tour-axis">11</text>
      </svg>
      <div className="tour-card" key={card.title} aria-live="polite">
        <div className="card-label">{card.title}</div>
        <p>{card.text}</p>
      </div>
      <div className="progress">
        <div className="progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)} aria-label="Loading GPT-2">
          <div style={{ width: `${frac * 100}%` }} />
        </div>
        <div className="progress-label">{label}</div>
      </div>
    </div>
  );
}
