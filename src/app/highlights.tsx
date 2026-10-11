// Chips that outline the heads doing one kind of job, on either head grid.

import { GLOSSARY } from "./glossary.ts";
import { jobsOf, type HeadJobs, type Job } from "./heads.ts";

interface Highlight {
  id: string;
  label: string;
  tip: string;
  measured?: Job; // also include heads measured in the browser doing this
  heads: string[];
}

const BACKUPS = ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"];

export const HIGHLIGHTS: Highlight[] = [
  { id: "previous", label: "Previous token", tip: "Look at the token just before. Measured in your browser, plus heads named by researchers.", measured: "previous", heads: ["2.2", "4.11"] },
  { id: "duplicate", label: "Duplicate token", tip: "Look back at earlier copies of the same token. Measured in your browser, plus heads named by researchers.", measured: "duplicate", heads: ["0.1", "3.0", "0.10"] },
  { id: "induction", label: "Induction", tip: "Spot a repeat and look at what came next last time. Measured in your browser, plus heads named by researchers.", measured: "induction", heads: ["5.1", "5.5", "6.9", "7.2", "7.10"] },
  { id: "movers", label: "Name movers", tip: "Copy the right name into the answer in sentences like the Mary-and-John one, with their backups (Wang et al. 2022).", heads: ["9.6", "9.9", "10.0", ...BACKUPS] },
  { id: "inhibitors", label: "Name inhibitors", tip: "Tell the name movers which name to skip (Wang et al. 2022).", heads: ["7.3", "7.9", "8.6", "8.10"] },
  { id: "copy", label: "Copy suppression", tip: "Stop the model repeating a word already in the text (McDougall et al. 2023).", heads: ["10.7", "11.10"] },
  { id: "successor", label: "Successor", tip: "Predict the next item in a sequence: Monday → Tuesday (Lan et al. 2023).", heads: ["9.1"] },
  { id: "years", label: "Year readers", tip: "In “from 1732 to 17…”, read the start year so the model picks a later one (Hanna et al. 2023).", heads: ["5.1", "5.5", "6.9", "7.10", "8.8", "8.11", "9.1"] },
  { id: "acronyms", label: "Acronym letters", tip: "Write the right capital letter when finishing an acronym (García-Carrasco et al. 2024).", heads: ["8.11", "9.9", "10.10", "11.4"] },
];

export function highlighted(id: string | null, jobs: HeadJobs, L: number, H: number): Set<string> {
  const h = HIGHLIGHTS.find((x) => x.id === id);
  if (!h) return new Set();
  const out = new Set(h.heads);
  if (h.measured) for (let l = 0; l < L; l++) for (let k = 0; k < H; k++) if (jobsOf(jobs, l, k, H).includes(h.measured)) out.add(`${l}.${k}`);
  return out;
}

export function HighlightChips({ value, onChange, compact = false }: { value: string | null; onChange: (id: string | null) => void; compact?: boolean }) {
  return (
    <div className={compact ? "hl-chips compact" : "hl-chips"} role="group" aria-label="Highlight heads by job">
      {!compact && <span className="hl-label" title={GLOSSARY.highlight}>Highlight</span>}
      {HIGHLIGHTS.map((h) => (
        <button key={h.id} className={value === h.id ? "hl-chip hl-on" : "hl-chip"} aria-pressed={value === h.id} title={h.tip}
          onClick={() => onChange(value === h.id ? null : h.id)}>{h.label}</button>
      ))}
    </div>
  );
}
