// What the letters on the head grids mean, and where each one comes from.

import { GLOSSARY } from "./glossary.ts";
import { JOB_TEXT, ROLES, jobsOf, IOI_ROLE, type HeadJobs } from "./heads.ts";
import { Tip } from "./Tip.tsx";

export function JobLegend({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "job-legend compact" : "job-legend"}>
      <div className="jl-group">
        <Tip tip={GLOSSARY.measured} align="start"><span className="jl-title">Tested in your browser just now</span></Tip>
        {Object.values(JOB_TEXT).map((j) => (
          <div key={j.letter} className="jl-row"><span className="badge">{j.letter}</span><span>{j.text}</span></div>
        ))}
      </div>
      <div className="jl-group">
        <Tip tip={GLOSSARY.research} align="start"><span className="jl-title">Found by researchers, in the Mary-and-John sentence</span></Tip>
        {Object.values(ROLES).map((r) => (
          <div key={r.letter} className="jl-row"><span className="badge badge-research">{r.letter}</span><span>{r.text}</span></div>
        ))}
      </div>
    </div>
  );
}

// The badges shown on one head's square.
export function HeadBadges({ jobs, l, h, H, max = 2 }: { jobs: HeadJobs; l: number; h: number; H: number; max?: number }) {
  const measured = jobsOf(jobs, l, h, H).map((j) => JOB_TEXT[j].letter);
  const role = IOI_ROLE[`${l}.${h}`];
  const all = [...measured.map((t) => ({ t, research: false })), ...(role ? [{ t: role, research: true }] : [])].slice(0, max);
  if (!all.length) return null;
  return (
    <span className="badges" aria-hidden="true">
      {all.map((b) => <span key={b.t} className={b.research ? "badge badge-research" : "badge"}>{b.t}</span>)}
    </span>
  );
}
