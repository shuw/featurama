// "What researchers found" for one head, each claim linked to its source.

import { dashboardUrl, research, SOURCES, type SourceKey } from "./research.ts";

function Source({ k }: { k: SourceKey }) {
  const s = SOURCES[k];
  return <a className="src" href={s.url} target="_blank" rel="noreferrer" title={s.name} aria-label={`Source: ${s.name}`}>source</a>;
}

export function ResearchNotes({ id }: { id: string }) {
  const r = research(id);
  return (
    <div className="research">
      <div className="card-label">What researchers found</div>
      {r.study ? (
        <p>{r.study.text} <span className="muted">Researchers found {r.study.breadth}.</span> <Source k="every" /></p>
      ) : (
        <p className="muted">The study of every head has no description for this one.</p>
      )}
      {r.circuits.length > 0 && (
        <ul>
          {r.circuits.map((c, i) => <li key={i}><strong>{c.role}.</strong> {c.text} <Source k={c.source} /></li>)}
        </ul>
      )}
      <a className="link-small" href={dashboardUrl(id)} target="_blank" rel="noreferrer">See head {id}'s patterns in the survey's dashboard ↗</a>
    </div>
  );
}
