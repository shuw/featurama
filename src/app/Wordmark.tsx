// The wordmark: the last "a" of featurama attends back to the earlier ones,
// the way an induction head looks back at a repeated token. Drawn in the
// monospace face so the letters sit at known positions.

import { useState } from "react";

const CH = 33.6; // IBM Plex Mono advances 0.6 em; the text is 56 units tall
const A = [2, 6, 8].map((i) => (i + 0.5) * CH);

export function Wordmark({ height, loop = false, label = "featurama" }: { height: number; loop?: boolean; label?: string }) {
  const [play, setPlay] = useState(0);
  return (
    <svg key={play} className={loop ? "wordmark wordmark-loop" : "wordmark"} height={height} viewBox="-6 -8 316 140" role="img" aria-label={label}
      onMouseEnter={() => !loop && setPlay((n) => n + 1)}>
      <path className="wm-lane" d="M0 128H303" pathLength={1} />
      <path className="wm-hop wm-near" d={`M${A[2]} 84 C ${A[2]} 50, ${A[1]} 50, ${A[1]} 84`} pathLength={1} />
      <path className="wm-hop wm-far" d={`M${A[2]} 84 C ${A[2]} 8, ${A[0]} 8, ${A[0]} 84`} pathLength={1} />
      <circle className="wm-q" cx={A[2]} cy={84} r={5.5} />
      <text x="0" y="118" className="wm-text">featurama</text>
    </svg>
  );
}

