// The area under an exhibit's experiment: the main beats on the left, the
// challenges and a hint about what to try next in a column on the right.

import type { ReactNode } from "react";

export function Beats({ children, side, guide }: { children: ReactNode; side?: ReactNode; guide?: string | false }) {
  return (
    <div className="beats">
      <div className="beats-main">{children}</div>
      <div className="beats-side">
        {guide && (
          <aside className="guide">
            <div className="card-label">Guide</div>
            <p>{guide}</p>
          </aside>
        )}
        {side}
      </div>
    </div>
  );
}
