// Tooltips that work with a mouse and with the keyboard (Tab to the term).
// <Term> underlines a word in prose; <Tip> wraps anything else.

import { useId, type ReactNode } from "react";

type Align = "center" | "start" | "end";

// `below` is for things near the top of the screen, like the header.
export function Tip({ tip, children, align = "center", below = false, className = "" }: { tip: ReactNode; children: ReactNode; align?: Align; below?: boolean; className?: string }) {
  const id = useId();
  return (
    <span className={`tip-anchor ${className}`} tabIndex={0} aria-describedby={id}>
      {children}
      <span role="tooltip" id={id} className={`tip tip-${align}${below ? " tip-below" : ""}`}>{tip}</span>
    </span>
  );
}

export function Term({ tip, children, align }: { tip: ReactNode; children: ReactNode; align?: Align }) {
  return <Tip tip={tip} align={align} className="term">{children}</Tip>;
}
