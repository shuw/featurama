import { useEffect, useRef, useState } from "react";

// Runs the model a moment after the input stops changing, and keeps showing
// the last finished result until the next one arrives, so the page doesn't
// flash empty on every keystroke.
// `key` names anything else the run depends on (such as unplugged heads).
export function useRun<R>(tokens: number[], run: (tokens: number[]) => Promise<R>, delay = 150, key = "") {
  const [result, setResult] = useState<R | null>(null);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // counts finished runs, for animations
  const latest = useRef(0);
  const hasResult = result !== null;

  useEffect(() => {
    const id = ++latest.current;
    setPending(true);
    const handle = setTimeout(async () => {
      try {
        const r = await run(tokens);
        if (id !== latest.current) return;
        setResult(r);
        setVersion((v) => v + 1);
        setError(null);
      } catch (e) {
        if (id === latest.current) setError(e instanceof Error ? e.message : String(e));
      }
      if (id === latest.current) setPending(false);
    }, hasResult ? delay : 0);
    return () => clearTimeout(handle);
    // `run` changes along with `key`; tokens and key are what trigger a run.
  }, [tokens, key]);

  return { result, pending, error, version };
}
