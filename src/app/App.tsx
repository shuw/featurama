import { useEffect, useState } from "react";
import { Layers } from "./Layers.tsx";
import { useModel } from "./model.ts";

type Theme = "system" | "light" | "dark";

function readPref<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T) ?? fallback;
  } catch {
    return fallback;
  }
}

function writePref(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

export function App() {
  const model = useModel();
  const [theme, setTheme] = useState<Theme>(() => readPref("featurama-theme", "system"));
  const [guide, setGuide] = useState(() => readPref("featurama-guide", "on") === "on");

  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    writePref("featurama-theme", theme);
  }, [theme]);
  useEffect(() => writePref("featurama-guide", guide ? "on" : "off"), [guide]);

  const nextTheme: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 28 28" aria-hidden="true">
            <path d="M0 9 H28 M0 19 H28" className="mark-lane" />
            <path d="M4 9 C 16 9, 12 19, 24 19" className="mark-arc" />
            <circle cx="24" cy="19" r="3.5" className="mark-dot" />
          </svg>
          <span>featurama</span>
        </div>
        <nav>
          <button className="pill" onClick={() => setGuide(!guide)} aria-pressed={guide}>Guide {guide ? "on" : "off"}</button>
          <button className="pill" onClick={() => setTheme(nextTheme[theme])}>Theme: {theme}</button>
        </nav>
      </header>
      <main>
        {model.status === "ready" ? (
          <Layers engine={model.engine} tokenizer={model.tokenizer} guide={guide} />
        ) : model.status === "error" ? (
          <section className="loading">
            <h1>GPT-2 couldn't start</h1>
            <p>{model.message}</p>
          </section>
        ) : (
          <section className="loading">
            <h1>Look inside a real language model</h1>
            <p>Featurama runs GPT-2, a 124-million-parameter language model, right here in your browser, with every one of its inner workings open to read and rewire.</p>
            <Loading progress={model.progress} />
          </section>
        )}
      </main>
    </div>
  );
}

function Loading({ progress }: { progress: { loaded: number; total: number; fromCache: boolean } | null }) {
  const frac = progress ? progress.loaded / progress.total : 0;
  const label = !progress
    ? "Starting…"
    : progress.fromCache
      ? "Loading GPT-2 from this browser's cache…"
      : `Downloading GPT-2: ${(progress.loaded / 1e6).toFixed(0)} of ${(progress.total / 1e6).toFixed(0)} MB. This happens once.`;
  return (
    <div className="progress">
      <div className="progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)}>
        <div style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="progress-label">{label}</div>
    </div>
  );
}
