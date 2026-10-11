import { useEffect, useState } from "react";
import { Attention } from "./Attention.tsx";
import { Layers } from "./Layers.tsx";
import { Machine } from "./Machine.tsx";
import { useModel } from "./model.ts";
import { href, useRoute } from "./route.ts";
import { Wordmark } from "./Wordmark.tsx";

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

const NEXT_THEME: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

export function App() {
  const model = useModel();
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(() => readPref("featurama-theme", "system"));
  const [guide, setGuide] = useState(() => readPref("featurama-guide", "on") === "on");

  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    writePref("featurama-theme", theme);
  }, [theme]);
  useEffect(() => writePref("featurama-guide", guide ? "on" : "off"), [guide]);

  const tab = (view: "machine" | "attention" | "layers", label: string) => (
    <a className={route.view === view ? "tab tab-on" : "tab"} href={href({ view })} aria-current={route.view === view ? "page" : undefined}>{label}</a>
  );

  return (
    <div className="app">
      <header className="top">
        <div className="top-inner">
          <a className="brand" href={href({ view: "machine" })} aria-label="Featurama home">
            <Wordmark height={44} label="featurama" />
          </a>
          <nav className="tabs" aria-label="Sections">
            {tab("machine", "Machine")}
            {tab("attention", "Attention")}
            {tab("layers", "Layers")}
          </nav>
          <div className="utils">
            <button className="switch" role="switch" aria-checked={guide} onClick={() => setGuide(!guide)}>
              <span className="switch-track"><span className="switch-thumb" /></span> Guide
            </button>
            <button className="icon-button" onClick={() => setTheme(NEXT_THEME[theme])} aria-label={`Theme: ${theme}. Change theme`} title={`Theme: ${theme}`}>
              <ThemeIcon theme={theme} />
            </button>
          </div>
        </div>
      </header>
      <main>
        {model.status === "ready" ? (
          route.view === "attention" ? (
            <Attention engine={model.engine} tokenizer={model.tokenizer} jobs={model.jobs} head={route.head} guide={guide} />
          ) : route.view === "layers" ? (
            <Layers engine={model.engine} tokenizer={model.tokenizer} guide={guide} />
          ) : (
            <Machine engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} jobs={model.jobs} />
          )
        ) : (
          <section className="hero">
            <Wordmark height={150} loop />
            {model.status === "error" ? (
              <>
                <h1>GPT-2 couldn't start</h1>
                <p className="dek">{model.message}</p>
              </>
            ) : (
              <>
                <h1>Look inside a real language model</h1>
                <p className="dek">Featurama runs GPT-2, a 124-million-parameter language model, on your own graphics card. Every one of its inner workings is open to read and to rewire.</p>
                <Loading progress={model.progress} />
              </>
            )}
          </section>
        )}
      </main>
      <footer className="foot">
        <span>GPT-2 small (OpenAI, 2019) running on your GPU with WebGPU. Nothing you type leaves your browser.</span>
        <a href="https://github.com/shuw/featurama">Source</a>
      </footer>
    </div>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  if (theme === "light")
    return (
      <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" className="icon">
        <circle cx="10" cy="10" r="4" />
        <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M4 4l1.4 1.4M14.6 14.6 16 16M4 16l1.4-1.4M14.6 5.4 16 4" />
      </svg>
    );
  if (theme === "dark")
    return (
      <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" className="icon">
        <path d="M16 12.5A7 7 0 0 1 7.5 4a7 7 0 1 0 8.5 8.5Z" />
      </svg>
    );
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" className="icon">
      <circle cx="10" cy="10" r="7" />
      <path d="M10 3a7 7 0 0 1 0 14Z" className="icon-fill" />
    </svg>
  );
}

function Loading({ progress }: { progress: { loaded: number; total: number; fromCache: boolean } | null }) {
  const frac = progress ? progress.loaded / progress.total : 0;
  const label = !progress
    ? "Starting…"
    : progress.fromCache
      ? "Loading GPT-2 from this browser's cache…"
      : `Downloading GPT-2: ${(progress.loaded / 1e6).toFixed(0)} of ${(progress.total / 1e6).toFixed(0)} MB. This happens once; afterwards it loads from your browser.`;
  return (
    <div className="progress">
      <div className="progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)} aria-label="Loading GPT-2">
        <div style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="progress-label">{label}</div>
    </div>
  );
}
