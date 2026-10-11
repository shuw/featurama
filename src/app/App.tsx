import { useEffect, useState } from "react";
import { Attention } from "./Attention.tsx";
import { Game } from "./Game.tsx";
import { Layers } from "./Layers.tsx";
import { Machine } from "./Machine.tsx";
import { Patching } from "./Patching.tsx";
import { useModel, type ModelState } from "./model.ts";
import { href, useRoute, type Route } from "./route.ts";
import { randomPhrase } from "./phrases.ts";
import { GLOSSARY } from "./glossary.ts";
import { Tip } from "./Tip.tsx";
import { useWriter } from "./useWriter.ts";
import { Wordmark } from "./Wordmark.tsx";

type Theme = "light" | "dark";

const TABS: { view: Route["view"]; label: string; key: string }[] = [
  { view: "machine", label: "Machine", key: "1" },
  { view: "attention", label: "Attention", key: "2" },
  { view: "layers", label: "Layers", key: "3" },
  { view: "patching", label: "Patching", key: "4" },
  { view: "play", label: "Play", key: "5" },
];

function readPref(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writePref(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

export function App() {
  const model = useModel();
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(() => (readPref("featurama-theme") === "dark" ? "dark" : "light"));
  const [guide, setGuide] = useState(() => readPref("featurama-guide") !== "off");
  // One prompt for every page, so switching pages keeps what you typed. It
  // starts as a random phrase; once you edit it, it's yours across reloads.
  const [prompt, setPromptState] = useState(() => readPref("featurama-prompt") ?? randomPhrase());
  const [showKeys, setShowKeys] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writePref("featurama-theme", theme);
  }, [theme]);
  useEffect(() => writePref("featurama-guide", guide ? "on" : "off"), [guide]);

  // ⌥1–⌥5 (Alt on Windows and Linux) switch pages, even while typing.
  // Holding ⌥ highlights the shortcuts on the tabs.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "Alt") setShowKeys(true);
      if (!e.altKey || e.metaKey || e.ctrlKey) return;
      const tab = TABS.find((t) => e.code === `Digit${t.key}`);
      if (!tab) return;
      e.preventDefault();
      location.hash = href({ view: tab.view } as Route);
    };
    const up = (e: KeyboardEvent) => e.key === "Alt" && setShowKeys(false);
    const blur = () => setShowKeys(false);
    addEventListener("keydown", down);
    addEventListener("keyup", up);
    addEventListener("blur", blur);
    return () => {
      removeEventListener("keydown", down);
      removeEventListener("keyup", up);
      removeEventListener("blur", blur);
    };
  }, []);

  const savePrompt = (text: string) => {
    setPromptState(text);
    writePref("featurama-prompt", text);
  };
  const ready = model.status === "ready" ? model : null;
  const writer = useWriter(ready?.engine ?? null, ready?.tokenizer ?? null, prompt, savePrompt);
  // Typing takes over from GPT-2.
  const setPrompt = (text: string) => {
    writer.stop();
    savePrompt(text);
  };
  const shuffle = () => {
    writer.stop();
    setPromptState(randomPhrase(prompt));
    try {
      localStorage.removeItem("featurama-prompt");
    } catch {}
  };
  const shared = { prompt, setPrompt, shuffle, writer };

  return (
    <div className="app">
      <header className={showKeys ? "top show-keys" : "top"}>
        <div className="top-inner">
          <a className="brand" href={href({ view: "machine" })} aria-label="Featurama home">
            <Wordmark height={44} label="featurama" />
          </a>
          <nav className="tabs" aria-label="Sections">
            {TABS.map((t) => (
              <a key={t.view} className={route.view === t.view ? "tab tab-on" : "tab"} href={href({ view: t.view } as Route)}
                aria-current={route.view === t.view ? "page" : undefined} aria-keyshortcuts={`Alt+${t.key}`} title={`${t.label} (⌥${t.key})`}>
                {t.label}
                <kbd className="tab-key">⌥{t.key}</kbd>
              </a>
            ))}
          </nav>
          <div className="utils">
            <Status model={model} />
            <button className="switch" role="switch" aria-checked={guide} onClick={() => setGuide(!guide)} title={GLOSSARY.guide}>
              <span className="switch-track"><span className="switch-thumb" /></span>
              <span className="switch-label">Guide</span>
            </button>
            <button className="icon-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")}
              aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"} title={theme === "light" ? "Dark mode" : "Light mode"}>
              <ThemeIcon theme={theme} />
            </button>
          </div>
        </div>
      </header>
      <main>
        {model.status === "ready" ? (
          route.view === "attention" ? (
            <Attention engine={model.engine} tokenizer={model.tokenizer} jobs={model.jobs} head={route.head} guide={guide} {...shared} />
          ) : route.view === "play" ? (
            <Game engine={model.engine} tokenizer={model.tokenizer} />
          ) : route.view === "patching" ? (
            <Patching engine={model.engine} tokenizer={model.tokenizer} guide={guide} />
          ) : route.view === "layers" ? (
            <Layers engine={model.engine} tokenizer={model.tokenizer} guide={guide} {...shared} />
          ) : (
            <Machine engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} jobs={model.jobs} {...shared} />
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
        <span>⌥1–⌥5 switch pages · Tab adds GPT-2's suggestion</span>
        <a href="https://github.com/shuw/featurama">Source</a>
      </footer>
    </div>
  );
}

// Where the model is: downloading, ready on this machine, or unavailable.
function Status({ model }: { model: ModelState }) {
  if (model.status === "ready") return <Tip tip={GLOSSARY.local} align="end" below><span className="status status-ready"><span className="status-dot" />GPT-2 · local</span></Tip>;
  if (model.status === "error") return <span className="status status-error"><span className="status-dot" />GPT-2 unavailable</span>;
  const pct = model.progress ? Math.round((model.progress.loaded / model.progress.total) * 100) : 0;
  return <span className="status"><span className="status-dot" />Loading GPT-2 {pct}%</span>;
}

function ThemeIcon({ theme }: { theme: Theme }) {
  if (theme === "dark")
    return (
      <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" className="icon">
        <circle cx="10" cy="10" r="4" />
        <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M4 4l1.4 1.4M14.6 14.6 16 16M4 16l1.4-1.4M14.6 5.4 16 4" />
      </svg>
    );
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" className="icon">
      <path d="M16 12.5A7 7 0 0 1 7.5 4a7 7 0 1 0 8.5 8.5Z" />
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
