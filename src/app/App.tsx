import { useEffect, useState } from "react";
import { Attention } from "./Attention.tsx";
import { Features } from "./Features.tsx";
import { Lab } from "./Lab.tsx";
import { LayersPage } from "./LayersPage.tsx";
import { MlpPage } from "./MlpPage.tsx";
import { Tower } from "./Tower.tsx";
import { PlayHub, PuzzleLevel } from "./Puzzles.tsx";
import { Loading } from "./Loading.tsx";
import { Game } from "./Game.tsx";
import { Machine } from "./Machine.tsx";
import { Overview } from "./Overview.tsx";
import { useModel, type ModelState } from "./model.ts";
import { href, navigate, promptFromUrl, sectionOf, setPromptInUrl, shareUrl, useRoute, type Route, type Section } from "./route.ts";
import { randomPhrase } from "./phrases.ts";
import { GLOSSARY } from "./glossary.ts";
import { Tip } from "./Tip.tsx";
import { useWriter } from "./useWriter.ts";
import { Wordmark } from "./Wordmark.tsx";

type Theme = "light" | "dark";

// Three sections, each with its pages. ⌥1–⌥3 switch sections; ⌥ and a
// letter, underlined in its name, opens a page. The letters skip Mac accent
// keys (⌥E, ⌥U, ⌥I, ⌥N) and Alt+D, E, F, which browsers use elsewhere.
interface Place {
  section: Section;
  label: string;
  key: string;
  to: Route;
  on: (r: Route) => boolean;
}
const PLACES: Place[] = [
  { section: "explore", label: "Overview", key: "O", to: { view: "overview" }, on: (r) => r.view === "overview" },
  { section: "explore", label: "Machine", key: "M", to: { view: "machine" }, on: (r) => r.view === "machine" },
  { section: "explore", label: "Heads", key: "H", to: { view: "head" }, on: (r) => r.view === "head" },
  { section: "explore", label: "MLPs", key: "P", to: { view: "mlp" }, on: (r) => r.view === "mlp" },
  { section: "explore", label: "Layers", key: "L", to: { view: "layers" }, on: (r) => r.view === "layers" },
  { section: "explore", label: "Features", key: "A", to: { view: "features" }, on: (r) => r.view === "features" },
  { section: "explore", label: "Weights", key: "W", to: { view: "weights" }, on: (r) => r.view === "weights" },
  { section: "lab", label: "Unplug", key: "G", to: { view: "lab", tool: "unplug" }, on: (r) => r.view === "lab" && r.tool === "unplug" },
  { section: "lab", label: "Swap", key: "S", to: { view: "lab", tool: "swap" }, on: (r) => r.view === "lab" && r.tool === "swap" },
  { section: "lab", label: "Steer", key: "R", to: { view: "lab", tool: "steer" }, on: (r) => r.view === "lab" && r.tool === "steer" },
  { section: "lab", label: "Trace", key: "T", to: { view: "lab", tool: "trace" }, on: (r) => r.view === "lab" && r.tool === "trace" },
  { section: "play", label: "Puzzles", key: "Z", to: { view: "play" }, on: (r) => r.view === "play" && r.level !== "guess" },
  { section: "play", label: "You vs GPT-2", key: "V", to: { view: "play", level: "guess" }, on: (r) => r.view === "play" && r.level === "guess" },
];

const SECTIONS: { section: Section; label: string; key: string }[] = [
  { section: "explore", label: "Explore", key: "1" },
  { section: "lab", label: "Lab", key: "2" },
  { section: "play", label: "Play", key: "3" },
];
const first = (section: Section) => PLACES.find((p) => p.section === section)!;

// The page's name with its shortcut letter underlined.
function Hot({ label, letter }: { label: string; letter: string }) {
  const i = label.toUpperCase().indexOf(letter);
  // One span, so the flex link doesn't split the text and drop its spaces.
  return <span>{label.slice(0, i)}<u className="hot">{label[i]}</u>{label.slice(i + 1)}</span>;
}

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
  const section = sectionOf(route);
  const [theme, setTheme] = useState<Theme>(() => (readPref("featurama-theme") === "dark" ? "dark" : "light"));
  // One prompt for every page, so switching pages keeps what you typed. It
  // starts as a random phrase, or the one in a shared link; once edited, it's
  // kept across reloads and in the address, so the link can be shared.
  const [prompt, setPromptState] = useState(() => promptFromUrl() ?? readPref("featurama-prompt") ?? randomPhrase());
  const [edited, setEdited] = useState(() => promptFromUrl() !== null || readPref("featurama-prompt") !== null);
  useEffect(() => {
    const t = setTimeout(() => setPromptInUrl(edited ? prompt : null), 300);
    return () => clearTimeout(t);
  }, [prompt, edited]);
  const [showKeys, setShowKeys] = useState(false);

  const [copied, setCopied] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writePref("featurama-theme", theme);
  }, [theme]);

  // ⌥1–⌥3 and ⌥ + a letter (Alt on Windows and Linux) switch pages, even while typing.
  // Holding ⌥ highlights the shortcuts on the tabs.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "Alt") setShowKeys(true);
      if (!e.altKey || e.metaKey || e.ctrlKey) return;
      const to = SECTIONS.find((t) => e.code === `Digit${t.key}`)?.section;
      const place = to ? first(to) : PLACES.find((p) => e.code === `Key${p.key}`);
      if (!place) return;
      e.preventDefault();
      navigate(href(place.to));
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
    setEdited(true);
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
    setEdited(false);
    try {
      localStorage.removeItem("featurama-prompt");
    } catch {}
  };
  const shared = { prompt, setPrompt, shuffle, writer };

  return (
    <div className="app">
      <header className={showKeys ? "top show-keys" : "top"}>
        <div className="top-inner">
          <a className="brand" href={href({ view: "overview" })} aria-label="Featurama home">
            <Wordmark height={44} label="featurama" />
          </a>
          {/* One row: the open section unfolds its pages to its right. */}
          <nav className="tabs" aria-label="Sections and pages">
            {SECTIONS.map((t) => {
              const on = section === t.section;
              return (
                <div key={t.section} className={on ? "sec sec-on" : "sec"}>
                  <a className="tab" href={href(first(t.section).to)} aria-current={on ? "true" : undefined}
                    aria-keyshortcuts={`Alt+${t.key}`} title={`${t.label} (⌥${t.key})`}>
                    {t.label}
                    <kbd className="tab-key">⌥{t.key}</kbd>
                  </a>
                  <div className="sec-pages" inert={!on}>
                    <div className="sec-pages-inner">
                      {PLACES.filter((p) => p.section === t.section).map((p) => (
                        <a key={p.label} href={href(p.to)} className={on && p.on(route) ? "place place-on" : "place"} aria-current={on && p.on(route) ? "page" : undefined}
                          aria-keyshortcuts={`Alt+${p.key}`} title={`${p.label} (⌥${p.key})`}>
                          <Hot label={p.label} letter={p.key} />
                        </a>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </nav>
          <div className="utils">
            <Status model={model} />
            <button className="icon-button" onClick={async () => {
              try {
                await navigator.clipboard.writeText(shareUrl(prompt));
                setCopied(true);
                setTimeout(() => setCopied(false), 1800);
              } catch {}
            }} aria-label="Copy a link to this page with your prompt" title={copied ? "Link copied" : "Copy link"}>
              {copied ? <span className="copied" aria-live="polite">✓</span> : <LinkIcon />}
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
          route.view === "head" ? (
            <Attention engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} jobs={model.jobs} head={route.head} {...shared} />
          ) : route.view === "mlp" ? (
            <MlpPage engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} layer={Number((route.arg ?? "7").split(".")[0])}
              neuron={route.arg?.includes(".") ? Number(route.arg.split(".")[1]) : null}
              onPick={(l, n) => navigate(href({ view: "mlp", arg: n === null ? `${l}` : `${l}.${n}` }), true)} {...shared} />
          ) : route.view === "layers" ? (
            <LayersPage engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} mode={route.mode} {...shared} />
          ) : route.view === "weights" ? (
            <Tower engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} {...shared} />
          ) : route.view === "features" ? (
            <Features engine={model.engine} tokenizer={model.tokenizer} {...shared} />
          ) : route.view === "lab" ? (
            <Lab engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} tool={route.tool ?? "unplug"} {...shared} />
          ) : route.view === "play" ? (
            route.level === "guess" ? <Game engine={model.engine} tokenizer={model.tokenizer} />
            : route.level ? <PuzzleLevel key={route.level} engine={model.engine} tokenizer={model.tokenizer} jobs={model.jobs} id={route.level} />
            : <PlayHub />
          ) : route.view === "machine" ? (
            <Machine engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} jobs={model.jobs} {...shared} />
          ) : (
            <Overview engine={model.engine} tokenizer={model.tokenizer} weights={model.weights} {...shared} />
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
        <span>⌥1–⌥3 switch sections · ⌥ + underlined letter opens a page · Tab adds GPT-2's suggestion</span>
        <span><a href="https://github.com/shuw/featurama">Source</a> · <a href="https://shuw.github.io/">More from the creator</a></span>
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

function LinkIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" />
      <path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" />
    </svg>
  );
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

