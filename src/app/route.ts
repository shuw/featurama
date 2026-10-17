import { useEffect, useState } from "react";

export type LayersMode = "numbers" | "map";
export type LabTool = "unplug" | "swap" | "steer" | "trace";
export type Route =
  | { view: "machine" } | { view: "head"; head?: string } | { view: "mlp"; arg?: string } | { view: "layers"; mode?: LayersMode } | { view: "features" } | { view: "weights" }
  | { view: "lab"; tool?: LabTool } | { view: "play"; level?: string };

// The three sections in the header.
export type Section = "explore" | "lab" | "play";
export const sectionOf = (r: Route): Section => (r.view === "lab" ? "lab" : r.view === "play" ? "play" : "explore");

// The site's root path: "/" locally, "/featurama/" on GitHub Pages. index.html
// sets <base> to it, so pages like /featurama/head/9.9 still find their files.
const base = () => new URL(document.baseURI).pathname;

const VIEWS = ["", "head", "mlp", "layers", "features", "weights", "lab", "play", "index.html"];
const HEAD = /^([0-9]|1[01])\.([0-9]|1[01])$/;
// An MLP's layer, optionally with one of its 3,072 neurons: 7 or 7.1234.
const MLP = /^([0-9]|1[01])(\.([0-9]{1,3}|[0-2][0-9]{3}|30[0-6][0-9]|307[01]))?$/;

function parse(path: string): Route {
  const [view, arg] = path.replace(/^\/+/, "").split("/");
  if (view === "head") return { view, head: arg && HEAD.test(arg) ? arg : undefined };
  if (view === "mlp") return { view, arg: arg && MLP.test(arg) ? arg : undefined };
  if (view === "layers") return { view, mode: arg === "numbers" || arg === "map" ? arg : undefined };
  if (view === "lab") return { view, tool: arg === "swap" || arg === "steer" || arg === "trace" ? arg : "unplug" };
  if (view === "play") return { view, level: arg && /^[a-z]+$/.test(arg) ? arg : undefined };
  if (view === "features" || view === "weights") return { view };
  return { view: "machine" };
}

// Addresses from before the site had three sections.
function moved(path: string): string | null {
  const [view, a, b] = path.replace(/^\/+/, "").split("/");
  if (view === "attention") return a ? `head/${a}` : "head";
  if (view === "patching") return "lab/swap";
  if (view !== "zoom") return null;
  if (a === "head" || a === "mlp") return b ? `${a}/${b}` : a;
  if (a === "stream") return "layers/numbers";
  if (a === "map") return "layers/map";
  if (a === "weights") return "head/5.5";
  return "lab/trace";
}

const current = () => parse(location.pathname.slice(base().length));

// A link to a page, relative to the site's root.
export function href(r: Route) {
  if (r.view === "machine") return "./";
  if (r.view === "head" && r.head) return `head/${r.head}`;
  if (r.view === "mlp" && r.arg) return `mlp/${r.arg}`;
  if (r.view === "layers" && r.mode) return `layers/${r.mode}`;
  if (r.view === "lab") return `lab/${r.tool ?? "unplug"}`;
  if (r.view === "play" && r.level) return `play/${r.level}`;
  return r.view;
}

// Go to a page, keeping the shared prompt in the address. `inPlace` is for
// changes within a page (another head, another neuron): no new history entry
// and no scroll to the top.
export function navigate(to: string, inPlace = false) {
  const url = new URL(to, document.baseURI);
  url.search = location.search;
  if (inPlace) history.replaceState(null, "", url);
  else history.pushState(null, "", url);
  dispatchEvent(new CustomEvent("routechange", { detail: { inPlace } }));
}

// Older links used #/attention/9.9; swap them for the path.
function fromHash() {
  if (!location.hash.startsWith("#/")) return false;
  const url = new URL(location.hash.slice(2) || "./", document.baseURI);
  url.search = location.search;
  history.replaceState(null, "", url);
  return true;
}

// Old addresses go to their new place, unknown ones to the home page.
function redirect() {
  const path = location.pathname.slice(base().length);
  const to = moved(path) ?? (VIEWS.includes(path.split("/")[0]) ? null : "./");
  if (to === null) return;
  const url = new URL(to, document.baseURI);
  url.search = location.search;
  history.replaceState(null, "", url);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => {
    fromHash();
    redirect();
    return current();
  });
  useEffect(() => {
    const on = (e?: Event) => {
      redirect();
      setRoute(current());
      if (!(e instanceof CustomEvent && e.detail?.inPlace)) window.scrollTo(0, 0);
    };
    // Links inside the app change the page without reloading it.
    const click = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.("a");
      if (!a || a.target || a.hasAttribute("download")) return;
      const url = new URL(a.href);
      if (url.origin !== location.origin || !url.pathname.startsWith(base()) || /\.[a-z]+$/i.test(url.pathname)) return;
      e.preventDefault();
      navigate(url.pathname);
    };
    const hash = () => fromHash() && on();
    addEventListener("hashchange", hash);
    addEventListener("popstate", on);
    addEventListener("routechange", on);
    document.addEventListener("click", click);
    return () => {
      removeEventListener("hashchange", hash);
      removeEventListener("popstate", on);
      removeEventListener("routechange", on);
      document.removeEventListener("click", click);
    };
  }, []);
  return route;
}

// The prompt from a shared link, if any.
export const promptFromUrl = () => new URLSearchParams(location.search).get("p");

// Keep one ?name= in the address in step with the page; null removes it.
export function setUrlParam(name: string, value: string | null) {
  const url = new URL(location.href);
  if (value === null) url.searchParams.delete(name);
  else url.searchParams.set(name, value);
  if (url.href !== location.href) history.replaceState(null, "", url);
}
export const setPromptInUrl = (text: string | null) => setUrlParam("p", text);
export const urlParam = (name: string) => new URLSearchParams(location.search).get(name);

// This page's address with the current prompt, edited or not, for sharing.
export function shareUrl(prompt: string) {
  const url = new URL(location.href);
  url.searchParams.set("p", prompt);
  return url.href;
}
