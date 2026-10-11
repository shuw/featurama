import { useEffect, useState } from "react";

export type ZoomPart = "head" | "mlp" | "stream" | "weights" | "map";
export type Route =
  | { view: "machine" } | { view: "attention"; head?: string } | { view: "layers" } | { view: "patching" } | { view: "features" }
  | { view: "zoom"; part?: ZoomPart; arg?: string } | { view: "play"; level?: string };

// The site's root path: "/" locally, "/featurama/" on GitHub Pages. index.html
// sets <base> to it, so pages like /featurama/attention/9.9 still find their files.
const base = () => new URL(document.baseURI).pathname;

const VIEWS = ["", "attention", "layers", "patching", "features", "zoom", "play", "index.html"];
const HEAD = /^([0-9]|1[01])\.([0-9]|1[01])$/;

function parse(path: string): Route {
  const [view, arg, arg2] = path.replace(/^\/+/, "").split("/");
  if (view === "attention") return { view, head: arg && HEAD.test(arg) ? arg : undefined };
  if (view === "zoom") {
    if (arg === "head") return { view, part: arg, arg: arg2 && HEAD.test(arg2) ? arg2 : "9.9" };
    // An MLP's layer, optionally with one of its 3,072 neurons: 7 or 7.1234.
    if (arg === "mlp") return { view, part: arg, arg: arg2 && /^([0-9]|1[01])(\.([0-9]{1,3}|[0-2][0-9]{3}|30[0-6][0-9]|307[01]))?$/.test(arg2) ? arg2 : "7" };
    if (arg === "stream" || arg === "weights" || arg === "map") return { view, part: arg };
    return { view };
  }
  if (view === "play") return { view, level: arg && /^[a-z]+$/.test(arg) ? arg : undefined };
  if (view === "layers" || view === "patching" || view === "features") return { view };
  return { view: "machine" };
}

const current = () => parse(location.pathname.slice(base().length));

// A link to a page, relative to the site's root.
export function href(r: Route) {
  if (r.view === "machine") return "./";
  if (r.view === "attention" && r.head) return `attention/${r.head}`;
  if (r.view === "play" && r.level) return `play/${r.level}`;
  if (r.view === "zoom" && r.part) return r.arg ? `zoom/${r.part}/${r.arg}` : `zoom/${r.part}`;
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

export function useRoute(): Route {
  const [route, setRoute] = useState(() => {
    fromHash();
    // Unknown addresses go to the home page.
    if (!VIEWS.includes(location.pathname.slice(base().length).split("/")[0])) {
      const url = new URL("./", document.baseURI);
      url.search = location.search;
      history.replaceState(null, "", url);
    }
    return current();
  });
  useEffect(() => {
    const on = (e?: Event) => {
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
