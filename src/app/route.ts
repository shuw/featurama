import { useEffect, useState } from "react";

export type Route = { view: "machine" } | { view: "attention"; head?: string } | { view: "layers" } | { view: "patching" } | { view: "features" } | { view: "play"; level?: string };

// The site's root path: "/" locally, "/featurama/" on GitHub Pages. index.html
// sets <base> to it, so pages like /featurama/attention/9.9 still find their files.
const base = () => new URL(document.baseURI).pathname;

const VIEWS = ["", "attention", "layers", "patching", "features", "play", "index.html"];

function parse(path: string): Route {
  const [view, arg] = path.replace(/^\/+/, "").split("/");
  if (view === "attention") return { view, head: arg && /^([0-9]|1[01])\.([0-9]|1[01])$/.test(arg) ? arg : undefined };
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
  return r.view;
}

// Go to a page, keeping the shared prompt in the address.
export function navigate(to: string) {
  const url = new URL(to, document.baseURI);
  url.search = location.search;
  history.pushState(null, "", url);
  dispatchEvent(new Event("routechange"));
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
    const on = () => {
      setRoute(current());
      window.scrollTo(0, 0);
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
