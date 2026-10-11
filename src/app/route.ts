import { useEffect, useState } from "react";

export type Route = { view: "machine" } | { view: "attention"; head?: string } | { view: "layers" };

function parse(hash: string): Route {
  const [view, arg] = hash.replace(/^#\/?/, "").split("/");
  if (view === "attention") return { view, head: arg && /^\d+\.\d+$/.test(arg) ? arg : undefined };
  if (view === "layers") return { view };
  return { view: "machine" };
}

export function href(r: Route) {
  return r.view === "machine" ? "#/" : r.view === "attention" && r.head ? `#/attention/${r.head}` : `#/${r.view}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parse(location.hash));
      window.scrollTo(0, 0);
    };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return route;
}
