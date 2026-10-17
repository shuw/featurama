// IBM Plex, the app's own type, embedded so frames never wait on the network.
import { continueRender, delayRender } from "remotion";
import { FONT_DATA } from "./fonts.generated";

const FACES: [string, string, string][] = [
  ["Plex Sans", "IBMPlexSans.ttf", "100 700"],
  ["Plex Mono", "IBMPlexMono-Medium.ttf", "500"],
  ["Plex Mono", "IBMPlexMono-SemiBold.ttf", "600"],
];

let loaded = false;
export function loadFonts() {
  if (loaded) return;
  loaded = true;
  const handle = delayRender("fonts");
  Promise.all(
    FACES.map(([family, file, weight]) => {
      const face = new FontFace(family, `url(data:font/ttf;base64,${FONT_DATA[file]})`, { weight });
      document.fonts.add(face);
      return face.load();
    }),
  ).then(() => continueRender(handle));
}

export const SANS = '"Plex Sans", system-ui, sans-serif';
export const MONO = '"Plex Mono", monospace';
