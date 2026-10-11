// Progress kept in this browser only. Storage can be missing or blocked
// (private windows), in which case progress lasts until the tab closes.

const KEY = "featurama-progress-v1";
type Progress = Record<string, boolean>;

let memory: Progress = {};
try {
  memory = JSON.parse(localStorage.getItem(KEY) ?? "{}");
} catch {}

export function done(id: string) {
  return !!memory[id];
}

export function markDone(id: string) {
  if (memory[id]) return;
  memory = { ...memory, [id]: true };
  try {
    localStorage.setItem(KEY, JSON.stringify(memory));
  } catch {}
}
