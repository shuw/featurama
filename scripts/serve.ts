// Dev server: rebuilds the pages when a source file changes (refresh to see
// it), and serves the model and test fixtures from disk.
import { watch } from "node:fs";
import { join, normalize } from "node:path";

const root = join(import.meta.dir, "..");
const pages = ["index.html", "tests/gpu/index.html"];
let files = new Map<string, Blob>();
let dirty = true;
let building: Promise<void> | null = null;

async function build() {
  const result = await Bun.build({ entrypoints: pages.map((p) => join(root, p)), root, sourcemap: "inline" });
  if (!result.success) {
    console.error(result.logs.join("\n"));
    return;
  }
  files = new Map(result.outputs.map((o) => ["/" + o.path.replace(/^\.\//, ""), o]));
  dirty = false;
}

for (const path of ["src", "tests", "index.html"]) watch(join(root, path), { recursive: true }, () => (dirty = true));

// Static files: the model under /model/, fixtures under /fixtures/.
function staticFile(path: string) {
  const clean = normalize(path).replace(/^(\.\.[/\\])+/, "");
  if (clean.startsWith("/model/")) return Bun.file(join(root, "public", clean));
  if (clean.startsWith("/fixtures/")) return Bun.file(join(root, clean));
  return null;
}

const server = Bun.serve({
  port: Number(process.env.PORT ?? 5190),
  async fetch(req) {
    const path = decodeURIComponent(new URL(req.url).pathname);
    const file = staticFile(path);
    if (file) return (await file.exists()) ? new Response(file) : new Response("Not found", { status: 404 });
    if (dirty) await (building ??= build().finally(() => (building = null)));
    const page = path === "/" ? "/index.html" : path.endsWith("/") ? `${path}index.html` : path;
    const out = files.get(page);
    return out ? new Response(out, { headers: { "Cache-Control": "no-store" } }) : new Response("Not found", { status: 404 });
  },
});
console.log(`Featurama on ${server.url}`);
