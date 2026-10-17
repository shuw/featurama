// Render the video. Remotion draws the frames in the installed Chrome and a system ffmpeg stitches
// them with the music, rather than Remotion's own downloaded browser and ffmpeg. Set CHROME and
// FFMPEG to override where they are.
//
//   node scripts/render.mjs                         full 1080p -> out/featurama-trailer.mp4
//   node scripts/render.mjs --preview               first 20 s at half size -> out/preview.mp4
//   node scripts/render.mjs --from 20 --to 40 --scale 0.5 --out out/clip.mp4
//   node scripts/render.mjs --stills 22,26.5,29     single frames -> build/stills/
import {bundle} from '@remotion/bundler';
import {renderFrames, renderStill, selectComposition} from '@remotion/renderer';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {embedFonts} from './embed-fonts.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const preview = args.includes('--preview');
const browserExecutable = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ffmpeg = process.env.FFMPEG ?? '/opt/homebrew/bin/ffmpeg';

embedFonts();
const serveUrl = await bundle({entryPoint: path.join(root, 'src/index.ts'), publicDir: path.join(root, 'public')});
const comp = opt('comp', 'Trailer');
const composition = await selectComposition({serveUrl, id: comp, browserExecutable});
const audio = {Trailer: 'public/music.wav', Trailer2: 'public/trailer2.wav', Trailer3: 'public/trailer3-mix.wav'}[comp];
const outName = {Trailer: 'featurama-trailer', Trailer2: 'featurama-trailer-2', Trailer3: 'featurama-trailer-3'}[comp];
const fps = composition.fps;

const stills = opt('stills');
if (stills) {
  const dir = path.join(root, 'build/stills');
  fs.mkdirSync(dir, {recursive: true});
  const scale = Number(opt('scale', '0.5'));
  for (const s of stills.split(',')) {
    const frame = Math.round(Number(s) * fps);
    const output = path.join(dir, `${comp}-${s}.jpeg`);
    await renderStill({composition, serveUrl, browserExecutable, frame, output, imageFormat: 'jpeg', scale});
    console.log(output);
  }
  process.exit(0);
}

const from = Number(opt('from', '0'));
const to = Number(opt('to', preview ? '20' : String(composition.durationInFrames / fps)));
const scale = Number(opt('scale', preview ? '0.5' : '1'));
const out = path.resolve(root, opt('out', preview ? 'out/preview.mp4' : `out/${outName}.mp4`));
const frameRange = [Math.round(from * fps), Math.min(composition.durationInFrames, Math.round(to * fps)) - 1];

const framesDir = path.join(root, 'build/frames');
const total = frameRange[1] - frameRange[0] + 1;
const t0 = Date.now();
// Render in chunks, each into its own folder, retrying a chunk if a render tab stalls. A finished
// chunk is kept, so re-running the same command resumes where it stopped.
const CHUNK = 600;
// The tag changes with the source and footage, so edits never reuse stale frames.
const hash = createHash('sha1');
for (const dir of ['src', 'public/footage', 'public/footage2', 'public/footage3']) {
  for (const f of fs.readdirSync(path.join(root, dir), {recursive: true}).sort()) {
    const full = path.join(root, dir, String(f));
    const st = fs.statSync(full);
    if (st.isFile() && !String(f).endsWith('.generated.ts')) hash.update(`${f}:${st.size}:${st.mtimeMs}`);
  }
}
const tag = `${comp}-${scale}-${frameRange[0]}-${hash.digest('hex')}`;
if (!fs.existsSync(path.join(framesDir, '.tag')) || fs.readFileSync(path.join(framesDir, '.tag'), 'utf8') !== tag) {
  fs.rmSync(framesDir, {recursive: true, force: true});
  fs.mkdirSync(framesDir, {recursive: true});
  fs.writeFileSync(path.join(framesDir, '.tag'), tag);
}
let done = 0;
for (let start = frameRange[0]; start <= frameRange[1]; start += CHUNK) {
  const end = Math.min(frameRange[1], start + CHUNK - 1);
  const dir = path.join(framesDir, `chunk-${String(start).padStart(6, '0')}`);
  const want = end - start + 1;
  if (fs.existsSync(dir) && fs.readdirSync(dir).filter((f) => f.endsWith('.jpeg')).length === want) {
    done += want;
    continue;
  }
  for (let attempt = 1; ; attempt++) {
    fs.rmSync(dir, {recursive: true, force: true});
    try {
      await renderFrames({
        composition, serveUrl, browserExecutable, outputDir: dir, imageFormat: 'jpeg', jpegQuality: 92, scale, frameRange: [start, end],
        concurrency: Number(opt('concurrency', '8')), timeoutInMilliseconds: 60000,
        onStart: () => {},
        onFrameUpdate: (n) => {
          if (n % 60 === 0) process.stdout.write(`\r${done + n}/${total} frames, ${((Date.now() - t0) / 1000).toFixed(0)}s   `);
        },
      });
      break;
    } catch (e) {
      if (attempt >= 3) throw e;
      console.log(`\nchunk ${start}-${end} failed (${String(e.message).split('\n')[0]}); retrying`);
    }
  }
  done += want;
}
process.stdout.write('\n');

fs.mkdirSync(path.dirname(out), {recursive: true});
execFileSync(ffmpeg, [
  '-v', 'error', '-y', '-framerate', String(fps), '-pattern_type', 'glob', '-i', path.join(framesDir, 'chunk-*', '*.jpeg'),
  '-ss', String(frameRange[0] / fps), '-i', path.join(root, audio),
  '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
  '-c:a', 'aac', '-b:a', '256k', '-af', 'apad', '-shortest', '-movflags', '+faststart', out,
], {stdio: 'inherit'});
console.log(`wrote ${out} (${total} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
// Remotion's bundler can keep the process alive after rendering; nothing is left to do.
process.exit(0);
