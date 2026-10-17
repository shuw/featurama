# Featurama trailers

Trailers made from real footage of the app: every number on screen
comes from GPT-2 running in the browser.

- **Trailer** (src/Trailer.tsx): the first cut, with a soundtrack made in code (`npm run music`).
- **Trailer2** (src/Trailer2.tsx): the 3D weights tower first, cut to a Suno track. The track
  isn't in git; `src/beats2.json` holds the cut points and beats (spliced from the full song
  on downbeats). Tower shots are rendered frame by frame from the app's film mode
  (`/weights?film`) by `node scripts/film.mjs`; the tour reuses `npm run capture` footage.
  Render with `node scripts/render.mjs --comp Trailer2`.
- **Trailer3** (src/Trailer3.tsx): one weight pulls back to the whole model, then a small
  mystery (why does GPT-2 say "Mary"?), a fast tour and GPT-2 writing a sentence. Same track,
  played straight through with one stop; `src/beats3.json` holds its beats. `node
  scripts/capture2.mjs` films the app at 1080p and logs each clip's clicks and keystrokes;
  `src/cut3.json` places the clips so clicks land on beats. `node scripts/film3.mjs` renders
  the tower shots, `python scripts/sfx3.py` synthesizes the sound effects and mixes them under
  the music, then `node scripts/render.mjs --comp Trailer3`.

    npm install --ignore-scripts
    npm run music              # synthesize public/music.wav (128 BPM)
    bun run dev                # in the repo root, then:
    npm run capture            # film the app into public/footage
    npm run render             # out/featurama-trailer.mp4

`npm run stills -- 5,20,45` renders single frames (seconds) to build/stills.
Rendering uses the installed Google Chrome and a system ffmpeg
(`CHROME` and `FFMPEG` override their paths). Finished frames are kept in
build/frames, so an interrupted render resumes; any change to src or the
footage starts afresh.

Cuts and captions are placed in beats in src/timeline.ts. Fonts are IBM Plex
(SIL Open Font License, public/fonts/OFL.txt).
