# 2D game harness (round 8: Brick Ninja, Animal Sanctuary)

`run2d.mjs` plays a canvas-2D game deterministically and writes the same per-run files as the
Betamax harness: 192 x 144 Rec.709 luma frames at 10 Hz (`<game><run>_luma.u16`) plus frame and
event logs, which `run_round8.py` turns into model inputs.

1. Clone `samuelling/Brick-Ninja` and `samuelling/Animal-Sanctuary` side by side in one folder
   and apply the hook patches (`patch -p1 -d Brick-Ninja < brick-ninja-harness-hook.patch`,
   likewise `animal-sanctuary-harness-hook.patch`). Each adds one line, `window.__eval`, inside
   the game's IIFE so the bot can run in the game's own scope.
2. Serve that folder on port 8770 (`python -m http.server 8770`).
3. `npm i playwright-core` and point `executablePath` at a Chrome binary, then
   `node run2d.mjs brickninja <run> 3000 <outdir>` (or `sanctuary`). A 5-minute run takes about
   10-20 s.

How it works: an init script replaces `performance.now` and `requestAnimationFrame` with a
virtual clock, so the game only advances when the harness steps it (`__vstep`); `bots/<game>.js`
reads the game state and sets its inputs every substep; each frame the canvas is letterboxed onto
a mid-grey 4:3 screen and reduced to luma. Then
`GAME=brickninja python run_round8.py all 12` (or `GAME=sanctuary`) and `python export_round8.py`.
