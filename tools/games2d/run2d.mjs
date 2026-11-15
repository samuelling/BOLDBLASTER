// node run2d.mjs <game> <run index> <n frames> <outdir>
// Generic harness for canvas-2D games: a virtual clock drives the game's own requestAnimationFrame
// loop, a per-game bot (bots/<game>.js) plays, and each 10 Hz frame is captured off the game canvas,
// letterboxed onto the 4:3 scanner screen and reduced to 192x144 Rec.709 luma.
import { chromium } from 'playwright-core';
import fs from 'fs';
const [game, ri, nf, outdir] = [process.argv[2], +process.argv[3], +process.argv[4], process.argv[5]];
const CFG = {
  brickninja: { url: 'http://localhost:8770/Brick-Ninja/index.html', vw: 384, vh: 288, sub: 3 },
  sanctuary: { url: 'http://localhost:8770/Animal-Sanctuary/index.html', vw: 416, vh: 312, sub: 2 },
}[game];
fs.mkdirSync(outdir, { recursive: true });
const pre = `${outdir}/${game}${ri}`;
const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
  args: ['--no-sandbox', '--mute-audio', '--disable-gpu', '--autoplay-policy=no-user-gesture-required'] });
const ctxB = await b.newContext({ viewport: { width: CFG.vw, height: CFG.vh }, deviceScaleFactor: 1 });
// virtual clock: the game's rAF callbacks only run when the harness steps them
await ctxB.addInitScript(() => {
  let vnow = 1000; const q = [];
  performance.now = () => vnow;
  window.requestAnimationFrame = (cb) => { q.push(cb); return q.length; };
  window.cancelAnimationFrame = () => {};
  window.__vstep = (ms) => { vnow += ms; const cbs = q.splice(0); for (const cb of cbs) cb(vnow); return cbs.length; };
  window.__vnow = () => vnow;
});
const pg = await ctxB.newPage();
pg.on('pageerror', (e) => console.log('[err]', e.message.slice(0, 300)));
await pg.goto(CFG.url, { waitUntil: 'load', timeout: 120000 });
await pg.waitForTimeout(1500);
// the games wrap everything in an IIFE; the patched copy exposes window.__eval inside it, so the bot runs in the game's scope
await pg.waitForFunction(() => !!window.__eval, null, { timeout: 60000 });
await pg.evaluate((src) => window.__eval(src), fs.readFileSync(new URL(`./bots/${game}.js`, import.meta.url), 'utf8'));
await pg.evaluate(([seed, sub]) => {
  const B = window.GAMEBOT; B.setup(seed);
  // capture: the game canvas, letterboxed (aspect kept) onto a mid-grey 4:3 screen at 192x144
  const src = B.canvas(), cv = document.createElement('canvas'); cv.width = 192; cv.height = 144;
  const g = cv.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  window.__frame = () => {
    for (let k = 0; k < sub; k++) { B.control(0.1 / sub); window.__vstep(100 / sub); }
    const rec = B.record();
    g.fillStyle = 'rgb(128,128,128)'; g.fillRect(0, 0, 192, 144);
    const a = src.width / src.height; let w = 192, h = 144;
    if (a > 4 / 3) h = 192 / a; else w = 144 * a;
    g.drawImage(src, (192 - w) / 2, (144 - h) / 2, w, h);
    const d = g.getImageData(0, 0, 192, 144).data, out = new Uint16Array(192 * 144);
    for (let i = 0; i < out.length; i++) out[i] = Math.round((0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2]) / 255 * 65535);
    const u8 = new Uint8Array(out.buffer); let bin = '';
    for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
    return { img: btoa(bin), rec };
  };
}, [2000 + ri, CFG.sub]);
for (let k = 0; k < 30; k++) await pg.evaluate(() => window.__frame());     // settle, not recorded
await pg.evaluate(() => window.GAMEBOT.resetEvents());
const img = fs.openSync(`${pre}_lum.u16`, 'w'), log = fs.openSync(`${pre}_log.jsonl`, 'w');
const t0 = Date.now();
for (let f = 0; f < nf; f++) {
  const { img: b64, rec } = await pg.evaluate(() => window.__frame());
  fs.writeSync(img, Buffer.from(b64, 'base64'));
  fs.writeSync(log, JSON.stringify(rec) + '\n');
  if (f % 500 === 0) console.log(`${game} ${ri} frame ${f}/${nf} ${((Date.now() - t0) / 1000 / (f + 1)).toFixed(3)} s/frame`);
}
fs.writeFileSync(`${pre}_events.json`, JSON.stringify(await pg.evaluate(() => window.GAMEBOT.events())));
fs.closeSync(img); fs.closeSync(log);
console.log(`${game} ${ri} done ${((Date.now() - t0) / 1000).toFixed(0)} s`);
await b.close();
