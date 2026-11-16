// node run.mjs <run index> <n frames> <outdir>
// Plays Betamax forward with the bot and writes, per 10 Hz frame, a 192x144 uint16 luma
// image (betamax<i>_lum.u16) and a JSON log line (betamax<i>_log.jsonl), plus events JSON.
import { chromium } from '/opt/npm-tools/node_modules/playwright-core/index.mjs';
import fs from 'fs';
const [ri, nf, outdir] = [+process.argv[2], +process.argv[3], process.argv[4]];
fs.mkdirSync(outdir, { recursive: true });
const pre = `${outdir}/${process.env.PREFIX || "betamaxt"}${ri}`;
const LP = process.env.GL === 'llvmpipe';
const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: !LP,
  args: LP ? ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist', '--enable-gpu', '--no-sandbox', '--mute-audio', '--window-size=384,288']
           : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox', '--mute-audio'] });
const pg = await b.newPage({ viewport: { width: 384, height: 288 }, deviceScaleFactor: 1 });
pg.on('pageerror', (e) => console.log('[err]', e.message.slice(0, 300)));
await pg.goto('http://localhost:8767/index.html?nointro&q=low&msaa=0', { waitUntil: 'load', timeout: 300000 });
await pg.waitForFunction(() => window.BW && document.getElementById('fwdgo') && window.BWfwd && window.BW.dig, null, { timeout: 600000, polling: 500 });
await pg.addScriptTag({ path: '/home/claude/bbx/bot_dig.js' });
await pg.evaluate(([seed, sh]) => { window.BBX_SHADOWS = sh; window.H = window.BBX(seed); }, [1000 + ri, process.env.SHADOWS === "1"]);
// warm-up: compile shaders, settle the first life (not recorded)
for (let k = 0; k < 20; k++) await pg.evaluate(() => window.H.frame());
const img = fs.openSync(`${pre}_lum.u16`, 'w'), log = fs.openSync(`${pre}_log.jsonl`, 'w');
const t0 = Date.now();
await pg.evaluate(() => { const e = window.H.events(); for (const k in e) e[k].length = 0; });
let tOff = null;
for (let f = 0; f < nf; f++) {
  const { img: b64, rec } = await pg.evaluate(() => window.H.frame());
  if (tOff === null) tOff = rec.t - 0.1;
  fs.writeSync(img, Buffer.from(b64, 'base64'));
  fs.writeSync(log, JSON.stringify(rec) + '\n');
  if (f % 300 === 0) console.log(`run ${ri} frame ${f}/${nf} ${((Date.now() - t0) / 1000 / (f + 1)).toFixed(3)} s/frame`);
}
const ev = await pg.evaluate(() => window.H.events());
fs.writeFileSync(`${pre}_events.json`, JSON.stringify({ tOff, ...ev }));
fs.closeSync(img); fs.closeSync(log);
console.log(`run ${ri} done ${((Date.now() - t0) / 1000).toFixed(0)} s`);
await b.close();
