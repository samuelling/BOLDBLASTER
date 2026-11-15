// fast behavioural test of the tunnel bot: no drawing, n frames at 10 Hz
import { chromium } from '/opt/npm-tools/node_modules/playwright-core/index.mjs';
import fs from 'fs';
const [ri, nf] = [+process.argv[2], +process.argv[3]];
const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox', '--mute-audio'] });
const pg = await b.newPage({ viewport: { width: 384, height: 288 } });
pg.on('pageerror', (e) => console.log('[err]', e.message.slice(0, 300)));
await pg.goto('http://localhost:8767/index.html?nointro&q=low&msaa=0', { waitUntil: 'load', timeout: 300000 });
await pg.waitForFunction(() => window.BW && document.getElementById('fwdgo') && window.BWfwd && window.BW.dig, null, { timeout: 600000, polling: 500 });
await pg.addScriptTag({ path: '/home/claude/bbx/bot_dig.js' });
await pg.evaluate(([seed]) => { window.BBX_NODRAW = true; window.H = window.BBX(seed); }, [1000 + ri]);
const t0 = Date.now(); const recs = [];
for (let f = 0; f < nf; f += 50) recs.push(...await pg.evaluate((n) => { const o = []; for (let k = 0; k < n; k++) o.push(window.H.frame().rec); return o; }, Math.min(50, nf - f)));
const ev = await pg.evaluate(() => window.H.events());
fs.writeFileSync(`/tmp/digsim${ri}.json`, JSON.stringify({ recs, ev }));
console.log('done', nf, 'frames in', (Date.now() - t0) / 1000, 's');
await b.close();
