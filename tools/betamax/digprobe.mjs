// probe the tunnels: boot, start forward play, go down a hatch, look around, screenshots
import { chromium } from '/opt/npm-tools/node_modules/playwright-core/index.mjs';
const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: false,
  args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist', '--enable-gpu', '--no-sandbox', '--mute-audio', '--window-size=640,480'] });
const pg = await b.newPage({ viewport: { width: 640, height: 480 }, deviceScaleFactor: 1 });
pg.on('pageerror', (e) => console.log('[err]', e.message.slice(0, 400)));
pg.on('console', (m) => { const t = m.text(); if (/dig|error|Error/.test(t)) console.log('[pg]', t.slice(0, 300)); });
let t0 = Date.now();
await pg.goto('http://localhost:8767/index.html?nointro&q=low&msaa=0', { waitUntil: 'load', timeout: 300000 });
await pg.waitForFunction(() => window.BW && document.getElementById('fwdgo') && window.BWfwd && window.BW.dig, null, { timeout: 600000, polling: 500 });
console.log('booted', (Date.now() - t0) / 1000, 's');
const info = await pg.evaluate(() => { const D = window.BW.dig, P = D.P; return { cells: P.cells.length, edges: P.edges.length, loops: P.loopEdges.length, hatches: P.hatches.map((h) => [h.id, h.kind, +h.x.toFixed(1), +h.z.toFixed(1)]), crates: P.crates.map((c) => [c.id, c.x, c.z]), figs: D.figs.length }; });
console.log(JSON.stringify(info));
await pg.evaluate(() => {
  const G = window.BW; G.hold = true; G.input.settings.vcr = false;
  const nul = new Proxy({}, { get: () => null }); for (const k of ['B', 'R']) if (!G.audio[k]) G.audio[k] = nul;
  window.INP = { moveX: 0, moveZ: 0, lookX: 0, lookY: 0, fire: false, ads: false, crouch: false, jump: false, reload: false, sprint: false, touchLook: false, slot: 0, use: false };
  G.input.read = () => Object.assign({}, window.INP);
  const a = G.S.actors.find((q) => q.side === 0 && q.spawn + 15 < 60);
  window.BWfwd(a.id, 60); window.BWstep(2, 0.05, false);
});
const shot = async (name) => { await pg.evaluate(() => window.BWstep(1, 0.05, true)); await pg.screenshot({ path: `/tmp/dig_${name}.png` }); };
// put the player beside a lot hatch, look at it
const r = await pg.evaluate(() => { const G = window.BW, h = G.dig.P.hatches[2], p = G.player; p.pos.set(h.x + 3, G.heightAt(h.x + 3, h.z), h.z); p.yaw = Math.atan2(3, 0); p.pitch = -0.4; window.BWstep(2, 0.05, false); return [h.x, h.z, p.pos.y]; });
console.log('at hatch', r); await shot('hatch');
await pg.evaluate(() => { const G = window.BW, h = G.dig.P.hatches[2], p = G.player; p.pos.set(h.x + 0.5, p.pos.y, h.z); window.INP.use = true; window.BWstep(1, 0.05, false); window.INP.use = false; window.BWstep(30, 0.05, false); });
console.log('after descend', await pg.evaluate(() => { const p = window.BW.player; return [p.pos.x, p.pos.y, p.pos.z, window.BW.dig.log.descents.length]; }));
for (const [k, yaw] of [['a', 0], ['b', Math.PI / 2], ['c', Math.PI], ['d', -Math.PI / 2]]) { await pg.evaluate((y) => { const p = window.BW.player; p.yaw = y; p.pitch = 0; }, yaw); await shot('under_' + k); }
// walk forward a bit
await pg.evaluate(() => { window.INP.moveZ = 1; window.BWstep(40, 0.05, false); window.INP.moveZ = 0; });
console.log('walked', await pg.evaluate(() => { const p = window.BW.player; return [p.pos.x.toFixed(2), p.pos.y.toFixed(2), p.pos.z.toFixed(2)]; }));
await shot('walk');
// lights out
await pg.evaluate(() => { const D = window.BW.dig, p = window.BW.player, z = D.zoneAt(p.pos.x, p.pos.z); D.zoneOn[z] = 0; D.darkUntil[z] = D.t + 100; });
await shot('dark');
// a figure: stand 6 m from one and look at it
await pg.evaluate(() => { const D = window.BW.dig, p = window.BW.player; D.zoneOn = [1, 1, 1, 1]; D.darkUntil = [0, 0, 0, 0];
  let ok = false;
  for (const k of D.P.cells) { const [cx, cz] = D.cellXZ(k); p.pos.set(cx, 0.2, cz);
    for (let a = 0; a < 4 && !ok; a++) { p.yaw = a * Math.PI / 2; ok = D.spawnFigure(); }
    if (ok) break; }
  const f = D.figs.find((q) => q.active); p.yaw = Math.atan2(-(f.x - p.pos.x), -(f.z - p.pos.z)); p.pitch = -0.03; window.BWstep(2, 0.05, false);
  window.FIGINFO = [f.foe, Math.hypot(f.x - p.pos.x, f.z - p.pos.z).toFixed(1)]; });
console.log('figure', await pg.evaluate(() => window.FIGINFO));
await shot('figure');
const t1 = Date.now(); await pg.evaluate(() => { for (let k = 0; k < 10; k++) window.BWstep(1, 0.05, true); }); console.log('ms/drawn step underground', (Date.now() - t1) / 10);
await b.close();
