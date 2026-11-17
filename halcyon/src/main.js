// Halcyon: boot, input, HUD, the loop. ?bot=1 hands the clock to a driver (harness/bot.js) through
// HX.tick(input) and HX.draw(); otherwise the game runs on requestAnimationFrame for a human.
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { buildWorld, U, los, lightAt, OBSTACLES } from './world.js';
import { wallDist } from './game.js';
import { Game } from './game.js';
import * as P from './plan.js';

const q = new URLSearchParams(location.search);
const LOW = q.get('q') === 'low' || (q.get('q') !== 'high' && matchMedia('(pointer: coarse)').matches), BOT = q.get('bot') === '1';
const renderer = new THREE.WebGLRenderer({ antialias: q.get('msaa') !== '0', powerPreference: 'high-performance', preserveDrawingBuffer: BOT });
renderer.setPixelRatio(LOW ? 1 : Math.min(2, devicePixelRatio));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.85;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.05, 400);
let composer = null;
if (!LOW) {
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.35, 0.5, 0.9));
  composer.addPass(new OutputPass());
}
const $ = (id) => document.getElementById(id);
const hud = {
  msgT: 0,
  msg(t) { $('msg').textContent = t; $('msg').style.opacity = 1; this.msgT = performance.now(); },
};
const G = window.HX = { THREE, scene, camera, renderer, P, U, ready: false, hud, HXlos: los, HXobs: OBSTACLES, HXwall: (L, x, z) => wallDist(L, x, z), HXlight: (L, x, y, z) => lightAt(L, x, y, z, 0, 1, 0) };
const game = new Game(G); G.game = game;

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
  if (composer) composer.setSize(innerWidth, innerHeight);
  if (game.viewCam) { game.viewCam.aspect = camera.aspect; game.viewCam.updateProjectionMatrix(); }
});

function draw() {
  if (composer) composer.render(); else { renderer.autoClear = true; renderer.render(scene, camera); }
  renderer.autoClear = false; renderer.clearDepth(); renderer.render(game.viewScene, game.viewCam); renderer.autoClear = true;
}
const FLOORS = ['ground floor', 'first floor', 'second floor'];
function drawHud() {
  const p = game.player, o = game.objective;
  let t = '';
  if (!o) t = 'Stand by…';
  else if (o.phase === 'find') t = `Find the guest in <b>${o.name}</b> · ${FLOORS[P.ROOMS[o.room].L]}`;
  else t = `Walk them out: <b>front doors</b> or <b>pool terrace</b> · ground floor`;
  t += `<br><small style="opacity:.7">Rescued ${game.score.rescued} · gunmen down ${game.score.hostiles}${game.score.wrong ? ' · wrong calls ' + game.score.wrong : ''}</small>`;
  if ($('obj').innerHTML !== t) $('obj').innerHTML = t;
  const s = `${Math.max(0, Math.round(p.hp))} <small>HEALTH</small> ${p.reloadT > 0 ? '…' : p.ammo} / ${p.reserve} <small>${p.reloadT > 0 ? 'RELOADING' : 'PISTOL'}</small>`;
  if ($('stat').innerHTML !== s) $('stat').innerHTML = s;
  const g = o && o.phase === 'find' && o.guest.alive && o.guest.L === p.L && Math.hypot(o.guest.pos.x - p.pos.x, o.guest.pos.z - p.pos.z) < 2.6;
  $('pick').textContent = 'E — take them with you'; $('pick').style.opacity = g ? 1 : 0;
  const hurt = Math.max(0, 1 - (game.t - (game.hurtT ?? -9)) / 0.6);
  $('hurt').style.boxShadow = `inset 0 0 ${160 + (100 - p.hp)}px rgba(160,0,0,${(0.55 * hurt + (p.hp < 35 ? 0.3 : 0)).toFixed(2)})`;
  if (performance.now() - hud.msgT > 2600) $('msg').style.opacity = 0;
}

// ---------------------------------------------------------------- input (human)
// Desktop: pointer lock (mouse look, click fire). Where pointer lock isn't allowed (some embedded
// pages), drag to look and click to fire. Touch: left thumb moves, right thumb looks, buttons.
const keys = new Set(); let mdx = 0, mdy = 0, firing = false, edge = new Set(), playing = false;
const TOUCH = matchMedia('(pointer: coarse)').matches;
addEventListener('keydown', (e) => { if (!keys.has(e.code)) edge.add(e.code); keys.add(e.code); if (playing && e.code.startsWith('Arrow')) e.preventDefault(); });
addEventListener('keyup', (e) => keys.delete(e.code));
let drag = null, locked = false;
addEventListener('mousemove', (e) => {
  if (document.pointerLockElement) { mdx += e.movementX; mdy += e.movementY; }
  else if (drag && playing && !TOUCH) { mdx += e.clientX - drag[0]; mdy += e.clientY - drag[1]; if (Math.abs(e.clientX - drag[2]) + Math.abs(e.clientY - drag[3]) > 6) drag[4] = true; drag[0] = e.clientX; drag[1] = e.clientY; }
});
addEventListener('mousedown', (e) => {
  if (TOUCH || !playing || e.button !== 0) return;
  if (document.pointerLockElement) firing = true; else drag = [e.clientX, e.clientY, e.clientX, e.clientY, false];
});
addEventListener('mouseup', (e) => { if (e.button !== 0) return; firing = false; if (drag && !drag[4] && !document.pointerLockElement && playing) { tapFire = 0.12; } drag = null; });
let tapFire = 0;
// touch controls
const tc = { move: null, look: null, mx: 0, mz: 0, fire: false, use: false, run: false };
function touchUI() {
  const css = document.createElement('style');
  css.textContent = `.tb{position:fixed;width:64px;height:64px;border-radius:50%;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.35);color:#fff;font:600 11px Helvetica;display:flex;align-items:center;justify-content:center;letter-spacing:.06em;z-index:4;user-select:none;-webkit-user-select:none;touch-action:none}
  .tb.on{background:rgba(242,196,109,.45)}#stick{position:fixed;width:110px;height:110px;border-radius:50%;border:1px solid rgba(255,255,255,.25);pointer-events:none;z-index:3;display:none}
  #nub{position:absolute;left:35px;top:35px;width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,.3)}`;
  document.head.appendChild(css);
  const st = document.createElement('div'); st.id = 'stick'; st.innerHTML = '<div id="nub"></div>'; document.body.appendChild(st);
  const btn = (label, r, b, w = 64) => { const d = document.createElement('div'); d.className = 'tb'; d.textContent = label; d.style.right = r + 'px'; d.style.bottom = b + 'px'; d.style.width = d.style.height = w + 'px'; document.body.appendChild(d); return d; };
  const bf = btn('FIRE', 24, 110, 84), bu = btn('USE', 126, 40), br = btn('RELOAD', 24, 30, 56), bt = btn('TORCH', 196, 40, 56), brun = btn('RUN', 126, 120, 56);
  const hold = (el, on, off) => { el.addEventListener('touchstart', (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add('on'); on(); }, { passive: false }); el.addEventListener('touchend', (e) => { e.preventDefault(); e.stopPropagation(); el.classList.remove('on'); off && off(); }, { passive: false }); };
  hold(bf, () => { tc.fire = true; }, () => { tc.fire = false; });
  hold(bu, () => { tc.use = true; }, () => { tc.use = false; });
  hold(br, () => edge.add('KeyR'));
  hold(bt, () => edge.add('KeyF'));
  hold(brun, () => { tc.run = !tc.run; brun.classList.toggle('on', tc.run); }, () => brun.classList.toggle('on', tc.run));
  const cv = renderer.domElement;
  cv.addEventListener('touchstart', (e) => {
    e.preventDefault();
    for (const t of e.changedTouches) {
      if (t.clientX < innerWidth * 0.45 && !tc.move) { tc.move = [t.identifier, t.clientX, t.clientY]; st.style.display = 'block'; st.style.left = t.clientX - 55 + 'px'; st.style.top = t.clientY - 55 + 'px'; }
      else if (!tc.look) tc.look = [t.identifier, t.clientX, t.clientY];
    }
  }, { passive: false });
  cv.addEventListener('touchmove', (e) => {
    e.preventDefault();
    for (const t of e.changedTouches) {
      if (tc.move && t.identifier === tc.move[0]) {
        let dx = (t.clientX - tc.move[1]) / 50, dz = (t.clientY - tc.move[2]) / 50; const l = Math.hypot(dx, dz); if (l > 1) { dx /= l; dz /= l; }
        tc.mx = dx; tc.mz = -dz; document.getElementById('nub').style.transform = `translate(${dx * 35}px,${dz * 35}px)`;
      } else if (tc.look && t.identifier === tc.look[0]) { mdx += (t.clientX - tc.look[1]) * 1.6; mdy += (t.clientY - tc.look[2]) * 1.6; tc.look[1] = t.clientX; tc.look[2] = t.clientY; }
    }
  }, { passive: false });
  const end = (e) => { for (const t of e.changedTouches) { if (tc.move && t.identifier === tc.move[0]) { tc.move = null; tc.mx = tc.mz = 0; st.style.display = 'none'; document.getElementById('nub').style.transform = ''; } if (tc.look && t.identifier === tc.look[0]) tc.look = null; } };
  cv.addEventListener('touchend', end); cv.addEventListener('touchcancel', end);
}
function humanInput(dt) {
  const turn = ((keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0)) * 2.2 * dt, tilt = ((keys.has('ArrowDown') ? 1 : 0) - (keys.has('ArrowUp') ? 1 : 0)) * 1.5 * dt;
  const inp = {
    moveZ: (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0) + tc.mz, moveX: (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0) + tc.mx,
    run: keys.has('ShiftLeft') || keys.has('ShiftRight') || tc.run, lookX: mdx * 0.0022 + turn, lookY: mdy * 0.0022 + tilt,
    fire: firing || tc.fire || tapFire > 0 || keys.has('Space'), reload: edge.has('KeyR'), use: keys.has('KeyE') || tc.use, torchToggle: edge.has('KeyF'),
  };
  tapFire -= dt; mdx = mdy = 0; edge.clear(); return inp;
}

// ---------------------------------------------------------------- boot
buildWorld(scene, renderer, (s) => { $('load').textContent = 'loading ' + s; }).then(async (w) => {
  G.world = w;
  game.seed = +(q.get('seed') || 1);
  await game.init();
  game.viewCam.aspect = camera.aspect; game.viewCam.updateProjectionMatrix();
  G.ready = true; $('load').textContent = BOT ? 'bot' : 'click to start';
  // warm the shaders once so the first frame isn't a stall
  renderer.compile(scene, camera); draw();
  if (BOT) {
    $('title').style.display = 'none';
    G.tick = (inp, dt = 0.05) => { const n = Math.max(1, Math.round(dt / 0.025)); for (let i = 0; i < n; i++) game.step(dt / n, i ? { ...inp, lookX: 0, lookY: 0, torchToggle: false, reload: false } : inp); drawHud(); };
    G.draw = draw;
    return;
  }
  if (TOUCH) { touchUI(); $('tkeys').textContent = 'Tap to start · left thumb moves · right thumb looks · FIRE · USE (take a guest) · TORCH'; }
  else $('tkeys').textContent = 'Click to start · WASD move · mouse look (or drag / arrow keys) · click or Space fire · R reload · E use · Shift run · F torch · Esc pause';
  const start = () => {
    $('title').style.display = 'none'; playing = true;
    if (!TOUCH) { try { const r = renderer.domElement.requestPointerLock(); if (r && r.catch) r.catch(() => {}); } catch (e) {} }
    else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  };
  $('title').addEventListener('click', start);
  $('title').addEventListener('touchend', (e) => { e.preventDefault(); start(); });
  document.addEventListener('pointerlockchange', () => { if (document.pointerLockElement) locked = true; else if (locked) { locked = false; playing = false; $('title').style.display = 'flex'; } });
  addEventListener('keydown', (e) => { if (e.code === 'Escape' && playing && !document.pointerLockElement) { playing = false; $('title').style.display = 'flex'; } });
  let last = performance.now(), acc = 0;
  const loop = (now) => {
    requestAnimationFrame(loop);
    const el = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) {
      acc += el; const inp = humanInput(el); let first = true;
      while (acc >= 1 / 120) { game.step(1 / 120, first ? inp : { ...inp, lookX: 0, lookY: 0, reload: false, torchToggle: false }); first = false; acc -= 1 / 120; }
      drawHud();
    }
    draw();
  };
  requestAnimationFrame(loop);
});
