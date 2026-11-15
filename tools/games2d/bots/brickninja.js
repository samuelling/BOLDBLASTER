// Brick Ninja bot for the BOLDBLASTER harness. Runs in the page's global scope, so it reads the
// game's own top-level state (P, enemies, studs, camX, camY, W, H, state, input, LEVELS...).
window.GAMEBOT = (() => {
  let s = 1;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const WORLDS = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11];     // side-scrolling worlds (no race, vertical climb or dragon flight)
  const ev = { kills: [], bursts: [], deaths: [], cuts: [] };
  const st = { t: 0, order: [], prog: 0, lastX: 0, stuckT: 0, jumpHold: 0, back: 0, pending: [], tgt: null, fireT: 0, dead: new Set(), hpLast: 0, li: -1 };
  const scr = (x, y) => [((x - camX) / W - 0.5) * 24, (0.5 - (y - camY) / H) * 18];
  const onScreen = (x, y) => x > camX - 4 && x < camX + W + 4 && y > camY - 4 && y < camY + H + 4;

  function nextWorld() {
    if (!st.order.length) { st.order = WORLDS.slice(); for (let i = st.order.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [st.order[i], st.order[j]] = [st.order[j], st.order[i]]; } }
    const i = st.order.pop();
    beginLevel(i); st.li = i; st.lastX = P.x; st.stuckT = 0; st.dead = new Set(); st.pending = [];
    ev.cuts.push([st.t, i]);
  }
  function jump() { if (!input.jump) input.jumpQ = 0.14; input.jump = true; st.jumpHold = 0.18 + rnd() * 0.15; }

  function control(dt) {
    st.t += dt;
    if (state === 'down') { ev.deaths.push([st.t]); document.getElementById('continueBtn').click(); return; }
    if (state === 'pause') resumeGame();
    if (state !== 'play' || ROOM || !L || L.race || L.flight || L.vertical) { if (state !== 'play' || st.li < 0 || L.race || L.flight || L.vertical) nextWorld(); return; }
    // hold the jump button briefly (higher jumps), then let go so the next press registers
    if (st.jumpHold > 0) { st.jumpHold -= dt; if (st.jumpHold <= 0) input.jump = false; }
    const cx = P.x + P.w / 2, cy = P.y + P.h / 2;
    // target: nearest hittable enemy on screen
    let best = null, bd = 1e9;
    for (const e of enemies) {
      if (!hittable(e) || !onScreen(ecx(e), ecy(e))) continue;
      const d = Math.hypot(ecx(e) - cx, ecy(e) - cy); if (d < bd) { bd = d; best = e; }
    }
    if (boss && hittable(boss) && onScreen(ecx(boss), ecy(boss))) { const d = Math.hypot(ecx(boss) - cx, ecy(boss) - cy); if (d < bd) { bd = d; best = boss; } }
    st.tgt = best;
    // move: forward, a little hesitation near enemies, back off now and then when stuck
    if (st.back > 0) { st.back -= dt; input.left = true; input.right = false; }
    else { input.left = false; input.right = !(best && bd < 70 && ecx(best) > cx && rnd() < 0.5); }
    // gaps, walls, hazards ahead -> jump; falling over a gap -> double jump
    const ahead = P.x + P.w + 14;
    const groundAhead = solidAt(ahead, P.y + P.h + 6, true) || solidAt(ahead + 22, P.y + P.h + 6, true) || solidAt(ahead, P.y + P.h + 40, true);
    const wallAhead = solidAt(P.x + P.w + 8, P.y + P.h * 0.5, false);
    const hazAhead = [...spikes, ...lavaPools].some((h) => h.x < ahead + 40 && h.x + (h.w || 20) > ahead - 10);
    if (P.onG && (!groundAhead || wallAhead || hazAhead) && input.right) jump();
    else if (!P.onG && P.vy > 60 && !solidAt(cx, P.y + P.h + 140, true) && rnd() < 0.3) jump();
    else if (P.cling > 0 && rnd() < 0.2) jump();
    // stuck detection
    if (P.x > st.lastX + 20) { st.lastX = P.x; st.stuckT = 0; } else st.stuckT += dt;
    if (st.stuckT > 1.5 && P.onG && rnd() < 0.1) jump();
    if (st.stuckT > 5) { st.back = 0.4 + rnd() * 0.6; st.stuckT = 2; }
    // attack: sword up close, aimed stars otherwise (like touching where you want them to go)
    input.sword = !!(best && bd < 55 && rnd() < 0.5);
    if (best && P.cd <= 0 && bd < W * 0.75 && (st.fireT -= dt) <= 0) {
      const dx = ecx(best) - cx, dy = ecy(best) - (P.y + 15);
      throwStar(Math.atan2(dy, dx) + gauss() * 0.06);
      const sc = scr(ecx(best), ecy(best));
      st.pending.push({ rec: [st.t, bd, Math.hypot(best.vx || 0, best.vy || 0), sc[0], sc[1], enemies.indexOf(best)], e: best, hp: best.hp, until: st.t + 0.8 });
      st.fireT = 0.15 + rnd() * 0.6;
    }
    for (let i = st.pending.length - 1; i >= 0; i--) {
      const q = st.pending[i], hit = q.e.dead || q.e.hp < q.hp;
      if (hit || st.t > q.until) { ev.bursts.push(q.rec.concat([hit ? 1 : 0])); st.pending.splice(i, 1); }
    }
    // kills: enemies taken down by the player (hp at or below zero)
    for (const e of enemies.concat(boss ? [boss] : [])) {
      if (e.dead && e.hp <= 0 && !st.dead.has(e)) { st.dead.add(e); const sc = scr(ecx(e), ecy(e)); ev.kills.push([st.t, sc[0], sc[1]]); }
    }
  }

  function record() {
    const vis = [];
    enemies.concat(boss ? [boss] : []).forEach((e, i) => {
      if (!hittable(e) || !onScreen(ecx(e), ecy(e))) return;
      const sc = scr(ecx(e), ecy(e));
      vis.push([+sc[0].toFixed(2), +sc[1].toFixed(2), +Math.hypot(ecx(e) - P.x, ecy(e) - P.y).toFixed(0), 1, i, 1]);
    });
    let items = 0; for (const q of studs) if (!q.got && onScreen(q.x, q.y)) items++;
    return { t: +st.t.toFixed(2), lv: LI, x: +P.x.toFixed(1), camX: +camX.toFixed(2), camY: +camY.toFixed(2), hp: P.hp, items, vis };
  }

  return {
    setup(seed) { s = seed >>> 0; try { soundOn = false; musicOn = false; } catch (e) {} nextWorld(); },
    canvas: () => document.getElementById('game'),
    control, record,
    events: () => ev,
    resetEvents() { for (const k in ev) ev[k].length = 0; st.t = 0; },
  };
})();
