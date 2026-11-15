// In-page BOLDBLASTER harness for Betamax with the Diggings (forward mode, no rewind, no possession).
// As bot.js, plus tunnel missions: down a hatch, through the warren to a magazine, ammunition back up
// to the line, rerouting round cave-ins, homing to the ladder when the lights go out, and deciding
// about the figures in the gloom.
// Installs a bot that drives G.input, steps the game on a fixed clock, and returns
// one 192x144 luma frame plus a log record per 10 Hz frame.
window.BBX = (seed) => {
  const G = window.BW, S = G.S, pb = G.pb, cam = G.camera, p = G.player;
  const V3 = cam.position.constructor;
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  G.hold = true;
  G.input.settings.vcr = false;
  // no audio in the scanner sim: sound banks are only built on a user click, so stub them
  const nul = new Proxy({}, { get: () => null });
  for (const k of ['B', 'R']) if (!G.audio[k]) G.audio[k] = nul;
  let inp = { moveX: 0, moveZ: 0, lookX: 0, lookY: 0, fire: false, ads: false, crouch: false, jump: false, reload: false, sprint: false, touchLook: false, slot: 0, use: false };
  G.input.read = () => { const o = Object.assign({}, inp); inp.reload = false; return o; };

  const st = { t: 0, tgt: -1, react: 0, aimOff: [0, 0], burst: 0, pause: 0, strafe: 0, strafeT: 0, crouch: false,
    lookYaw: null, lookPitch: 0, lookT: 0, anchor: null, body: -1, seen: new Map(), tick: 0,
    kills: 0, burstRec: null, hp0: 0, settle: -1 };
  const ev = { kills: [], bursts: [], cuts: [], deaths: [], junctions: [], stale: [], homing: [], figures: [], missions: [], digev: [], plan: G.dig.describe() };
  const digN = {};
  const D = G.dig, P = D.P, ek = (a, b) => (a < b ? a + ':' + b : b + ':' + a);
  const ms = { phase: 'none', at: 0, hatch: null, home: null, known: new Set(), path: [], cur: -1, prev: -1, next: -1, goal: -1, trial: null, fig: null, figT: new Map(), stuck: 0, lastPos: null, pressUse: false, dark: false };
  // shortest believed routes: the plan's links minus the cave-ins this soldier has run into
  function dij(src) {
    const dist = new Map([[src, 0]]), q = [[0, src]];
    while (q.length) { q.sort((a, b) => a[0] - b[0]); const [d, a] = q.shift(); if (d > dist.get(a)) continue;
      for (const b of P.adj.get(a)) { if (ms.known.has(ek(a, b))) continue; const nd = d + 1; if (nd < (dist.get(b) ?? 1e9)) { dist.set(b, nd); q.push([nd, b]); } } }
    return dist;
  }
  function trueDist(src) {
    const dist = new Map([[src, 0]]), q = [src];
    while (q.length) { const a = q.shift(); for (const b of P.adj.get(a)) { if (!D.linkOpen(a, b)) continue; if (!dist.has(b)) { dist.set(b, dist.get(a) + 1); q.push(b); } } }
    return dist;
  }
  function goalCell() {
    if (ms.trial) return ms.home.cell;
    if (!D.state.carrying) { const ds = dij(ms.cur); let b = -1, bd = 1e9; for (const c of P.crates) { const d = ds.get(c.cell) ?? 1e9; if (d < bd) { bd = d; b = c.cell; } } return b; }
    const ds = dij(ms.cur); let b = -1, bd = 1e9; for (const h of P.hatches) { if (h.kind !== 'line') continue; const d = ds.get(h.cell) ?? 1e9; if (d < bd) { bd = d; b = h.cell; } } return b;
  }
  function chooseNext() {   // at a cell centre: pick the next cell toward the goal (softmax over believed route lengths)
    ms.goal = goalCell(); if (ms.goal < 0 || ms.cur === ms.goal) return -1;
    const dg = dij(ms.goal), tg = trueDist(ms.goal);
    let opts = P.adj.get(ms.cur).filter((b) => !ms.known.has(ek(ms.cur, b)));
    if (!opts.length) return -1;
    if (opts.length > 1 && opts.includes(ms.prev)) opts = opts.filter((b) => b !== ms.prev);   // don't turn straight back unless it's a dead end
    const L = opts.map((b) => 1 + (dg.get(b) ?? 99)), T = opts.map((b) => 1 + (tg.get(b) ?? 99));
    const temp = ms.dark ? 1.6 : 0.35, w = L.map((l) => Math.exp(-(l - Math.min(...L)) / temp)), sw = w.reduce((a, b) => a + b, 0);
    let u = rnd() * sw, k = 0; for (; k < w.length - 1; k++) { u -= w[k]; if (u <= 0) break; }
    if (opts.length >= 2)
      ev.junctions.push([st.t, ms.cur, ms.prev, opts, L, T, opts[k], ms.goal, ms.dark ? 1 : 0, D.state.carrying ? 1 : 0, ms.trial ? 1 : 0]);
    return opts[k];
  }
  const tmp = new V3(), eye = new V3(), aim = new V3();
  const enemySide = () => 1 - S.actors[G.body].side;
  const hpSum = () => { let t = 0; for (let i = 0; i < S.N; i++) t += Math.max(0, G.enemyHp[i]); return t; };

  // screen geometry: the 384x288 canvas fills a 24 x 18 deg display
  function screen(v) {
    tmp.copy(v).applyMatrix4(cam.matrixWorldInverse); if (tmp.z > -0.2) return null;
    tmp.copy(v).project(cam);
    if (Math.abs(tmp.x) > 1 || Math.abs(tmp.y) > 1) return null;
    return [tmp.x * 12, tmp.y * 9];
  }
  function losClear(a, b) {   // terrain only (dunes are the occluders on the beach)
    for (let k = 1; k < 12; k++) { const f = k / 12, x = a.x + (b.x - a.x) * f, y = a.y + (b.y - a.y) * f, z = a.z + (b.z - a.z) * f;
      if (G.heightAt(x, z) > y + 0.05) return false; }
    return true;
  }
  function chest(i, out) { const h = pb.hit[i]; return out.copy(h.neck).multiplyScalar(0.6).addScaledVector(h.pelvis, 0.4); }

  function restart() {
    endMission('restart');
    const life = (q) => (q.deathTime > 0 ? q.deathTime : 1e9);
    for (let tries = 0; tries < 50; tries++) {
      const t0 = 30 + rnd() * 140;
      // all of the Cape's men come in by helicopter: take one who has been on the sand at least 15 s
      const el = S.actors.filter((q) => q.side === 0 && q.spawn + 15 <= t0 && life(q) > t0 + 25);
      if (!el.length) continue;
      const a = el[Math.floor(rnd() * el.length)];
      window.BWfwd(a.id, t0);
      G.hold = true; window.BWstep(1, 0.05, false);
      if (p.ride || G.state !== 'play' || !G.forward) continue;   // still aboard something: pick again
      ev.cuts.push([st.t, 0]); st.body = -1; return;
    }
  }

  function scan() {   // every soldier on screen this frame: [xdeg, ydeg, dist, visible, id, enemy]
    const out = []; eye.copy(cam.position); const es = enemySide();
    for (let i = 0; i < S.N; i++) {
      if (i === G.body) continue; const h = pb.hit[i]; if (!h.ok || !pb.alive(i, G.tau)) continue;
      chest(i, aim); const d = eye.distanceTo(aim); if (d > 220) continue;
      const sc = screen(aim); if (!sc) continue;
      const vis = d < 140 && losClear(eye, aim);
      out.push([sc[0], sc[1], d, vis ? 1 : 0, i, S.actors[i].side === es ? 1 : 0]);
    }
    if (D.underground(p.pos.y)) for (const f of D.figs) {   // figures in the tunnels, standing (chest height)
      if (f.down) continue; aim.set(f.x, 0.2 + 1.2, f.z); const d = eye.distanceTo(aim); if (d > 25) continue;
      const sc = screen(aim); if (!sc) continue;
      out.push([sc[0], sc[1], d, D.clear(p.pos.x, p.pos.z, f.x, f.z) ? 1 : 0, 1000 + f.id, 1]);
    }
    return out;
  }

  function pickTarget(list) {
    let best = -1, bs = 1e9;
    for (const v of list) { if (!v[3] || !v[5] || v[2] > 120) continue; const sc = v[2] + 18 * Math.hypot(v[0], v[1]); if (sc < bs) { bs = sc; best = v[4]; } }
    return best;
  }

  function endMission(why) {
    if (ms.phase !== 'none') ev.missions.push([ms.t0, st.t, ms.hatch ? ms.hatch.id : -1, why]);
    if (ms.trial) { ms.trial.push(st.t, p.pos.x, p.pos.z, why === 'home' ? 1 : 0); ev.homing.push(ms.trial); ms.trial = null; }
    ms.phase = 'none'; ms.fig = null; ms.dark = false; st.anchor = p.pos.clone(); st.lookYaw = null; ms.at = st.t + 40 + rnd() * 60;
  }
  function steer(tx, tz, dt, speed) {   // turn toward a point and walk
    const yd = Math.atan2(-(tx - p.pos.x), -(tz - p.pos.z));
    const g = Math.min(1, dt * 6); const dy = wrap(yd - p.yaw) * g + gauss() * 0.001;
    const dpch = (0 - p.pitch) * Math.min(1, dt * 4);
    inp.lookX = -dy / (0.0022 * (1 - p.ads * 0.32)); inp.lookY = -dpch / (0.0022 * (1 - p.ads * 0.32));
    inp.moveZ = Math.abs(wrap(yd - p.yaw)) < 0.9 ? speed : 0.1; inp.moveX = 0; inp.ads = false; inp.crouch = false; inp.sprint = false;
  }
  function figureCheck(dt) {   // a figure ahead in the gloom: stop, look, decide
    if (ms.fig) {
      const f = D.figs[ms.fig.id]; ms.fig.t += dt;
      const fx = f.x, fz = f.z; tmp.set(fx, 0.2 + 1.2, fz); eye.copy(cam.position);
      const yd = Math.atan2(-(fx - eye.x), -(fz - eye.z)), pd = Math.atan2(1.4 - eye.y, Math.hypot(fx - eye.x, fz - eye.z));
      const g = Math.min(1, dt * 8), sens = 0.0022 * (1 - p.ads * 0.32);
      inp.lookX = -(wrap(yd - p.yaw) * g) / sens; inp.lookY = -((pd - p.pitch) * g) / sens; inp.moveZ = 0; inp.moveX = 0; inp.ads = true;
      if (ms.fig.t >= ms.fig.rt) {
        if (ms.fig.choice === 1 && !f.down && ms.fig.t < ms.fig.rt + 0.5) { inp.fire = st.tick % 3 === 0; return true; }
        ev.figures.push([ms.fig.t0, ms.fig.rt, ms.fig.choice, f.foe ? 1 : 0, ms.fig.dist, ms.fig.lit, ms.fig.q, f.down ? 1 : 0, f.id]);
        ms.figT.set(f.id, st.t); ms.fig = null; inp.ads = false;
      }
      return true;
    }
    for (const f of D.figs) {
      if (f.down || st.t - (ms.figT.get(f.id) ?? -1e9) < 60) continue;
      const d = Math.hypot(f.x - p.pos.x, f.z - p.pos.z); if (d > 13 || d < 1) continue;
      const ang = Math.abs(wrap(Math.atan2(-(f.x - p.pos.x), -(f.z - p.pos.z)) - p.yaw)); if (ang > 0.7) continue;
      if (!D.clear(p.pos.x, p.pos.z, f.x, f.z)) continue;
      // evidence: how well you can see him (sector lights or just the lantern) and how far off he is
      const lit = D.zoneOn[D.zoneAt(f.x, f.z)] > 0.5 ? 1 : 0, q = (lit ? 1 : 0.35) * Math.max(0.05, 1 - d / 16);
      const v = 3.2 * q * (f.foe ? 1 : -1), a = 1.3; let x = 0, tt = 0;
      while (Math.abs(x) < a / 2 && tt < 4) { x += v * 0.01 + Math.sqrt(0.01) * gauss(); tt += 0.01; }
      ms.fig = { id: f.id, t: 0, t0: st.t, rt: tt + 0.3, choice: x > 0 ? 1 : 0, dist: d, lit, q };
      return true;
    }
    return false;
  }
  function mission(dt) {
    const under = D.underground(p.pos.y);
    if (D.state.climb) { inp.moveZ = 0; inp.moveX = 0; return true; }
    if (ms.phase === 'none') {
      if (under) { ms.phase = 'route'; ms.t0 = st.t; ms.home = D.near(P.hatches, p.pos.x, p.pos.z, 99) || P.hatches[0]; ms.cur = D.cellAt(p.pos.x, p.pos.z); ms.next = -1; ms.prev = -1; return true; }
      if (st.t < ms.at || st.tgt >= 0 && st.burst > 0) return false;
      const h = D.near(P.hatches, p.pos.x, p.pos.z, 75);
      if (!h || rnd() < 0.3) { ms.at = st.t + 30 + rnd() * 30; return false; }
      ms.phase = 'toHatch'; ms.hatch = h; ms.t0 = st.t; ms.stuck = 0; ms.lastPos = p.pos.clone(); st.tgt = -1; return true;
    }
    if (ms.phase === 'toHatch') {
      const h = ms.hatch, d = Math.hypot(h.x - p.pos.x, h.z - p.pos.z);
      if (d < 1.0) { inp.use = true; inp.moveZ = 0; ms.phase = 'down'; return true; }
      steer(h.x, h.z, dt, d > 6 ? 1 : 0.6); if (d > 10) inp.sprint = true;
      ms.stuck += dt; if (ms.stuck > 2) { if (p.pos.distanceTo(ms.lastPos) < 1.0) { inp.moveX = rnd() < 0.5 ? -1 : 1; ms.stuckN = (ms.stuckN || 0) + 1; } ms.stuck = 0; ms.lastPos.copy(p.pos); }
      if (st.t - ms.t0 > 45 || (ms.stuckN || 0) > 6) { ms.stuckN = 0; endMission('lost'); ms.at = st.t + 30; }
      return true;
    }
    if (ms.phase === 'down') {
      if (under) { ms.phase = 'route'; ms.home = ms.hatch; ms.cur = D.cellAt(p.pos.x, p.pos.z); ms.prev = -1; ms.next = -1; ms.dark = false; ms.outbound = [[st.t, p.pos.x, p.pos.z]]; }
      else { if (st.t - ms.t0 > 60) endMission('lost'); inp.use = rnd() < 0.2; }
      return true;
    }
    if (ms.phase === 'up') { if (!under) endMission(ms.trial ? 'home' : 'delivered'); return true; }
    // ---- underground
    if (!under) { endMission('surfaced'); return true; }
    if (ms.outbound && st.tick % 4 === 0) ms.outbound.push([st.t, p.pos.x, p.pos.z]);
    const zn = D.zoneAt(p.pos.x, p.pos.z), dark = D.zoneOn[zn] < 0.5;
    if (dark && !ms.trial) {   // the lights go: find your way back to the ladder you came down
      ms.trial = [st.t, p.pos.x, p.pos.z, ms.home.id, ms.home.x, ms.home.z, ms.outbound ? ms.outbound.slice(-400) : []];
      ms.dark = true; ms.next = -1;
    }
    if (ms.trial && !dark) {   // lights back before you reached the ladder: the trial ends where you are
      ms.trial.push(st.t, p.pos.x, p.pos.z, 2); ev.homing.push(ms.trial); ms.trial = null; ms.next = -1;
    }
    ms.dark = dark || !!ms.trial;
    if (!ms.trial && figureCheck(dt)) return true;
    const speed = ms.dark ? 0.45 : 0.85;
    if (ms.next < 0) {
      const [cx0, cz0] = D.cellXZ(ms.cur), dc = Math.hypot(cx0 - p.pos.x, cz0 - p.pos.z);
      if (dc > 0.45) { steer(cx0, cz0, dt, 0.6); return true; }
      // at the goal?
      const g = goalCell();
      if (g === ms.cur) {
        if (ms.trial || D.state.carrying) {
          const h = P.hatches.find((q) => q.cell === ms.cur);
          if (Math.hypot(h.x - p.pos.x, h.z - p.pos.z) > 1.1) { steer(h.x, h.z, dt, 0.4); return true; }
          inp.use = true; ms.phase = 'up'; return true;
        }
        inp.use = true; inp.moveZ = 0; return true;   // the magazine: pick up a box
      }
      ms.next = chooseNext(); if (ms.next < 0) { endMission('stuck'); return true; }
    }
    // heading for the next cell: is the way caved in? (you see the heap a few metres off in the light)
    const [nx, nz] = D.cellXZ(ms.next), [cx1, cz1] = D.cellXZ(ms.cur), mx = (nx + cx1) / 2, mz = (nz + cz1) / 2;
    if (!D.linkOpen(ms.cur, ms.next) && Math.hypot(mx - p.pos.x, mz - p.pos.z) < (ms.dark ? 1.8 : 4.5)) {
      ms.known.add(ek(ms.cur, ms.next)); ev.stale.push([st.t, ms.cur, ms.next, ms.dark ? 1 : 0]); ms.next = -1; return true;
    }
    if (Math.hypot(nx - p.pos.x, nz - p.pos.z) < 0.45) { ms.prev = ms.cur; ms.cur = ms.next; ms.next = -1; return true; }
    steer(nx, nz, dt, speed);
    // wobble a little in the dark (you can't see the walls coming)
    if (ms.dark) inp.moveX = gauss() * 0.15;
    return true;
  }

  function bot(dt, list) {
    if (G.body !== st.body) {   // new life (start, or next man up)
      st.body = G.body; st.anchor = p.pos.clone(); st.tgt = -1; st.lookYaw = null; st.burst = 0; st.pause = 0.6;
      endMission('newlife'); ms.at = st.t + 4 + rnd() * 14; }
    inp.lookX = inp.lookY = 0; inp.fire = false; inp.jump = false; inp.use = false;
    if (mission(dt)) return;
    // target upkeep, every ~0.3 s
    if (st.tick % 6 === 0) {
      const tv = list.find((v) => v[4] === st.tgt);
      if (!tv || !tv[3] || !pb.alive(st.tgt, G.tau)) {
        const nt = pickTarget(list);
        if (nt !== st.tgt) { st.tgt = nt; st.react = 0.15 + rnd() * 0.25; st.aimOff = [gauss() * 0.02, gauss() * 0.015]; st.crouch = rnd() < 0.35; }
      }
    }
    const sens = 0.0022 * (1 - p.ads * 0.32);
    let dyaw = 0, dpitch = 0, err = 99, dist = 999;
    if (st.tgt >= 0 && pb.hit[st.tgt].ok) {
      chest(st.tgt, aim); eye.copy(cam.position);
      const dx = aim.x - eye.x, dy = aim.y - eye.y, dz = aim.z - eye.z, hz = Math.hypot(dx, dz); dist = Math.hypot(hz, dy);
      const yd = Math.atan2(-dx, -dz) + st.aimOff[0], pd = Math.atan2(dy, hz) + st.aimOff[1];
      dyaw = wrap(yd - p.yaw); dpitch = pd - p.pitch;
      st.aimOff[0] *= Math.exp(-dt * 1.5); st.aimOff[1] *= Math.exp(-dt * 1.5);
      cam.getWorldDirection(tmp); err = tmp.angleTo(aim.clone().sub(eye)) * 180 / Math.PI;
      if ((st.react -= dt) > 0) { dyaw = dpitch = 0; }
      else { const g = Math.min(1, dt * 10); dyaw = dyaw * g + gauss() * 0.0015; dpitch = dpitch * g + gauss() * 0.0012; }
      inp.ads = dist > 25; inp.crouch = st.crouch;
      // fire in bursts once on target
      const tol = Math.max(0.9, 2.6 * 20 / Math.max(dist, 1));
      if (st.burst > 0) { st.burst -= dt; inp.fire = p.gun.userData.auto ? true : st.tick % 3 === 0; if (st.burst <= 0) { st.pause = 0.3 + rnd() * 0.7; st.settle = 0.2; } }
      else if ((st.pause -= dt) <= 0 && st.react <= 0 && err < tol) {
        st.burst = 0.25 + rnd() * 0.35; inp.fire = true;
        const sc = screen(aim) || [0, 0];
        st.burstRec = [st.t, dist, err, sc[0], sc[1], st.tgt]; st.hp0 = hpSum(); st.k0 = G.fwd ? G.fwd.kills : 0;
      }
      // strafe while engaged
      if ((st.strafeT -= dt) <= 0) { st.strafeT = 0.8 + rnd() * 2; st.strafe = [-0.7, 0, 0, 0.7][Math.floor(rnd() * 4)]; }
      inp.moveX = st.strafe; inp.moveZ = dist > 70 ? 0.6 : 0; inp.sprint = false;
    } else {
      inp.ads = false; inp.crouch = false;
      if ((st.lookT -= dt) <= 0) {
        st.lookT = 1.5 + rnd() * 3;
        let near = -1, nd = 1e9; const es = enemySide();
        for (let i = 0; i < S.N; i++) { if (S.actors[i].side !== es || !pb.hit[i].ok || !pb.alive(i, G.tau)) continue; const d = pb.hit[i].pos.distanceTo(p.pos); if (d < nd) { nd = d; near = i; } }
        if (near >= 0 && rnd() < 0.65) { const h = pb.hit[near].pos; st.lookYaw = Math.atan2(-(h.x - p.pos.x), -(h.z - p.pos.z)) + gauss() * 0.3; }
        else st.lookYaw = p.yaw + (rnd() - 0.5) * 2.4;
        st.lookPitch = gauss() * 0.05;
        st.walk = rnd() < 0.6; st.sprint = rnd() < 0.25;
      }
      const g = Math.min(1, dt * 3.5);
      dyaw = wrap(st.lookYaw - p.yaw) * g + gauss() * 0.001; dpitch = (st.lookPitch - p.pitch) * g;
      // leash: stay within 28 m of where this life began (the beach's edges are mined)
      const off = Math.hypot(p.pos.x - st.anchor.x, p.pos.z - st.anchor.z);
      if (off > 28) { st.lookYaw = Math.atan2(-(st.anchor.x - p.pos.x), -(st.anchor.z - p.pos.z)); st.walk = true; }
      inp.moveZ = st.walk ? 0.8 : 0; inp.moveX = 0; inp.sprint = st.walk && st.sprint;
      if (p.ammo < 8 && p.reloadT < 0) inp.reload = true;
    }
    inp.lookX = -dyaw / sens; inp.lookY = -dpitch / sens;
    // close out a burst: did any round land?
    if (st.burstRec && st.burst <= 0 && (st.settle -= dt) <= 0) {
      const hit = hpSum() < st.hp0 - 0.01 || (G.fwd && G.fwd.kills > st.k0);
      ev.bursts.push(st.burstRec.concat([hit ? 1 : 0])); st.burstRec = null;
    }
  }

  // capture: readPixels straight off the drawn WebGL canvas -> Rec.709 luma (display-encoded, like the
  // raycaster's pixel values), 2x2 box -> 192x144. GL rows run bottom-up, so flip.
  const glc = document.querySelector('canvas').getContext('webgl2'), d = new Uint8Array(384 * 288 * 4);
  function grab() {
    glc.readPixels(0, 0, 384, 288, glc.RGBA, glc.UNSIGNED_BYTE, d);
    const out = new Uint16Array(192 * 144);
    for (let y = 0; y < 144; y++) for (let x = 0; x < 192; x++) {
      let acc = 0; for (let yy = 0; yy < 2; yy++) for (let xx = 0; xx < 2; xx++) { const k = ((287 - (2 * y + yy)) * 384 + 2 * x + xx) * 4; acc += 0.2126 * d[k] + 0.7152 * d[k + 1] + 0.0722 * d[k + 2]; }
      out[y * 192 + x] = Math.round(acc / 4 / 255 * 65535);
    }
    const u8 = new Uint8Array(out.buffer); let bin = ''; for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
    return btoa(bin);
  }

  // instance culling for the soldier mesh: draw only soldiers in (a generous cone around) the view and
  // within 260 m. Pure render optimisation: soldiers outside the view or beyond ~1 px are never on screen.
  const sg = G.sol.mesh.geometry, NB = G.sol.boneData.length / (S.N * 16);
  const aI = sg.attributes.aInst, aS = sg.attributes.aSide, aT = sg.attributes.aTone;
  const side0 = Float32Array.from(aS.array), tone0 = Float32Array.from(aT.array);
  const fw = new V3(), rel = new V3(), cosLim = Math.cos(67 * Math.PI / 180);
  function cull() {
    cam.getWorldDirection(fw); const c = cam.position, bd = G.sol.boneData; let n = 0;
    for (let i = 0; i < S.N; i++) {
      const o = i * NB * 16; rel.set(bd[o + 12] - c.x, bd[o + 13] - c.y, bd[o + 14] - c.z);
      const d = rel.length(); if (d > 260) continue;
      if (d > 6 && rel.dot(fw) < cosLim * d) continue;
      aI.array[n] = i; aS.array[n] = side0[i]; aT.array[n] = tone0[i]; n++;
    }
    sg.instanceCount = n; aI.needsUpdate = aS.needsUpdate = aT.needsUpdate = true;
    return n;
  }
  window.BBXcull = cull;
  const shadows = !!window.BBX_SHADOWS;
  G.post.composer.renderer.shadowMap.enabled = shadows;

  restart();
  const DT = 0.05;
  return {
    frame() {   // one 10 Hz frame = two 50 ms game ticks, the second drawn
      let list = null;
      for (let k = 0; k < 2; k++) {
        if (!G.forward || G.state !== 'play') restart();
        const b0 = G.body;
        list = scan(); bot(DT, list);
        if (k === 1) cull();
        window.BWstep(1, DT, k === 1 && !window.BBX_NODRAW);
        st.t += DT; st.tick++;
        if (G.body !== b0 && G.forward) ev.deaths.push([st.t]);
        if (G.fwd && G.fwd.kills > st.kills) {
          const tv = list.find((v) => v[4] === st.tgt) || [0, 0];
          for (let n = st.kills; n < G.fwd.kills; n++) ev.kills.push([st.t, tv[0], tv[1]]);
          st.kills = G.fwd.kills;
        } else if (G.fwd && G.fwd.kills < st.kills) st.kills = G.fwd.kills;   // a restart resets the count
      }
      for (const k in D.log) { const L = D.log[k]; if (!digN[k]) digN[k] = 0; while (digN[k] < L.length) { ev.digev.push([st.t, k].concat(L[digN[k]].slice(1))); digN[k]++; } }
      list = scan();
      const rec = { t: +st.t.toFixed(2), tau: +G.tau.toFixed(2), body: G.body, x: +p.pos.x.toFixed(2), z: +p.pos.z.toFixed(2),
        yaw: +p.yaw.toFixed(4), pitch: +p.pitch.toFixed(4), speed: +Math.hypot(p.vel.x, p.vel.z).toFixed(3),
        vis: list.map((v) => [+v[0].toFixed(2), +v[1].toFixed(2), +v[2].toFixed(1), v[3], v[4], v[5]]),
        y: +p.pos.y.toFixed(2), under: D.underground(p.pos.y) ? 1 : 0, cell: D.cellAt(p.pos.x, p.pos.z), dark: D.zoneOn[D.zoneAt(p.pos.x, p.pos.z)] < 0.5 ? 1 : 0,
        carry: D.state.carrying ? 1 : 0, climb: D.state.climb ? 1 : 0, phase: ms.phase, wall: D.underground(p.pos.y) ? +D.wallDist(p.pos.x, p.pos.z).toFixed(2) : -1 };
      return { img: window.BBX_NODRAW ? '' : grab(), rec };
    },
    events() { ev.dig = D.log; return ev; },
  };
};
