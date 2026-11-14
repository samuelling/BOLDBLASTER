// In-page BOLDBLASTER harness for Betamax (forward mode, no rewind, no possession).
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
  const ev = { kills: [], bursts: [], cuts: [], deaths: [] };
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
    return out;
  }

  function pickTarget(list) {
    let best = -1, bs = 1e9;
    for (const v of list) { if (!v[3] || !v[5] || v[2] > 120) continue; const sc = v[2] + 18 * Math.hypot(v[0], v[1]); if (sc < bs) { bs = sc; best = v[4]; } }
    return best;
  }

  function bot(dt, list) {
    if (G.body !== st.body) {   // new life (start, or next man up)
      st.body = G.body; st.anchor = p.pos.clone(); st.tgt = -1; st.lookYaw = null; st.burst = 0; st.pause = 0.6; }
    inp.lookX = inp.lookY = 0; inp.fire = false; inp.jump = false;
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
        window.BWstep(1, DT, k === 1);
        st.t += DT; st.tick++;
        if (G.body !== b0 && G.forward) ev.deaths.push([st.t]);
        if (G.fwd && G.fwd.kills > st.kills) {
          const tv = list.find((v) => v[4] === st.tgt) || [0, 0];
          for (let n = st.kills; n < G.fwd.kills; n++) ev.kills.push([st.t, tv[0], tv[1]]);
          st.kills = G.fwd.kills;
        } else if (G.fwd && G.fwd.kills < st.kills) st.kills = G.fwd.kills;   // a restart resets the count
      }
      list = scan();
      const rec = { t: +st.t.toFixed(2), tau: +G.tau.toFixed(2), body: G.body, x: +p.pos.x.toFixed(2), z: +p.pos.z.toFixed(2),
        yaw: +p.yaw.toFixed(4), pitch: +p.pitch.toFixed(4), speed: +Math.hypot(p.vel.x, p.vel.z).toFixed(3),
        vis: list.map((v) => [+v[0].toFixed(2), +v[1].toFixed(2), +v[2].toFixed(1), v[3], v[4], v[5]]) };
      return { img: grab(), rec };
    },
    events() { return ev; },
  };
};
