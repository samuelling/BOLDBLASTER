// Animal Sanctuary bot for the BOLDBLASTER harness (runs inside the game's closure via __eval).
// Explores the wild zones, walks up to creatures, plays the befriending timing game with human-like
// timing error, picks berries when no creature is around, and moves between zones now and then.
window.GAMEBOT = (() => {
  let s = 1;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const ZS = ['meadow', 'woods', 'peaks', 'lake', 'plains'];
  const ev = { kills: [], bursts: [], deaths: [], cuts: [] };
  const st = { t: 0, zoneT: 0, zoneLen: 150, lastP: [0, 0], stuckT: 0, detour: null, detourT: 0, err: null, prevPos: null, ids: new Map(), nid: 0, wander: null, wanderT: 0 };
  const scr = (x, y) => [((x - camX) / VW - 0.5) * 24, (0.5 - (y - camY) / VH) * 18];
  const onScreen = (x, y) => x > camX - 4 && x < camX + VW + 4 && y > camY - 4 && y < camY + VH + 4;
  const id = (o) => { if (!st.ids.has(o)) st.ids.set(o, st.nid++); return st.ids.get(o); };
  function setKeys(dx, dy) {
    for (const k of ['L', 'R', 'U', 'D']) keys.delete(k);
    if (dx < -2) keys.add('L'); if (dx > 2) keys.add('R'); if (dy < -2) keys.add('U'); if (dy > 2) keys.add('D');
  }
  function newZone() {
    if (scene === 'wild' && wild) returnHome();
    if (panelOpen) closePanel();
    const zk = ZS[Math.floor(rnd() * ZS.length)];
    startExpedition(zk);
    st.zoneT = 0; st.zoneLen = 100 + rnd() * 100; st.ids = new Map();
    ev.cuts.push([st.t, ZS.indexOf(zk)]);
  }

  function control(dt) {
    st.t += dt; st.zoneT += dt;
    if (titleOpen) { titleOpen = false; titleEl.hidden = true; }
    if (panelOpen) closePanel();
    if (!state.tutDone) { state.tutDone = true; try { tutEl.hidden = true; stage.classList.remove('tut'); } catch (e) {} }
    if (scene !== 'wild' || !wild || st.zoneT > st.zoneLen) { if (!catchG) { newZone(); return; } }
    // the befriending game: press when the marker reaches the zone centre, with timing error
    if (catchG) {
      setKeys(0, 0);
      const g = catchG;
      if (g.res) return;
      if (st.err === null) st.err = gauss() * 0.13;   // timing error, in bar widths
      const aim = g.zs + g.zw / 2 + st.err;
      const crossed = st.prevPos !== null && ((st.prevPos - aim) * (g.pos - aim) <= 0);
      st.prevPos = g.pos;
      if (crossed) {
        const c = g.cr, sc = scr(c.x, c.y - 8), tries0 = g.tries;
        catchPress();
        const hit = g.res === 'win' ? 1 : 0;
        ev.bursts.push([st.t, g.speed / g.zw, Math.abs(st.err), sc[0], sc[1], id(c), hit]);
        if (hit) ev.kills.push([st.t, sc[0], sc[1]]);
        st.err = null; st.prevPos = null;
      }
      return;
    }
    st.err = null; st.prevPos = null;
    // pick a target: nearest creature, else nearest bush with berries, else wander
    let tgt = null, bd = 1e9;
    for (const c of wild.crs) { if (c.gone) continue; const d = Math.hypot(c.x - player.x, c.y - player.y); if (d < bd) { bd = d; tgt = c; } }
    if (!tgt || bd > 260) for (const b of wild.bushes) { if (!b.has) continue; const bx = b.x != null ? b.x : b.i * 16 + 8, by = b.y != null ? b.y : b.j * 16 + 8; const d = Math.hypot(bx - player.x, by - player.y); if (d < bd) { bd = d; tgt = { x: bx, y: by }; } }
    const near = findTarget();
    if (near && (near.type === 'cr' || near.type === 'bush' || near.type === 'spark') && rnd() < 0.5) { setKeys(0, 0); pressA(); return; }
    let gx, gy;
    if (tgt) { gx = tgt.x; gy = tgt.y; }
    else { if ((st.wanderT -= dt) <= 0 || !st.wander) { st.wander = [player.x + (rnd() - 0.5) * 300, player.y + (rnd() - 0.5) * 300]; st.wanderT = 3; } [gx, gy] = st.wander; }
    // walk, with a short sideways detour when blocked
    if (st.detourT > 0) { st.detourT -= dt; setKeys(st.detour[0], st.detour[1]); }
    else setKeys(gx - player.x, gy - player.y);
    const moved = Math.hypot(player.x - st.lastP[0], player.y - st.lastP[1]);
    if (moved < 0.3) st.stuckT += dt; else st.stuckT = 0;
    st.lastP = [player.x, player.y];
    if (st.stuckT > 0.4) { const a = rnd() * 6.28; st.detour = [Math.cos(a) * 10, Math.sin(a) * 10]; st.detourT = 0.4 + rnd() * 0.6; st.stuckT = 0; }
  }

  function record() {
    const vis = [];
    if (scene === 'wild' && wild) for (const c of wild.crs) {
      if (c.gone || !onScreen(c.x, c.y - 8)) continue;
      const sc = scr(c.x, c.y - 8);
      vis.push([+sc[0].toFixed(2), +sc[1].toFixed(2), +Math.hypot(c.x - player.x, c.y - player.y).toFixed(0), 1, id(c), 1]);
    }
    return { t: +st.t.toFixed(2), zone: wild ? wild.zone : scene, x: +player.x.toFixed(1), y: +player.y.toFixed(1), camX: +camX.toFixed(2), camY: +camY.toFixed(2), catching: catchG ? 1 : 0, items: vis.length, vis };
  }

  return {
    setup(seed) { s = seed >>> 0; try { muted = true; } catch (e) {} control(0); },
    canvas: () => cvs,
    control, record,
    events: () => ev,
    resetEvents() { for (const k in ev) ev[k].length = 0; st.t = 0; },
  };
})();
