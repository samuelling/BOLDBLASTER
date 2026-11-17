// In-page BOLDBLASTER harness for Halcyon. window.BBX(seed) installs a bot that plays the hotel on a
// fixed clock and returns H with frame() (one 10 Hz frame = two 50 ms ticks, the second drawn; a
// 192x144 luma image plus a log record) and events() (everything the model fits need).
//
// The bot is a simulated player, built from the same models the analysis fits:
//   routes   shortest believed paths; security doors it has seen shut are believed shut (map updating)
//   stairs   a softmax choice between the flights, by distance and a Rescorla-Wagner estimate of each
//            flight's ambush risk (the restless bandit)
//   calls    drift diffusion on what each person in the corridor is holding and how well it can be
//            seen (light, distance), to shoot or hold
//   aim      a pursuit controller with motor noise; fires when on target
//   dark     torch on in a blackout, slower and noisier heading
window.BBX = (seed) => {
  const G = window.HX, g = G.game, P = G.P, T = G.THREE;
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  g.seed = seed; g.reset();
  const p = () => g.player;
  const NN = P.NL * P.NX * P.NZ;

  // ---------------------------------------------------------------- zones (rooms split by security doors)
  // cells joined by open floor (no wall, doorway or security door between them) form a zone; zones are
  // joined by doorways, security doors and flights of stairs. This is the plan the map-updating model uses.
  const zone = new Int32Array(NN).fill(-1);
  let nz = 0;
  for (let n = 0; n < NN; n++) {
    const L = P.nodeL(n), i = P.nodeI(n), k = P.nodeK(n);
    if (!P.walkable(L, i, k) || zone[n] >= 0) continue;
    const q = [n]; zone[n] = nz;
    for (let h = 0; h < q.length; h++) {
      const a = q[h], al = P.nodeL(a), ai = P.nodeI(a), ak = P.nodeK(a);
      for (const sd of ['E', 'W', 'N', 'S']) {
        const [j, m] = P.neighbour(ai, ak, sd);
        if (!P.walkable(al, j, m) || P.edgeAt(al, ai, ak, sd)) continue;
        const b = P.nodeId(al, j, m); if (zone[b] < 0) { zone[b] = nz; q.push(b); }
      }
    }
    nz++;
  }
  const zEdges = new Set(), zAdj = Array.from({ length: nz }, () => new Set()), zx = new Float64Array(nz), zz = new Float64Array(nz), zc = new Float64Array(nz), zL = new Int32Array(nz);
  const doorZones = new Map();
  for (let n = 0; n < NN; n++) {
    if (zone[n] < 0) continue;
    const L = P.nodeL(n), i = P.nodeI(n), k = P.nodeK(n), za = zone[n];
    zx[za] += P.cx(i); zz[za] += P.cz(k); zc[za]++; zL[za] = L;
    for (const sd of ['E', 'N']) {
      const [j, m] = P.neighbour(i, k, sd), e = P.edgeAt(L, i, k, sd);
      if (!P.walkable(L, j, m) || (e !== 'door' && e !== 'sdoor')) continue;
      const zb = zone[P.nodeId(L, j, m)]; if (zb === za) continue;
      zAdj[za].add(zb); zAdj[zb].add(za); zEdges.add(Math.min(za, zb) + ':' + Math.max(za, zb));
      if (e === 'sdoor') doorZones.set(P.DOOR_OF.get(P.edgeKey(L, i, k, sd)).id, [za, zb]);
    }
    const rp = P.rampAt(L, i, k);
    if (rp >= 0 && L === P.RAMPS[rp].L) { const zb = zone[P.nodeId(L + 1, i, k)]; if (zb >= 0 && zb !== za) { zAdj[za].add(zb); zAdj[zb].add(za); zEdges.add(Math.min(za, zb) + ':' + Math.max(za, zb)); } }
  }
  for (let z = 0; z < nz; z++) { zx[z] /= zc[z]; zz[z] /= zc[z]; }
  const plan = { cells: Array.from({ length: nz }, (_, z) => z), edges: [...zEdges], zones: Array.from({ length: nz }, (_, z) => [zL[z], +zx[z].toFixed(1), +zz[z].toFixed(1), zc[z]]),
    doors: P.DOORS.map((d) => [d.id, d.L, d.i, d.k, d.s, doorZones.get(d.id) || null]), ramps: P.RAMPS.map((r, n) => [n, r.L, r.group, r.lane]),
    grid: { x0: P.X0, z0: P.Z0, cell: P.C, nx: P.NX, nz: P.NZ, nl: P.NL } };

  // ---------------------------------------------------------------- beliefs and routes
  const bel = new Map();                  // door id -> believed open (seen state), absent = open
  const Q = P.RAMPS.map(() => 0.5);        // believed ambush risk per flight
  const RW = 0.35, LAMBDA = 9, TEMP = 2.2;
  const heap = { a: [], push(d, n) { const A = this.a; A.push([d, n]); let i = A.length - 1; while (i > 0) { const j = (i - 1) >> 1; if (A[j][0] <= A[i][0]) break; [A[i], A[j]] = [A[j], A[i]]; i = j; } },
    pop() { const A = this.a, top = A[0], last = A.pop(); if (A.length) { A[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < A.length && A[l][0] < A[m][0]) m = l; if (r < A.length && A[r][0] < A[m][0]) m = r; if (m === i) break; [A[i], A[m]] = [A[m], A[i]]; i = m; } } return top; } };
  // believed neighbours: [node, cost]; ramps allowed by the 'ramps' predicate
  function nbrs(n, ramps) {
    const L = P.nodeL(n), i = P.nodeI(n), k = P.nodeK(n), out = [];
    for (const sd of ['E', 'W', 'N', 'S']) {
      const [j, m] = P.neighbour(i, k, sd);
      if (!P.walkable(L, j, m)) continue;
      const e = P.edgeAt(L, i, k, sd);
      let c = 1;
      if (e === 'sdoor') { const d = P.DOOR_OF.get(P.edgeKey(L, i, k, sd)); if (bel.has(d.id) && !bel.get(d.id)) continue; }
      else if (e && e !== 'door') continue;
      const b = P.nodeId(L, j, m); if (!P.nodeOk(b)) continue;
      if (blockedCell(b)) c += 3;
      const av = avoid.get(b); if (av && st.t < av) c += 6;
      out.push([b, c]);
    }
    const rp = P.rampAt(L, i, k);
    if (rp >= 0 && (!ramps || ramps(rp))) { const r = P.RAMPS[rp], L2 = L === r.L ? r.L + 1 : r.L, b = P.nodeId(L2, i, k); if (P.nodeOk(b)) out.push([b, 0.01]); }
    return out;
  }
  const avoid = new Map();   // cells where we got stuck recently -> until when to avoid them
  // cells with furniture in them cost a little more (routes keep to the clear floor)
  const blocked = new Uint8Array(NN);
  for (const ob of G.HXobs) for (let i = P.ci(ob.x - ob.r); i <= P.ci(ob.x + ob.r); i++) for (let k = P.ck(ob.z - ob.r); k <= P.ck(ob.z + ob.r); k++) if (P.inGrid(i, k)) blocked[P.nodeId(ob.L, i, k)] = 1;
  function blockedCell(n) { return blocked[n] === 1; }
  // where to aim for in a cell: its centre, or the free point nearest it when furniture is in the way
  const wpt = new Map();
  function wp(n) {
    if (!blocked[n]) return [P.cx(P.nodeI(n)), P.cz(P.nodeK(n))];
    if (wpt.has(n)) return wpt.get(n);
    const L = P.nodeL(n), x0 = P.cx(P.nodeI(n)), z0 = P.cz(P.nodeK(n)); let best = [x0, z0], bd = 1e9;
    for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) {
      const x = x0 + a * 0.35, z = z0 + b * 0.35;
      if (G.HXobs.some((o) => o.L === L && Math.hypot(o.x - x, o.z - z) < o.r * 0.8 + 0.4)) continue;
      const d = a * a + b * b; if (d < bd) { bd = d; best = [x, z]; }
    }
    wpt.set(n, best); return best;
  }
  function dijkstra(src, ramps, dst = -1) {
    const dist = new Float64Array(NN).fill(Infinity), prev = new Int32Array(NN).fill(-1);
    dist[src] = 0; heap.a.length = 0; heap.push(0, src);
    while (heap.a.length) {
      const [d0, a] = heap.pop(); if (d0 > dist[a]) continue; if (a === dst) break;
      for (const [b, c] of nbrs(a, ramps)) { const nd = d0 + c; if (nd < dist[b]) { dist[b] = nd; prev[b] = a; heap.push(nd, b); } }
    }
    return { dist, prev };
  }
  const trace = (prev, src, dst) => { const out = [dst]; while (out[out.length - 1] !== src) { const q = prev[out[out.length - 1]]; if (q < 0) return null; out.push(q); } return out.reverse(); };
  const nodeAt = (L, x, z) => P.nodeId(L, P.ci(x), P.ck(z));

  // ---------------------------------------------------------------- state
  const st = { t: 0, path: null, pi: 0, goal: -1, goalKey: '', tgt: -1, ddm: new Map(), cool: 0, lookT: 0, glance: 0, stuckT: 0, lastPos: null, lives: 0, lastZone: -1,
    home: null, outb: [], trial: null, lastL: 0, hopRamp: -1, aimNoise: [0, 0], pursue: 0, useT: 0, replanT: 0, lastHurt: -9, dark: false, pickupT: -9 };
  const ev = { kills: [], bursts: [], cuts: [], deaths: [], calls: [], resp: [], stairs: [], outcomes: [], doors: [], stale: [], junctions: [], homing: [], objectives: [], pickups: [], blackouts: [], spawns: [], shots: [], plan };
  const N = {}; const take = (k, src, map) => { if (!N[k]) N[k] = 0; while (N[k] < src.length) { ev[k].push(map ? map(src[N[k]]) : src[N[k]]); N[k]++; } };
  let inp = {};
  // ---------------------------------------------------------------- goals
  function goalNodes() {
    const o = g.objective, pl = p();
    if (!o) return null;
    if (o.phase === 'find') { const gu = o.guest; return { key: 'g' + gu.id, nodes: [nodeAt(gu.L, gu.pos.x, gu.pos.z)] }; }
    // walk out: the front doors or the pool terrace, whichever is believed nearer
    const outs = [P.nodeId(0, 13, 0), P.nodeId(0, 14, 0), P.nodeId(0, 10, 22), P.nodeId(0, 17, 22)];
    if (st.trial && st.home && st.home.L === pl.L) return { key: 'home', nodes: [st.home.n] };
    return { key: 'out', nodes: outs };
  }
  // route to the goal; a change of floor goes through a flight chosen by softmax over believed cost + risk
  function replan(reason) {
    const pl = p(), gl = goalNodes(); st.replanT = st.t;
    if (!gl) { st.path = null; return; }
    const src = nodeAt(pl.L, pl.pos.x, pl.pos.z);
    const gL = P.nodeL(gl.nodes[0]);
    st.goalKey = gl.key;
    if (gL === pl.L) {
      const D = dijkstra(src, (rp) => false);
      let best = -1; for (const n of gl.nodes) if (D.dist[n] < (best < 0 ? Infinity : D.dist[best])) best = n;
      if (best < 0 || !isFinite(D.dist[best])) { const D2 = dijkstra(src, null); for (const n of gl.nodes) if (best < 0 || D2.dist[n] < D2.dist[best]) best = n; st.path = isFinite(D2.dist[best]) ? trace(D2.prev, src, best) : null; }
      else st.path = trace(D.prev, src, best);
      st.pi = 0; return;
    }
    // the flights from this floor towards the goal's floor
    const Lnext = gL > pl.L ? pl.L + 1 : pl.L - 1, lo = Math.min(pl.L, Lnext);
    const cands = P.RAMPS.map((r, n) => [r, n]).filter(([r]) => r.L === lo);
    const opts = [];
    for (const [r, n] of cands) {
      const D = dijkstra(src, (rp) => rp === n || P.RAMPS[rp].L !== lo);
      let best = -1; for (const q of gl.nodes) if (best < 0 || D.dist[q] < D.dist[best]) best = q;
      if (best >= 0 && isFinite(D.dist[best])) opts.push({ n, cost: D.dist[best], path: trace(D.prev, src, best) });
    }
    if (!opts.length) { st.path = null; return; }
    // stick with the flight already chosen for this hop while it is still a way there
    const keep = st.hop && st.hop.L === pl.L && st.hop.gL === gL && opts.find((o) => o.n === st.hop.n);
    if (keep) { st.path = keep.path; st.pi = 0; return; }
    const c0 = Math.min(...opts.map((o) => o.cost));
    const w = opts.map((o) => Math.exp(-((o.cost - c0) + LAMBDA * Q[o.n]) / TEMP)), W = w.reduce((a, b) => a + b, 0);
    let u = rnd() * W, pick = opts[0]; for (let m = 0; m < opts.length; m++) { u -= w[m]; if (u <= 0) { pick = opts[m]; break; } }
    st.path = pick.path; st.pi = 0; st.hopRamp = pick.n; st.hop = { L: pl.L, gL, n: pick.n };
    ev.stairs.push([+st.t.toFixed(2), pick.n, opts.map((o) => o.n), opts.map((o) => +o.cost.toFixed(1)), pl.L, gL, Q.map((v) => +v.toFixed(3)), reason]);
  }

  // ---------------------------------------------------------------- perception: doors
  function seeDoors() {
    const pl = p();
    for (const d of P.DOORS) {
      if (d.L !== pl.L) continue;
      const mx = P.cx(d.i) + (d.s === 'E' ? P.C / 2 : d.s === 'W' ? -P.C / 2 : 0), mz = P.cz(d.k) + (d.s === 'N' ? P.C / 2 : d.s === 'S' ? -P.C / 2 : 0);
      const dd = Math.hypot(mx - pl.pos.x, mz - pl.pos.z); if (dd > 14) continue;
      // in front of us (within 70 deg) and in sight
      const f = Math.atan2(-(mx - pl.pos.x), -(mz - pl.pos.z)); if (Math.abs(wrap(f - pl.yaw)) > 1.2 && dd > 2.5) continue;
      const ex = mx - (mx - pl.pos.x) / dd * 0.3, ez = mz - (mz - pl.pos.z) / dd * 0.3;
      if (!G.HXlos(pl.L, pl.pos.x, pl.pos.z, ex, ez)) continue;
      const was = bel.has(d.id) ? bel.get(d.id) : true;
      if (was !== d.open) {
        bel.set(d.id, d.open);
        if (!d.open) {
          // a door we thought open is shut: is it on our way?
          const onPath = st.path && st.path.slice(st.pi).some((n, m, A) => m > 0 && crosses(A[m - 1], n, d));
          ev.stale.push([+st.t.toFixed(2), ...(doorZones.get(d.id) || [-1, -1]), d.id, onPath ? 1 : 0, +dd.toFixed(1)]);
          if (onPath) replan('door');
        }
      }
    }
  }
  function crosses(a, b, d) {
    if (P.nodeL(a) !== d.L || P.nodeL(b) !== d.L) return false;
    return d.keys.some((key) => { const ai = P.nodeI(a), ak = P.nodeK(a), bi = P.nodeI(b), bk = P.nodeK(b); const s = bi > ai ? 'E' : bi < ai ? 'W' : bk > ak ? 'N' : 'S'; return P.edgeKey(d.L, ai, ak, s) === key; });
  }

  // ---------------------------------------------------------------- the step
  function bot(dt) {
    const pl = p(), o = g.objective;
    inp = { moveX: 0, moveZ: 0, lookX: 0, lookY: 0, fire: false, run: false, use: false, reload: false, torchToggle: false };
    // lives
    if (pl.lives !== st.lives) { st.lives = pl.lives; ev.deaths.push([+st.t.toFixed(2)]); ev.cuts.push([+st.t.toFixed(2), 0]); st.path = null; st.tgt = -1; st.ddm.clear(); endTrial(0); st.home = null; st.outb = []; }
    // floor change: a new home (the head or foot of the flight we came by)
    if (pl.L !== st.lastL) {
      st.lastL = pl.L; st.home = { L: pl.L, x: pl.pos.x, z: pl.pos.z, n: nodeAt(pl.L, pl.pos.x, pl.pos.z), id: st.hopRamp }; st.outb = [[st.t, pl.pos.x, pl.pos.z]];
      if (st.path) replan('floor');
    }
    if (st.home && st.outb.length && st.t - st.outb[st.outb.length - 1][0] > 0.4) { st.outb.push([+st.t.toFixed(2), +pl.pos.x.toFixed(2), +pl.pos.z.toFixed(2)]); if (st.outb.length > 600) st.outb.shift(); }
    // the torch
    const dark = g.power < 0.5;
    if (dark !== pl.torch) inp.torchToggle = true;
    if (dark && !st.dark) startTrial();
    if (!dark && st.dark) endTrial(2);
    st.dark = dark;
    if (st.trial && st.home && Math.hypot(pl.pos.x - st.home.x, pl.pos.z - st.home.z) < 2.0 && pl.L === st.home.L) endTrial(1);
    seeDoors();
    // ------------------------------------------------ calls: drift diffusion per person in view
    const vis = g.visible();
    if (pl.hp < (st.hp0 ?? 100)) st.lastHurt = st.t; st.hp0 = pl.hp;
    for (const v of vis) {
      const [ax, ay, d, clear, id, , light, pose, obj, , , firing] = v;
      if (!clear || d > 24 || Math.abs(ax) > 34) continue;
      let D = st.ddm.get(id);
      if (!D) { D = { x: 0, t0: st.t, ndt: 0.2 + rnd() * 0.15, done: 0 }; st.ddm.set(id, D); }
      if (D.done || st.t - D.t0 < D.ndt) continue;
      const sig = obj ? -1 : pose === 'aim' ? 1 : pose === 'low' ? 0.75 : pose === 'phone' ? -0.4 : -1;
      const q = Math.min(1, light / 0.55) * Math.max(0.15, 1 - d / 26) * (Math.abs(ax) < 12 ? 1 : 0.6);
      D.x += 4.0 * sig * q * dt + 0.75 * Math.sqrt(dt) * gauss() + (firing ? 4 * dt : 0) + (st.t - st.lastHurt < 0.5 && !obj && sig > 0 ? 2 * dt : 0);
      if (D.x >= 1) { D.done = 1; if (st.tgt < 0) { st.tgt = id; st.aimNoise = [gauss() * 1.2, gauss() * 1.0]; st.pursue = 0; } }
      else if (D.x <= -1) D.done = -1;
    }
    // ------------------------------------------------ aim and fire at the engaged target
    let tv = st.tgt >= 0 ? vis.find((v) => v[4] === st.tgt) : null;
    if (st.tgt >= 0 && (!tv || !tv[3])) { const a = g.actors.find((q) => q.id === st.tgt); if (!a || !a.alive || (st.t - (st.lostT ?? st.t)) > 1.5) { st.tgt = -1; st.lostT = null; } else st.lostT = st.lostT ?? st.t; tv = null; }
    else st.lostT = null;
    let fighting = false;
    if (tv) {
      fighting = true; st.pursue += dt;
      // pursuit with a decaying aim offset (motor noise) plus jitter
      const k = Math.exp(-st.pursue * 2.2);
      const ex = tv[0] + st.aimNoise[0] * k + gauss() * 0.25, ey = tv[1] - 0.2 + st.aimNoise[1] * k + gauss() * 0.2;
      const gain = Math.min(1, dt * 9);
      inp.lookX = ex * Math.PI / 180 * gain; inp.lookY = -ey * Math.PI / 180 * gain;
      st.cool -= dt;
      const err = Math.hypot(ex, ey);
      // hold fire if someone we've judged a bystander is nearer, right by the line of fire
      const fouled = vis.some((v) => v[4] !== tv[4] && v[3] && v[2] < tv[2] - 0.3 && Math.hypot(v[0] - tv[0], v[1] - tv[1]) < 2.2 && (v[8] || (st.ddm.get(v[4]) || {}).done === -1));
      if (err < 1.6 && st.cool <= 0 && pl.ammo > 0 && pl.reloadT < 0 && !fouled) {
        inp.fire = true; st.cool = 0.22 + rnd() * 0.12;
        st.tgtScr = [+tv[9].toFixed(2), +tv[10].toFixed(2)]; st.tgtDist = +tv[2].toFixed(1);
      }
      if (pl.ammo === 0) inp.reload = true;
      if (inp.fire) st.lastFire = st.t; else if (st.pursue > 4 && st.t - (st.lastFire ?? 0) > 4) { st.ddm.get(tv[4]).done = -2; st.tgt = -1; }
      // stand and fight: small strafes
      inp.moveX = Math.sin(st.t * 1.7 + seed) * 0.5;
    }
    if (!fighting && pl.ammo < 7 && pl.reserve > 0 && pl.reloadT < 0) inp.reload = true;
    // ------------------------------------------------ the guest
    if (o && o.phase === 'find' && o.guest.alive && o.guest.L === pl.L && Math.hypot(o.guest.pos.x - pl.pos.x, o.guest.pos.z - pl.pos.z) < 2.4 && G.HXlos(pl.L, pl.pos.x, pl.pos.z, o.guest.pos.x, o.guest.pos.z)) { inp.use = true; st.pickupT = st.t; }
    // ------------------------------------------------ routes
    const gl = goalNodes();
    if (!gl) st.path = null;
    else if (gl.key !== st.goalKey || !st.path || (st.t - st.replanT > 6)) replan(gl.key !== st.goalKey ? 'goal' : 'refresh');
    if (!fighting && st.path) {
      if (!st.pts || st.ptsFor !== st.path) buildPts();
      // drop points already reached (doorway points need to be hit closely; floor twins pass on the flight)
      while (st.qi < st.pts.length - 1) {
        const q = st.pts[st.qi];
        let hit = q.L !== pl.L ? q.ramp : Math.hypot(q.x - pl.pos.x, q.z - pl.pos.z) < q.r;
        if (q.pre && q.L === pl.L) {   // lined up with the doorway's gap and close to it is enough
          const rx = pl.pos.x - q.mx, rz = pl.pos.z - q.mz, nrm = rx * q.ux + rz * q.uz, lat = Math.abs(rx * q.uz - rz * q.ux);
          if (lat < 0.3 && nrm > -0.9 && nrm < 0.1) hit = true;
        }
        if (hit) st.qi++; else break;
      }
      // look ahead to the furthest point in a clear straight line (never past a doorway)
      let ti = st.qi;
      if (!st.pts[st.qi].door) for (let m = st.qi + 1; m < Math.min(st.pts.length, st.qi + 6); m++) {
        const q = st.pts[m]; if (q.L !== pl.L) break;
        if (G.HXlos(pl.L, pl.pos.x, pl.pos.z, q.x, q.z, false) && clearLineXZ(pl, q.x, q.z)) ti = m; else break;
        if (q.door) break;
      }
      const tq = st.pts[ti], dx = tq.x - pl.pos.x, dz = tq.z - pl.pos.z, dist = Math.hypot(dx, dz);
      // heading: towards the point, with drift (more in the dark) and glances round the room
      st.glanceT = (st.glanceT ?? 2) - dt;
      if (st.glanceT <= 0) { st.glance = gauss() * (dark ? 0.25 : 0.55); st.glanceT = 1.2 + rnd() * 2.5; }
      st.glance *= Math.exp(-dt * 0.8);
      const head = Math.atan2(-dx, -dz), want = head + (tq.door ? 0 : st.glance) + (dark ? gauss() * 0.04 : 0);
      const err = wrap(want - pl.yaw);
      inp.lookX = -Math.max(-1, Math.min(1, err * 4 * dt));
      inp.lookY = -(-0.03 - pl.pitch) * Math.min(1, dt * 3);
      // move in the world direction of the point whatever the gaze (strafe as needed)
      const rel = wrap(head - pl.yaw), sp = (dark ? 0.65 : 1) * Math.min(1, dist / 0.6 + 0.3);
      inp.moveZ = Math.cos(rel) * sp; inp.moveX = -Math.sin(rel) * sp;
      if (st.qi >= st.pts.length - 1 && dist < 0.4) { st.path = null; st.pts = null; }
    } else if (!fighting && !st.path) {
      // nothing to do: look around
      inp.lookX = Math.sin(st.t * 0.6 + seed) * 0.02;
    }
    // stuck: no progress for 2.5 s while trying to move -> nudge and replan
    if (st.t < (st.backT ?? -1)) { inp.moveZ = -0.7; inp.moveX = st.backX; }
    else if (!fighting && inp.moveZ > 0.3) {
      if (!st.lastPos || Math.hypot(pl.pos.x - st.lastPos[0], pl.pos.z - st.lastPos[1]) > 0.6) { st.lastPos = [pl.pos.x, pl.pos.z]; st.stuckT = st.t; }
      else if (st.t - st.stuckT > 2.0) {
        // back off, avoid where we jammed for a while, and find another way
        st.stuckT = st.t; st.backT = st.t + 0.7; st.backX = rnd() < 0.5 ? -0.8 : 0.8;
        const here = nodeAt(pl.L, pl.pos.x, pl.pos.z); avoid.set(here, st.t + 20);
        if (st.path && st.pi < st.path.length) avoid.set(st.path[st.pi], st.t + 20);
        st.nStuck = (st.nStuck ?? 0) + 1; replan('stuck');
      }
    }
    // junctions on the zone graph
    const zc = zone[nodeAt(pl.L, pl.pos.x, pl.pos.z)];
    if (zc >= 0 && zc !== st.lastZone) {
      if (zAdj[zc].size >= 3 && gl && st.path) { const gz = zone[gl.nodes[0]]; ev.junctions.push([+st.t.toFixed(2), zc, st.lastZone, [...zAdj[zc]], pl.L, gl.key, 0, gz]); }
      st.lastZone = zc;
    }
  }
  function doorBetween(a, b) {   // a doorway or security door between two cells of the path (walk through its middle)
    const L = P.nodeL(a); if (L !== P.nodeL(b)) return false;
    const ai = P.nodeI(a), ak = P.nodeK(a), bi = P.nodeI(b), bk = P.nodeK(b);
    const sd = bi > ai ? 'E' : bi < ai ? 'W' : bk > ak ? 'N' : 'S';
    return !!P.edgeAt(L, ai, ak, sd);
  }
  // the path as points to walk through: free spots in each cell, and either side of every doorway
  function buildPts() {
    const pts = [], A = st.path.slice(st.pi);
    for (let m = 0; m < A.length; m++) {
      const n = A[m], L = P.nodeL(n), w = wp(n);
      if (m > 0 && P.nodeL(A[m - 1]) !== L) { pts.push({ x: w[0], z: w[1], L, r: 0.7, ramp: true }); continue; }
      if (m > 0 && doorBetween(A[m - 1], n)) {
        const a = A[m - 1], ax = P.cx(P.nodeI(a)), az = P.cz(P.nodeK(a)), bx = P.cx(P.nodeI(n)), bz = P.cz(P.nodeK(n));
        const mx = (ax + bx) / 2, mz = (az + bz) / 2, ux = Math.sign(bx - ax), uz = Math.sign(bz - az);
        pts.push({ x: mx - ux * 0.55, z: mz - uz * 0.55, L, r: 0.22, door: true, pre: true, mx, mz, ux, uz }, { x: mx + ux * 0.6, z: mz + uz * 0.6, L, r: 0.35, door: true });
      }
      pts.push({ x: w[0], z: w[1], L, r: 0.7 });
    }
    st.pts = pts; st.qi = 0; st.ptsFor = st.path;
  }
  function clearLineXZ(pl, x1, z1) {
    for (const ob of G.HXobs) { if (ob.L !== pl.L) continue; const dx = x1 - pl.pos.x, dz = z1 - pl.pos.z, l2 = dx * dx + dz * dz || 1e-9; const t = Math.max(0, Math.min(1, ((ob.x - pl.pos.x) * dx + (ob.z - pl.pos.z) * dz) / l2)); if (Math.hypot(pl.pos.x + dx * t - ob.x, pl.pos.z + dz * t - ob.z) < ob.r * 0.8 + 0.35) return false; }
    return true;
  }
  function clearLine(pl, n) {   // no obstacle circles on the straight line
    const [x1, z1] = wp(n);
    for (const ob of G.HXobs) { if (ob.L !== pl.L) continue; const dx = x1 - pl.pos.x, dz = z1 - pl.pos.z, l2 = dx * dx + dz * dz; const t = Math.max(0, Math.min(1, ((ob.x - pl.pos.x) * dx + (ob.z - pl.pos.z) * dz) / l2)); if (Math.hypot(pl.pos.x + dx * t - ob.x, pl.pos.z + dz * t - ob.z) < ob.r + 0.35) return false; }
    return true;
  }
  // ---------------------------------------------------------------- homing trials (the walk back in the dark)
  function startTrial() {
    const pl = p();
    if (!st.home || st.home.L !== pl.L || !g.objective || g.objective.phase !== 'escort') { st.trial = null; return; }
    st.trial = [+st.t.toFixed(2), +pl.pos.x.toFixed(2), +pl.pos.z.toFixed(2), st.home.id, +st.home.x.toFixed(2), +st.home.z.toFixed(2), st.outb.map((q) => q.slice())];
    replan('dark');
  }
  function endTrial(code) {
    if (!st.trial) return;
    const pl = p();
    ev.homing.push(st.trial.concat([+st.t.toFixed(2), +pl.pos.x.toFixed(2), +pl.pos.z.toFixed(2), code, pl.L]));
    st.trial = null; replan('lights');
  }

  // ---------------------------------------------------------------- capture
  const glc = G.renderer.getContext(), buf = new Uint8Array(384 * 288 * 4);
  function grab() {
    glc.readPixels(0, 0, 384, 288, glc.RGBA, glc.UNSIGNED_BYTE, buf);
    const out = new Uint16Array(192 * 144);
    for (let y = 0; y < 144; y++) for (let x = 0; x < 192; x++) {
      let acc = 0; for (let yy = 0; yy < 2; yy++) for (let xx = 0; xx < 2; xx++) { const k = ((287 - (2 * y + yy)) * 384 + 2 * x + xx) * 4; acc += 0.2126 * buf[k] + 0.7152 * buf[k + 1] + 0.0722 * buf[k + 2]; }
      out[y * 192 + x] = Math.round(acc / 4 / 255 * 65535);
    }
    const u8 = new Uint8Array(out.buffer); let bin = ''; for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
    return btoa(bin);
  }
  const DT = 0.05;
  const H = {
    frame() {
      for (let k = 0; k < 2; k++) {
        bot(DT);
        const killsBefore = g.log.kills.length, shotsBefore = g.log.shots.length;
        G.tick(inp, DT);
        st.t += DT;
        // shots fired this tick -> bursts [t, dist, err, sx, sy, target, hit]
        for (let m = shotsBefore; m < g.log.shots.length; m++) {
          const sh = g.log.shots[m];
          ev.bursts.push([+st.t.toFixed(2), sh[2] > 0 ? sh[2] : (st.tgtDist ?? 10), sh[5] >= 0 ? sh[5] : 3, ...(st.tgtScr || [0, 0]), sh[4], sh[1] >= 0 && sh[1] === sh[4] ? 1 : 0]);
        }
        for (let m = killsBefore; m < g.log.kills.length; m++) { const kv = g.log.kills[m]; ev.kills.push([+st.t.toFixed(2), ...(st.tgtScr || [0, 0]), kv[1], kv[2]]); }
      }
      // game-side logs, timed on the bot's clock (the game's clock runs in step)
      const off = st.t - g.t;
      const tt = (q) => [+(q[0] + off).toFixed(2), ...q.slice(1)];
      take('calls', g.log.calls, tt); take('resp', g.log.resp, tt); take('outcomes', g.log.outcomes, tt); take('doors', g.log.doors, tt);
      take('objectives', g.log.objectives, tt); take('pickups', g.log.pickups, tt); take('blackouts', g.log.blackouts, tt); take('spawns', g.log.spawns, tt); take('shots', g.log.shots, tt);
      // learning from the stairs: ambush (1) or supplies (0) at the far end
      while ((st.nOut ?? 0) < g.log.outcomes.length) { const oc = g.log.outcomes[st.nOut ?? 0]; Q[oc[1]] += RW * (oc[2] - Q[oc[1]]); st.nOut = (st.nOut ?? 0) + 1; }
      if (!window.BBX_NODRAW) G.draw();
      const pl = p(), vis = g.visible();
      const tv = st.tgt >= 0 ? vis.find((v) => v[4] === st.tgt) : null;
      if (tv) { st.tgtScr = [+tv[9].toFixed(2), +tv[10].toFixed(2)]; st.tgtDist = +tv[2].toFixed(1); }
      const n = nodeAt(pl.L, pl.pos.x, pl.pos.z), rm = P.roomAt(pl.L, P.ci(pl.pos.x), P.ck(pl.pos.z));
      const ld = G.HXlight(pl.L, pl.pos.x, pl.pos.y + 1.6, pl.pos.z);
      const rec = { t: +st.t.toFixed(2), tau: +st.t.toFixed(2), body: pl.lives, x: +pl.pos.x.toFixed(2), z: +pl.pos.z.toFixed(2), y: +pl.pos.y.toFixed(2), L: pl.L,
        yaw: +pl.yaw.toFixed(4), pitch: +pl.pitch.toFixed(4), speed: +Math.hypot(pl.vel.x, pl.vel.z).toFixed(3),
        vis: vis.map((v) => [+v[9].toFixed(2), +v[10].toFixed(2), +v[2].toFixed(1), v[3], v[4], v[5], v[6], v[7], v[8]]),
        cell: n, zone: zone[n] ?? -1, room: rm, wall: +G.HXwall(pl.L, pl.pos.x, pl.pos.z).toFixed(2), dark: g.power < 0.5 ? 1 : 0, light: +(ld[0] * G.U.uPower.value + ld[1]).toFixed(3),
        torch: pl.torch ? 1 : 0, phase: g.objective ? g.objective.phase : 'none', carry: g.objective && g.objective.phase === 'escort' ? 1 : 0, under: 0,
        hp: Math.round(pl.hp), ammo: pl.ammo, tgt: st.tgt, ramp: P.rampAt(pl.L, P.ci(pl.pos.x), P.ck(pl.pos.z)) };
      return { img: window.BBX_NODRAW ? '' : grab(), rec };
    },
    events() { return ev; },
    state: st, Q, bel, zone, plan,
  };
  return H;
};
