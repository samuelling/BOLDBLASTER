// The Halcyon's plan: three floors on a 2 m grid. The same grid is the collision, the bots'
// pathfinding and the analysis (cells, rooms, walls, path distance), so all three agree.
//
//   x runs east (+i), z runs north (+k). Floor L sits at y = L * LH.
//   Ground floor: lobby round the atrium, reception and front doors (south), ballroom (north) with the
//   pool terrace beyond, bar and library (west), kitchen and staff room (east).
//   Floors 1-2: a gallery corridor round the atrium void, guest rooms north and south, suites at
//   the corners. Two enclosed stairwells (west, east) and the grand stair in the atrium join floors.

export const C = 2;            // cell size (m)
export const NX = 28, NZ = 28; // grid; k >= 20 is the outdoor terrace (ground floor only)
export const NL = 3;           // floors
export const LH = 4.2;         // floor-to-floor height (m)
export const CEIL = 3.5;       // ceiling height above the floor (m)
export const X0 = -NX * C / 2, Z0 = -20 * C / 2;   // world position of cell (0, 0)'s corner

export const cx = (i) => X0 + (i + 0.5) * C;
export const cz = (k) => Z0 + (k + 0.5) * C;
export const ci = (x) => Math.floor((x - X0) / C);
export const ck = (z) => Math.floor((z - Z0) / C);

// ------------------------------------------------------------------------------------------ rooms
// Each room: id, name, level, rectangle [i0, k0, i1, k1] inclusive, kind (style), and its doors:
// [i, k, side] = the edge on that side of cell (i, k).
const R = [];
const room = (name, L, i0, k0, i1, k1, kind, extra = {}) => { R.push({ id: R.length, name, L, rect: [i0, k0, i1, k1], kind, ...extra }); return R.length - 1; };

// ground floor
room('Lobby', 0, 8, 4, 19, 15, 'lobby');
room('Reception', 0, 8, 0, 19, 3, 'reception', { windows: 'S' });
room('Ballroom', 0, 8, 16, 19, 19, 'ballroom', { windows: 'N' });
room('Bar', 0, 0, 0, 7, 4, 'bar', { windows: 'SW' });
room('Library', 0, 0, 15, 7, 19, 'library', { windows: 'NW' });
room('Kitchen', 0, 20, 0, 27, 4, 'kitchen', { windows: 'SE' });
room('Staff room', 0, 20, 15, 27, 19, 'staff', { windows: 'NE' });
room('Pool terrace', 0, 4, 20, 23, 27, 'terrace', { outdoor: true });
for (const [s, i0] of [['W', 0], ['E', 20]]) {
  room(`${s} stair hall`, 0, s === 'W' ? 6 : 20, 5, s === 'W' ? 7 : 21, 14, 'hall');
}
// upper floors
for (const L of [1, 2]) {
  room(`Gallery ${L}`, L, 8, 4, 19, 15, 'gallery');
  const f = L * 100;
  for (let n = 0; n < 4; n++) room(`Room ${f + 1 + n}`, L, 8 + 3 * n, 0, 10 + 3 * n, 3, n % 2 ? 'guestB' : 'guestA', { windows: 'S' });
  for (let n = 0; n < 4; n++) room(`Room ${f + 5 + n}`, L, 8 + 3 * n, 16, 10 + 3 * n, 19, n % 2 ? 'guestA' : 'guestB', { windows: 'N' });
  room(`Suite ${f + 10}`, L, 0, 0, 7, 4, 'suite', { windows: 'SW' });
  room(`Suite ${f + 11}`, L, 0, 15, 7, 19, 'suite', { windows: 'NW' });
  room(`Suite ${f + 12}`, L, 20, 0, 27, 4, 'suite', { windows: 'SE' });
  room(`Suite ${f + 13}`, L, 20, 15, 27, 19, 'suite', { windows: 'NE' });
  room(`W hall ${L}`, L, 6, 5, 7, 14, 'hall');
  room(`E hall ${L}`, L, 20, 5, 21, 14, 'hall');
}
// stairwells: one room per floor each (the shaft), floor cells marked by the ramps below
for (const L of [0, 1, 2]) { room(`W stairwell ${L}`, L, 2, 5, 5, 14, 'stair'); room(`E stairwell ${L}`, L, 22, 5, 25, 14, 'stair'); }
export const ROOMS = R;

// ------------------------------------------------------------------------------------------ ramps
// A ramp is a straight flight: lane cells [i0..i1] x [k0..k1], rising along axis ('+k', '-k',
// '+i', '-i') from floor L to L+1 over its length. Its foot and head connect to landing cells.
export const RAMPS = [
  { id: 0, name: 'W stairs 0-1', group: 'W', L: 0, lane: [2, 7, 3, 12], axis: '+k' },
  { id: 1, name: 'W stairs 1-2', group: 'W', L: 1, lane: [4, 7, 5, 12], axis: '-k' },
  { id: 2, name: 'E stairs 0-1', group: 'E', L: 0, lane: [24, 7, 25, 12], axis: '+k' },
  { id: 3, name: 'E stairs 1-2', group: 'E', L: 1, lane: [22, 7, 23, 12], axis: '-k' },
  { id: 4, name: 'Grand stair 0-1', group: 'G', L: 0, lane: [11, 12, 16, 13], axis: '+i' },
  { id: 5, name: 'Grand stair 1-2', group: 'G', L: 1, lane: [11, 6, 16, 7], axis: '-i' },
];
for (const r of RAMPS) {
  const [i0, k0, i1, k1] = r.lane;
  const along = r.axis[1] === 'k' ? [k0, k1] : [i0, i1];
  r.len = (along[1] - along[0] + 1) * C;
  // world coordinate where the ramp starts (foot) along its axis, and the direction
  r.dir = r.axis[0] === '+' ? 1 : -1;
  r.s0 = r.axis[1] === 'k' ? (r.dir > 0 ? Z0 + k0 * C : Z0 + (k1 + 1) * C) : (r.dir > 0 ? X0 + i0 * C : X0 + (i1 + 1) * C);
}
// height of a ramp at world (x, z): 0 at the foot .. LH at the head, plus its floor's base
export function rampY(r, x, z) {
  const s = r.axis[1] === 'k' ? z : x;
  const u = Math.min(1, Math.max(0, (s - r.s0) * r.dir / r.len));
  return r.L * LH + u * LH;
}
export function inLane(r, i, k) { const [i0, k0, i1, k1] = r.lane; return i >= i0 && i <= i1 && k >= k0 && k <= k1; }

// ------------------------------------------------------------------------------------------ grid
// cell kind per level: 0 solid/outside, 1 floor, 2 void (open to a floor below), 3 ramp
export const KIND = Array.from({ length: NL }, () => new Uint8Array(NX * NZ));
export const ROOM = Array.from({ length: NL }, () => new Int16Array(NX * NZ).fill(-1));
export const RAMP = Array.from({ length: NL }, () => new Int8Array(NX * NZ).fill(-1));
const idx = (i, k) => k * NX + i;
export const inGrid = (i, k) => i >= 0 && k >= 0 && i < NX && k < NZ;
for (const r of R) {
  const [i0, k0, i1, k1] = r.rect;
  for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) { KIND[r.L][idx(i, k)] = 1; ROOM[r.L][idx(i, k)] = r.id; }
}
// stairwells: only landings and lanes are floor; the rest of the shaft is solid or void
for (const L of [0, 1, 2]) for (const [a, b] of [[2, 5], [22, 25]]) for (let i = a; i <= b; i++) for (let k = 5; k <= 14; k++) {
  const landingS = k <= 6, landingN = k >= 13;
  let kind = 0;
  if (L === 0) kind = landingS || landingN ? 1 : 0;               // ground: both landings; lanes handled by ramps / passage
  if (L === 1) kind = landingN ? 1 : landingS ? 2 : 0;            // first floor: north landing, open shaft at the south
  if (L === 2) kind = landingS ? 1 : landingN ? 2 : 0;            // second floor: south landing
  KIND[L][idx(i, k)] = kind; if (kind !== 1) ROOM[L][idx(i, k)] = kind === 2 ? ROOM[L][idx(i, k)] : -1;
}
// ground floor: the passage beside the stairs (under the 1-2 flight)
for (const [a, rn] of [[4, 'W stairwell 0'], [22, 'E stairwell 0']]) for (let i = a; i <= a + 1; i++) for (let k = 7; k <= 12; k++) { KIND[0][idx(i, k)] = 1; ROOM[0][idx(i, k)] = R.findIndex((q) => q.name === rn); }
// atrium void on the upper floors (minus the grand stair and its landings)
for (const L of [1, 2]) for (let i = 10; i <= 17; i++) for (let k = 6; k <= 13; k++) { KIND[L][idx(i, k)] = 2; }
// grand stair landings: head of 0-1 at (17, 12..13) on floor 1, head of 1-2 at (10, 6..7) on floor 2
for (const k of [12, 13]) KIND[1][idx(17, k)] = 1;
for (const k of [6, 7]) KIND[1][idx(17, k)] = 1;            // foot of the 1-2 flight on the first floor
for (const k of [6, 7]) KIND[2][idx(10, k)] = 1;
// ramps occupy their lane on both the floor they start from and the one they reach
for (const r of RAMPS) {
  const [i0, k0, i1, k1] = r.lane;
  for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) for (const L of [r.L, r.L + 1]) {
    KIND[L][idx(i, k)] = 3; RAMP[L][idx(i, k)] = r.id;
    if (ROOM[L][idx(i, k)] < 0) ROOM[L][idx(i, k)] = R.findIndex((q) => q.L === L && q.rect[0] <= i && q.rect[2] >= i && q.rect[1] <= k && q.rect[3] >= k);
  }
}
export const kindAt = (L, i, k) => (inGrid(i, k) && L >= 0 && L < NL ? KIND[L][idx(i, k)] : 0);
export const roomAt = (L, i, k) => (inGrid(i, k) && L >= 0 && L < NL ? ROOM[L][idx(i, k)] : -1);
export const rampAt = (L, i, k) => (inGrid(i, k) && L >= 0 && L < NL ? RAMP[L][idx(i, k)] : -1);

// ------------------------------------------------------------------------------------------ edges
// Every edge between two cells on a floor is open, a wall, a railing (blocks walking, not sight),
// a doorway, a window (wall you can see through to the outside) or a security door (open/shut).
// Edges are keyed by level and the cell on their west/south side: 'L:i:k:E' or 'L:i:k:N'.
export const EDGE = new Map();
const ek = (L, i, k, s) => (s === 'E' || s === 'N' ? `${L}:${i}:${k}:${s}` : s === 'W' ? `${L}:${i - 1}:${k}:E` : `${L}:${i}:${k - 1}:N`);
export const edgeKey = ek;
const nb = (i, k, s) => (s === 'E' ? [i + 1, k] : s === 'W' ? [i - 1, k] : s === 'N' ? [i, k + 1] : [i, k - 1]);
function defaultEdge(L, i, k, s) {
  const [j, m] = nb(i, k, s);
  const a = kindAt(L, i, k), b = kindAt(L, j, m);
  const ra = roomAt(L, i, k), rb = roomAt(L, j, m);
  const rpA = rampAt(L, i, k), rpB = rampAt(L, j, m);
  if (a === 0 && b === 0) return null;
  if (a === 0 || b === 0) {   // building shell or a solid block
    const inside = a ? [i, k, ra] : [j, m, rb];
    const rm = R[inside[2]];
    if (rm && rm.outdoor) return 'rail';                    // the terrace balustrade over the cliff
    return 'wall';
  }
  if (a === 2 && b === 2) return null;
  if ((a === 2) !== (b === 2)) return a === 3 || b === 3 ? 'rail' : 'rail';   // edge of a void
  if (a === 3 && b === 3) return rpA === rpB ? null : 'wall';              // two lanes side by side
  if (a === 3 || b === 3) {                                                  // a ramp's side or end
    const r = RAMPS[a === 3 ? rpA : rpB], [i0, k0, i1, k1] = r.lane, cell = a === 3 ? [i, k] : [j, m], other = a === 3 ? [j, m] : [i, k];
    const alongK = r.axis[1] === 'k';
    const atEnd = alongK ? other[0] === cell[0] : other[1] === cell[1];
    if (!atEnd) return r.group === 'G' ? 'rail' : 'wall';
    // which end: the foot is open on the ramp's own floor, the head on the floor above
    const isFoot = alongK ? (r.dir > 0 ? other[1] < k0 : other[1] > k1) : (r.dir > 0 ? other[0] < i0 : other[0] > i1);
    return (isFoot && L === r.L) || (!isFoot && L === r.L + 1) ? null : 'wall';
  }
  if (ra === rb) return null;
  return 'wall';
}
for (let L = 0; L < NL; L++) for (let i = -1; i < NX; i++) for (let k = -1; k < NZ; k++) for (const s of ['E', 'N']) {
  const e = defaultEdge(L, i, k, s);
  if (e) EDGE.set(ek(L, i, k, s), e);
}
// doorways (always open) and security doors (the gang opens and shuts them)
export const DOORS = [];
const door = (L, i, k, s, secure = false, name = '', more = []) => {
  const keys = [[i, k], ...more].map(([a, b]) => ek(L, a, b, s));
  for (const key of keys) EDGE.set(key, secure ? 'sdoor' : 'door');
  if (secure) DOORS.push({ id: DOORS.length, L, i, k, s, key: keys[0], keys, open: true, name });
};
// ground floor
for (const i of [12, 15]) door(0, i, 3, 'N');                 // reception <-> lobby
for (const i of [12, 15]) door(0, i, 15, 'N');                // lobby <-> ballroom
for (const i of [10, 17]) door(0, i, 19, 'N');                // ballroom <-> terrace
door(0, 13, 0, 'S'); door(0, 14, 0, 'S');                     // front doors (to the street)
door(0, 7, 2, 'E'); door(0, 7, 17, 'E');                      // bar / library <-> lobby
door(0, 20, 2, 'W'); door(0, 20, 17, 'W');                    // kitchen / staff <-> lobby
door(0, 7, 9, 'E', true, 'W hall'); door(0, 20, 10, 'W', true, 'E hall');   // halls <-> lobby
door(0, 5, 6, 'E', true, 'W stairs S'); door(0, 5, 13, 'E', true, 'W stairs N');
door(0, 22, 6, 'W', true, 'E stairs S'); door(0, 22, 13, 'W', true, 'E stairs N');
door(0, 6, 5, 'S'); door(0, 21, 5, 'S');                      // halls <-> bar / kitchen
door(0, 6, 14, 'N'); door(0, 21, 14, 'N');                    // halls <-> library / staff
// upper floors
for (const L of [1, 2]) {
  for (let n = 0; n < 4; n++) { door(L, 9 + 3 * n, 3, 'N'); door(L, 9 + 3 * n, 16, 'S'); }
  door(L, 7, 9, 'E', true, `W hall ${L}`); door(L, 20, 10, 'W', true, `E hall ${L}`);
  door(L, 6, 5, 'S'); door(L, 6, 14, 'N'); door(L, 21, 5, 'S'); door(L, 21, 14, 'N');   // halls <-> suites
  if (L === 1) { door(1, 5, 13, 'E', true, 'W stairs 1'); door(1, 22, 13, 'W', true, 'E stairs 1'); }
  if (L === 2) { door(2, 5, 6, 'E', true, 'W stairs 2'); door(2, 22, 6, 'W', true, 'E stairs 2'); }
  // fire doors across the gallery at its four corners
  door(L, 9, 4, 'E', true, `Gallery ${L} SW`, [[9, 5]]); door(L, 17, 4, 'E', true, `Gallery ${L} SE`, [[17, 5]]);
  door(L, 9, 14, 'E', true, `Gallery ${L} NW`, [[9, 15]]); door(L, 17, 14, 'E', true, `Gallery ${L} NE`, [[17, 15]]);
}
// the suites' second doors onto the gallery (more loops)
for (const L of [1, 2]) { door(L, 7, 4, 'E'); door(L, 7, 15, 'E'); door(L, 20, 4, 'W'); door(L, 20, 15, 'W'); }
// windows on outer walls of rooms that have them, every other cell
for (const r of R) {
  if (!r.windows) continue;
  const [i0, k0, i1, k1] = r.rect;
  for (const s of r.windows) {
    const cells = s === 'S' || s === 'N' ? Array.from({ length: i1 - i0 + 1 }, (_, n) => [i0 + n, s === 'S' ? k0 : k1]) : Array.from({ length: k1 - k0 + 1 }, (_, n) => [s === 'W' ? i0 : i1, k0 + n]);
    cells.forEach(([i, k], n) => { const key = ek(r.L, i, k, s); if (EDGE.get(key) === 'wall' && n % 2 === 0 && kindAt(r.L, ...nb(i, k, s)) === 0) EDGE.set(key, 'window'); });
  }
}
// the ballroom's north wall is glass onto the terrace
for (let i = 8; i <= 19; i++) if (EDGE.get(ek(0, i, 19, 'N')) === 'wall') EDGE.set(ek(0, i, 19, 'N'), 'glass');
export const edgeAt = (L, i, k, s) => EDGE.get(ek(L, i, k, s)) || null;
export function passable(L, i, k, s) {
  const e = edgeAt(L, i, k, s);
  if (!e || e === 'door') return true;
  if (e === 'sdoor') return DOOR_OF.get(ek(L, i, k, s)).open;
  return false;
}
export const DOOR_OF = new Map(); for (const d of DOORS) for (const key of d.keys) DOOR_OF.set(key, d);
export function seeThrough(L, i, k, s) {
  const e = edgeAt(L, i, k, s);
  if (!e || e === 'door' || e === 'rail' || e === 'glass' || e === 'window') return true;
  if (e === 'sdoor') return DOOR_OF.get(ek(L, i, k, s)).open;
  return false;
}
export const neighbour = nb;

// ------------------------------------------------------------------------------------------ graph
// Nodes: walkable cells (floor or ramp) per floor. Ramp cells exist on two floors; they are joined
// so that a path can climb. Node id = L * NX * NZ + k * NX + i.
export const nodeId = (L, i, k) => L * NX * NZ + k * NX + i;
export const nodeL = (n) => Math.floor(n / (NX * NZ)), nodeI = (n) => n % NX, nodeK = (n) => Math.floor((n % (NX * NZ)) / NX);
export function walkable(L, i, k) { const t = kindAt(L, i, k); return t === 1 || t === 3; }
export function neighbours(n) {
  const L = nodeL(n), i = nodeI(n), k = nodeK(n), out = [];
  for (const s of ['E', 'W', 'N', 'S']) {
    const [j, m] = nb(i, k, s);
    if (!walkable(L, j, m) || !passable(L, i, k, s)) continue;
    // on a ramp, only move along the lane (or off its open ends)
    out.push(nodeId(L, j, m));
  }
  // a ramp cell: the same cell on the other floor of its flight
  const rp = rampAt(L, i, k);
  if (rp >= 0) { const r = RAMPS[rp]; const L2 = L === r.L ? r.L + 1 : r.L; out.push(nodeId(L2, i, k)); }
  return out;
}
// Ramp cells on the wrong floor are only reachable via their twin; the twin edge has cost 0.
export function edgeCost(a, b) { return nodeL(a) !== nodeL(b) && nodeI(a) === nodeI(b) && nodeK(a) === nodeK(b) ? 0.01 : 1; }
// cost of a node pair given the ramp: a cell at the lower half of a ramp belongs to the lower floor
export function nodeOk(n) {
  const L = nodeL(n), i = nodeI(n), k = nodeK(n), rp = rampAt(L, i, k);
  if (rp < 0) return walkable(L, i, k);
  const r = RAMPS[rp], y = rampY(r, cx(i), cz(k)) - r.L * LH;   // 0..LH along the flight
  return L === r.L ? y <= LH * 0.5 + 0.6 : y >= LH * 0.5 - 0.6;
}
export function bfs(src, opts = {}) {   // shortest cell-step distances (each step 2 m), honouring doors
  const d = new Map([[src, 0]]), q = [src];
  for (let h = 0; h < q.length; h++) {
    const a = q[h];
    for (const b of neighbours(a)) { if (d.has(b) || !nodeOk(b)) continue; d.set(b, d.get(a) + edgeCost(a, b)); q.push(b); }
  }
  return d;
}
export function path(src, dst, cost = null) {   // A*-free Dijkstra on small grid with optional extra cost per ramp
  const dist = new Map([[src, 0]]), prev = new Map(), heap = [[0, src]];
  while (heap.length) {
    let bi = 0; for (let h = 1; h < heap.length; h++) if (heap[h][0] < heap[bi][0]) bi = h;
    const [d0, a] = heap[bi]; heap[bi] = heap[heap.length - 1]; heap.pop();
    if (a === dst) break;
    if (d0 > dist.get(a)) continue;
    for (const b of neighbours(a)) {
      if (!nodeOk(b)) continue;
      const nd = d0 + edgeCost(a, b) + (cost ? cost(a, b) : 0);
      if (nd < (dist.get(b) ?? 1e9)) { dist.set(b, nd); prev.set(b, a); heap.push([nd, b]); }
    }
  }
  if (!dist.has(dst)) return null;
  const out = [dst]; while (out[out.length - 1] !== src) out.push(prev.get(out[out.length - 1]));
  return out.reverse();
}
export const ROOMS_BY_LEVEL = [0, 1, 2].map((L) => R.filter((r) => r.L === L));
