// The building: geometry from the plan, CC0 PBR materials, baked lighting, fixtures and props.
//
// Lighting is baked per vertex into two channels, lamp light and daylight, so the blackouts can
// switch the lamps off building-wide (uPower) while windows and the atrium's glass roof keep
// their daylight. On top of that the materials keep their normal maps (they shade the baked light
// and reflect the sky), and a torch and muzzle flashes are real lights.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import * as P from './plan.js';

const { C, NX, NZ, LH, CEIL, X0, Z0, cx, cz } = P;
const ASSET = 'assets/';

// ------------------------------------------------------------------------------------------ materials
const FLOOR = { lobby: ['marble_tiles', 3], reception: ['terrazzo_tiles', 2.5], ballroom: ['diagonal_parquet', 2.5], bar: ['herringbone_parquet', 2],
  library: ['herringbone_parquet', 2], kitchen: ['blue_floor_tiles_01', 1.6], staff: ['concrete_floor_painted', 3], terrace: ['wood_floor_deck', 2.5],
  hall: ['quatrefoil_jacquard_fabric', 1.2], gallery: ['quatrefoil_jacquard_fabric', 1.2], guestA: ['fabric_pattern_07', 1.5], guestB: ['herringbone_parquet', 2],
  suite: ['diagonal_parquet', 2.5], stair: ['concrete_floor_painted', 3] };
const WALL = { lobby: ['patterned_plaster_wall', 2.5], reception: ['patterned_plaster_wall', 2.5], ballroom: ['patterned_clay_plaster', 2.5], bar: ['decrepit_wallpaper', 2],
  library: ['decrepit_wallpaper', 2], kitchen: ['grey_tiles', 1.5], staff: ['painted_plaster_wall', 2.5], terrace: ['white_stucco', 3], hall: ['painted_plaster_wall', 2.5],
  gallery: ['patterned_plaster_wall', 2.5], guestA: ['decrepit_wallpaper', 2], guestB: ['blue_plaster_weathered', 2.5], suite: ['patterned_clay_plaster', 2.5],
  stair: ['painted_concrete', 3], outside: ['white_stucco', 3] };
const CEILM = ['white_stucco', 3];

// Lamp tint (warm), daylight tint (cool) and their strengths
export const LAMP_RGB = new THREE.Color(1.0, 0.78, 0.55), DAY_RGB = new THREE.Color(0.85, 0.93, 1.05);

// One shader patch for every surface: indirect diffuse from the baked lamp/day channels, modulated by
// the normal map (so grooves and tiles read), sky reflections scaled by how lit the spot is.
// Sources of the bake: per-vertex attribute 'bake' (static geometry), instance attribute 'ibake'
// (instanced props) or uniform uBakeObj (single moving objects).
export const U = { uPower: { value: 1 }, uDay: { value: 1 }, uLampRGB: { value: LAMP_RGB }, uDayRGB: { value: DAY_RGB }, uExposure: { value: 1 } };
export function patch(mat, mode = 'vertex', objU = null) {
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);
    if (mode === 'object') sh.uniforms.uBakeObj = objU;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        ${mode === 'vertex' ? 'attribute vec2 bake;' : ''}${mode === 'instance' ? 'attribute vec2 ibake;' : ''}${mode === 'object' ? 'uniform vec2 uBakeObj;' : ''}
        varying vec2 vBake; varying vec3 vGeoN;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vBake = ${mode === 'vertex' ? 'bake' : mode === 'instance' ? 'ibake' : 'uBakeObj'};
        vGeoN = normalize(normalMatrix * objectNormal);`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform float uPower, uDay, uExposure; uniform vec3 uLampRGB, uDayRGB; varying vec2 vBake; varying vec3 vGeoN;`)
      .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>
        { vec3 irr = vBake.x * uPower * uLampRGB + vBake.y * uDay * uDayRGB;
          float relief = clamp(0.55 + 0.45 * dot(normal, normalize(vGeoN)), 0.0, 1.0);
          irr *= relief * relief;
          reflectedLight.indirectDiffuse += irr * BRDF_Lambert(material.diffuseColor) * 3.14159;
          float lit = clamp(dot(irr, vec3(0.33)), 0.0, 1.5);
          reflectedLight.indirectSpecular *= lit * 0.6; }`);
  };
  mat.customProgramCacheKey = () => 'hx-' + mode;
  return mat;
}

const texCache = new Map();
function tex(loader, name, kind, srgb) {
  const key = name + kind;
  if (!texCache.has(key)) {
    const t = loader.load(`${ASSET}textures/${name}/${kind}.jpg`);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    texCache.set(key, t);
  }
  return texCache.get(key);
}
const matCache = new Map();
function surfMat(loader, name, opts = {}) {
  const key = name + JSON.stringify(opts);
  if (matCache.has(key)) return matCache.get(key);
  const m = patch(new THREE.MeshStandardMaterial({ map: tex(loader, name, 'diff', true), normalMap: tex(loader, name, 'nor', false), roughnessMap: tex(loader, name, 'rough', false),
    roughness: 1, metalness: 0, envMapIntensity: 0.6, side: opts.side || THREE.FrontSide, color: opts.color || 0xffffff }));
  matCache.set(key, m);
  return m;
}

// ------------------------------------------------------------------------------------------ light sources
// lamps: {L, x, y, z, I, room, fixture}; windows: {L, x, y, z, nx, nz, I}; skylight: the atrium roof
export const LAMPS = [], WINDOWS = [];
const ATRIUM = [10, 6, 17, 13];
function addLamps() {
  for (const r of P.ROOMS) {
    const [i0, k0, i1, k1] = r.rect, L = r.L, y0 = L * LH;
    const put = (i, k, I, fixture, y = y0 + CEIL - 0.05) => { if (P.walkable(L, Math.floor(i), Math.floor(k)) || fixture === 'chandelier') LAMPS.push({ L, x: X0 + (i + 0.5) * C, y, z: Z0 + (k + 0.5) * C, I, room: r.id, fixture }); };
    if (r.outdoor) continue;
    if (r.kind === 'lobby') {
      for (let i = 8; i <= 19; i += 2) { put(i + 0.5, 4.5, 1.6, 'ceiling'); put(i + 0.5, 14.5, 1.6, 'ceiling'); }
      for (let k = 6; k <= 13; k += 3) { put(8.5, k + 0.5, 1.6, 'ceiling'); put(18.5, k + 0.5, 1.6, 'ceiling'); }
      LAMPS.push({ L: 0, x: X0 + 14 * C, y: 2 * LH + 1.0, z: Z0 + 10 * C, I: 10, room: r.id, fixture: 'chandelier' });
    } else if (r.kind === 'gallery') {
      for (let i = 8; i <= 19; i += 3) { put(i + 0.5, 4.5, 1.4, 'ceiling'); put(i + 0.5, 14.5, 1.4, 'ceiling'); }
      for (let k = 7; k <= 12; k += 3) { put(8.5, k, 1.4, 'ceiling'); put(18.5, k, 1.4, 'ceiling'); }
    } else if (r.kind === 'ballroom') {
      for (const i of [10.5, 13.5, 16.5]) LAMPS.push({ L, x: X0 + (i + 0.5) * C, y: y0 + CEIL - 0.9, z: Z0 + 18 * C, I: 4.5, room: r.id, fixture: 'chandelier3' });
    } else if (r.kind === 'hall') {
      for (let k = k0 + 1; k <= k1; k += 3) put((i0 + i1) / 2, k, 1.1, 'ceiling');
    } else if (r.kind === 'stair') {
      // sconces on the landings of this floor
      for (let i = i0; i <= i1; i += 2) for (let k = k0; k <= k1; k += 1) if (P.kindAt(L, i, k) === 1 && (k === k0 || k === k1)) put(i + 0.5, k, 0.9, 'sconce', y0 + 2.3);
    } else {
      const w = i1 - i0 + 1, d = k1 - k0 + 1;
      const nx = Math.max(1, Math.round(w / 3)), nz = Math.max(1, Math.round(d / 3));
      const I = { bar: 1.2, library: 1.0, kitchen: 2.0, staff: 1.4, reception: 1.6, suite: 1.2, guestA: 1.1, guestB: 1.1 }[r.kind] || 1.2;
      for (let a = 0; a < nx; a++) for (let b = 0; b < nz; b++) put(i0 + (a + 0.5) * w / nx - 0.5, k0 + (b + 0.5) * d / nz - 0.5, I, 'ceiling');
    }
  }
  for (const [key, e] of P.EDGE) {
    if (e !== 'window' && e !== 'glass') continue;
    const [L, i, k, s] = key.split(':'); const Ln = +L, ii = +i, kk = +k;
    // window light enters from the outside cell into the inside cell
    const inA = P.walkable(Ln, ii, kk), [j, m] = P.neighbour(ii, kk, s);
    const inside = inA ? [ii, kk] : [j, m];
    const dir = inA ? (s === 'E' ? [-1, 0] : [0, -1]) : (s === 'E' ? [1, 0] : [0, 1]);
    const ex = s === 'E' ? X0 + (ii + 1) * C : X0 + (ii + 0.5) * C, ez = s === 'N' ? Z0 + (kk + 1) * C : Z0 + (kk + 0.5) * C;
    WINDOWS.push({ L: Ln, x: ex, y: Ln * LH + (e === 'glass' ? 1.6 : 1.75), z: ez, nx: -dir[0], nz: -dir[1], I: e === 'glass' ? 5 : 3.2, inside });
  }
}

// ------------------------------------------------------------------------------------------ visibility
// Line of sight on one floor through the grid's edges (walls block; doors, rails, glass do not).
export function los(L, x0, z0, x1, z1, sight = true) {
  let i = P.ci(x0), k = P.ck(z0); const i1 = P.ci(x1), k1 = P.ck(z1);
  const dx = x1 - x0, dz = z1 - z0;
  const stepI = dx > 0 ? 1 : -1, stepK = dz > 0 ? 1 : -1;
  const tDI = Math.abs(C / (dx || 1e-9)), tDK = Math.abs(C / (dz || 1e-9));
  let tI = dx > 0 ? (X0 + (i + 1) * C - x0) / dx : dx < 0 ? (X0 + i * C - x0) / dx : 1e9;
  let tK = dz > 0 ? (Z0 + (k + 1) * C - z0) / dz : dz < 0 ? (Z0 + k * C - z0) / dz : 1e9;
  for (let n = 0; n < 64 && (i !== i1 || k !== k1); n++) {
    let s;
    if (tI < tK) { s = stepI > 0 ? 'E' : 'W'; if (!(sight ? P.seeThrough(L, i, k, s) : P.passable(L, i, k, s))) return false; i += stepI; tI += tDI; }
    else { s = stepK > 0 ? 'N' : 'S'; if (!(sight ? P.seeThrough(L, i, k, s) : P.passable(L, i, k, s))) return false; k += stepK; tK += tDK; }
    if (tI > 1 && tK > 1) break;
    if (P.kindAt(L, i, k) === 0 && P.inGrid(i, k)) return false;
  }
  return true;
}

// baked light at a point with normal n on floor L: [lamp, day]
export function lightAt(L, x, y, z, nx, ny, nz) {
  let lamp = 0.025, day = 0;
  const i = P.ci(x), k = P.ck(z);
  for (const l of LAMPS) {
    const sameFloor = l.L === L || (l.fixture === 'chandelier' && (i >= ATRIUM[0] - 2 && i <= ATRIUM[2] + 2 && k >= ATRIUM[1] - 2 && k <= ATRIUM[3] + 2));
    if (!sameFloor) continue;
    const dx = l.x - x, dy = l.y - y, dz = l.z - z, d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > 220) continue;
    const d = Math.sqrt(d2), cos = (dx * nx + dy * ny + dz * nz) / d;
    if (cos <= 0) continue;
    if (l.fixture !== 'chandelier' && !los(L, x - nx * 0.05, z - nz * 0.05, l.x, l.z)) continue;
    lamp += l.I * 1.5 * cos / (1 + d2 * 0.75);
  }
  for (const w of WINDOWS) {
    if (w.L !== L) continue;
    const dx = w.x - x, dy = w.y - y, dz = w.z - z, d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > 300) continue;
    const d = Math.sqrt(d2), cosS = (dx * nx + dy * ny + dz * nz) / d, cosW = -(dx * w.nx + dz * w.nz) / d;
    if (cosS <= 0 || cosW <= 0) continue;
    if (!los(L, x - nx * 0.05, z - nz * 0.05, w.x - w.nx * 0.2, w.z - w.nz * 0.2)) continue;
    day += w.I * cosS * cosW / (1 + d2 * 0.25);
  }
  // the atrium's glass roof: light from above over the void and round its edge
  const inAtrium = i >= ATRIUM[0] && i <= ATRIUM[2] && k >= ATRIUM[1] && k <= ATRIUM[3];
  const nearAtrium = i >= ATRIUM[0] - 2 && i <= ATRIUM[2] + 2 && k >= ATRIUM[1] - 2 && k <= ATRIUM[3] + 2;
  if (inAtrium) day += 0.5 + 0.9 * Math.max(0, ny) + 0.25 * (y / (3 * LH));
  else if (nearAtrium && P.kindAt(L, i, k) !== 0) day += 0.22 * (L === 2 ? 1.5 : 1);
  // outdoors
  const rm = P.roomAt(L, i, k);
  if (rm >= 0 && P.ROOMS[rm].outdoor) day += 1.6 + 1.4 * Math.max(0, ny * 0.8 + nx * 0.3 + nz * 0.2);
  return [lamp, day];
}

// ------------------------------------------------------------------------------------------ geometry
class Batch {   // accumulate quads per material, with per-vertex bake
  constructor() { this.m = new Map(); }
  get(mat) { if (!this.m.has(mat)) this.m.set(mat, { p: [], n: [], u: [], b: [], t: [], ix: [] }); return this.m.get(mat); }
  // quad from origin o along a (u axis) and b (v axis), split nu x nv; uv scale in metres
  quad(mat, L, o, a, b, nrm, scale, nu = 1, nv = 1, uv0 = [0, 0]) {
    const g = this.get(mat), base = g.p.length / 3, la = Math.hypot(...a), lb = Math.hypot(...b);
    for (let v = 0; v <= nv; v++) for (let u = 0; u <= nu; u++) {
      const s = u / nu, t = v / nv;
      const x = o[0] + a[0] * s + b[0] * t, y = o[1] + a[1] * s + b[1] * t, z = o[2] + a[2] * s + b[2] * t;
      g.p.push(x, y, z); g.n.push(...nrm);
      g.u.push((uv0[0] + s * la) / scale, (uv0[1] + t * lb) / scale);
      const ld = lightAt(L, x + nrm[0] * 0.08, y + nrm[1] * 0.08, z + nrm[2] * 0.08, ...nrm);
      g.b.push(ld[0], ld[1]);
    }
    for (let v = 0; v < nv; v++) for (let u = 0; u < nu; u++) {
      const q = base + v * (nu + 1) + u;
      g.ix.push(q, q + 1, q + nu + 2, q, q + nu + 2, q + nu + 1);
    }
  }
  meshes() {
    const out = [];
    for (const [mat, g] of this.m) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(g.p, 3));
      geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.n, 3));
      geo.setAttribute('uv', new THREE.Float32BufferAttribute(g.u, 2));
      geo.setAttribute('bake', new THREE.Float32BufferAttribute(g.b, 2));
      geo.setIndex(g.ix); geo.computeTangents?.();
      const m = new THREE.Mesh(geo, mat); m.matrixAutoUpdate = false; out.push(m);
    }
    return out;
  }
}

function roomKind(L, i, k) { const r = P.roomAt(L, i, k); return r >= 0 ? P.ROOMS[r].kind : null; }

export async function buildWorld(scene, renderer, onProgress = () => {}) {
  const tl = new THREE.TextureLoader();
  addLamps();
  const B = new Batch();
  const fm = (kind) => { const f = FLOOR[kind] || FLOOR.hall; return [surfMat(tl, f[0], f[0].startsWith('quatrefoil') ? { color: 0xc9b2a6 } : {}), f[1]]; };
  const wm = (kind) => { const f = WALL[kind] || WALL.hall; return [surfMat(tl, f[0]), f[1]]; };
  const [ceilMat, ceilS] = [surfMat(tl, CEILM[0]), CEILM[1]];
  const trim = patch(new THREE.MeshStandardMaterial({ color: 0x3a2618, roughness: 0.55, metalness: 0.0 }));
  const brass = patch(new THREE.MeshStandardMaterial({ color: 0xb08d57, roughness: 0.3, metalness: 1.0, envMapIntensity: 1.2 }));
  const steel = patch(new THREE.MeshStandardMaterial({ color: 0x55595e, roughness: 0.45, metalness: 0.8 }));
  const glassM = new THREE.MeshPhysicalMaterial({ color: 0xbfd8e8, roughness: 0.05, metalness: 0, transmission: 0, transparent: true, opacity: 0.18, envMapIntensity: 1.0, side: THREE.DoubleSide, depthWrite: false });

  // floors, ceilings and ramp treads
  for (let L = 0; L < P.NL; L++) for (let i = 0; i < NX; i++) for (let k = 0; k < NZ; k++) {
    const t = P.kindAt(L, i, k), x0 = X0 + i * C, z0 = Z0 + k * C, y = L * LH;
    if (t === 1) {
      const [m, s] = fm(roomKind(L, i, k));
      B.quad(m, L, [x0, y, z0 + C], [C, 0, 0], [0, 0, -C], [0, 1, 0], s, 2, 2, [x0, -z0 - C]);
      const above = P.kindAt(L + 1, i, k), rm = P.roomAt(L, i, k), outdoor = rm >= 0 && P.ROOMS[rm].outdoor;
      if (!outdoor && above !== 2 && above !== 3 && !(L === 2 && P.kindAt(L, i, k) === 1 && false)) B.quad(ceilMat, L, [x0, y + CEIL, z0], [C, 0, 0], [0, 0, C], [0, -1, 0], ceilS, 1, 1, [x0, z0]);
    }
    if (t === 3) {
      const r = P.RAMPS[P.rampAt(L, i, k)];
      if (L !== r.L) continue;           // build each flight once, from its own floor
      // treads and risers
      const along = r.axis[1], dir = r.dir, nSteps = 7;   // per 2 m cell
      const [m, s] = fm(r.group === 'G' ? 'gallery' : 'stair');
      for (let q = 0; q < nSteps; q++) {
        const a0 = q / nSteps, a1 = (q + 1) / nSteps;
        let xa, xb, za, zb;
        if (along === 'k') { xa = x0; xb = x0 + C; za = dir > 0 ? z0 + a0 * C : z0 + C - a1 * C; zb = dir > 0 ? z0 + a1 * C : z0 + C - a0 * C; }
        else { za = z0; zb = z0 + C; xa = dir > 0 ? x0 + a0 * C : x0 + C - a1 * C; xb = dir > 0 ? x0 + a1 * C : x0 + C - a0 * C; }
        const yy = P.rampY(r, (xa + xb) / 2 + (along === 'i' ? dir * (C / nSteps) * 0.49 : 0), (za + zb) / 2 + (along === 'k' ? dir * (C / nSteps) * 0.49 : 0));
        const yLow = yy - LH / (r.len / C * nSteps);
        B.quad(m, L, [xa, yy, zb], [xb - xa, 0, 0], [0, 0, za - zb], [0, 1, 0], s, 1, 1, [xa, -zb]);
        // riser on the low side
        if (along === 'k') { const zr = dir > 0 ? za : zb; B.quad(m, L, [dir > 0 ? xa : xb, yLow, zr], [dir > 0 ? C : -C, 0, 0], [0, yy - yLow, 0], [0, 0, -dir], s, 1, 1); }
        else { const xr = dir > 0 ? xa : xb; B.quad(m, L, [xr, yLow, dir > 0 ? zb : za], [0, 0, dir > 0 ? -C : C], [0, yy - yLow, 0], [-dir, 0, 0], s, 1, 1); }
      }
      // soffit underneath (seen from below)
      const yA = P.rampY(r, along === 'i' ? x0 : cx(i), along === 'k' ? z0 : cz(k)) - 0.3, yB = P.rampY(r, along === 'i' ? x0 + C : cx(i), along === 'k' ? z0 + C : cz(k)) - 0.3;
      if (along === 'k') B.quad(ceilMat, L, [x0, yA, z0], [C, 0, 0], [0, yB - yA, C], [0, -1, 0], ceilS);
      else B.quad(ceilMat, L, [x0, yA, z0 + C], [0, 0, -C], [C, yB - yA, 0], [0, -1, 0], ceilS);
    }
  }
  // atrium glass roof and the slab edges round the voids
  for (let i = ATRIUM[0]; i <= ATRIUM[2]; i++) for (let k = ATRIUM[1]; k <= ATRIUM[3]; k++) {
    const x0 = X0 + i * C, z0 = Z0 + k * C, y = 2 * LH + CEIL + 0.6;
    B.quad(glassM, 2, [x0, y, z0], [C, 0, 0], [0, 0, C], [0, -1, 0], 4);
  }
  const H = { wall: [0, CEIL], door: [2.3, CEIL], sdoor: [2.3, CEIL], window: [[0, 0.95], [2.65, CEIL]], glass: null };
  // walls: each edge gets a face on each walkable side, in that side's material
  const faces = (L, x0, z0, x1, z1, side, kind, y0, y1, u0 = 0) => {
    const [m, s] = kind === 'outside' ? wm('outside') : wm(kind);
    const nx = z1 - z0 === 0 ? 0 : side, nz = x1 - x0 === 0 ? 0 : side;   // normal points into the room on that side
    const off = 0.07;
    const ox = nx * off, oz = nz * off;
    const len = Math.hypot(x1 - x0, z1 - z0), dx = (x1 - x0) / len, dz = (z1 - z0) / len;
    // orient so the face points along the normal: u along (dx, dz) when (dz, -dx) == n
    const flip = (dz * nx - dx * nz) > 0 ? 1 : 0;   // wind so that u x v points along n
    const ax = flip ? x1 : x0, az = flip ? z1 : z0, ux = flip ? -dx : dx, uz = flip ? -dz : dz;
    B.quad(m, L, [ax + ox, L * LH + y0, az + oz], [ux * len, 0, uz * len], [0, y1 - y0, 0], [nx, 0, nz], s, Math.max(1, Math.round(len)), Math.max(1, Math.round((y1 - y0) / 1.2)), [u0 + (flip ? -1 : 1) * (ax * Math.abs(dz) + az * Math.abs(dx)), y0]);
  };
  const outerKind = (L, i, k) => { const r = P.roomAt(L, i, k); return r >= 0 ? P.ROOMS[r].kind : 'outside'; };
  const props = [];   // door frames, rails, door leaves: built below as their own meshes
  for (const [key, e] of P.EDGE) {
    const [Ls, is, ks, s] = key.split(':'); const L = +Ls, i = +is, k = +ks;
    const [j, m] = P.neighbour(i, k, s);
    let x0, z0, x1, z1;
    if (s === 'E') { x0 = x1 = X0 + (i + 1) * C; z0 = Z0 + k * C; z1 = z0 + C; } else { z0 = z1 = Z0 + (k + 1) * C; x0 = X0 + i * C; x1 = x0 + C; }
    const sides = [];
    if (P.kindAt(L, i, k) === 1 || P.kindAt(L, i, k) === 3) sides.push([s === 'E' ? -1 : -1, outerKind(L, i, k)]);
    if (P.kindAt(L, j, m) === 1 || P.kindAt(L, j, m) === 3) sides.push([1, outerKind(L, j, m)]);
    // the outside of the shell (seen through windows and from the terrace)
    if (sides.length === 1 && e !== 'rail') sides.push([-sides[0][0], 'outside']);
    if (e === 'rail') {
      props.push({ type: 'rail', L, x0, z0, x1, z1 });
      continue;
    }
    const top = L * LH + CEIL;
    const extra = sides.some((q) => q[1] === 'outside') ? LH - CEIL + (L === 2 ? 0.7 : 0) : 0;   // the shell runs past the slabs, with a parapet on top
    for (const [sd, kind] of sides) {
      const y1 = CEIL + (kind === 'outside' ? extra : 0);
      if (e === 'wall') faces(L, x0, z0, x1, z1, sd, kind, 0, y1);
      else if (e === 'door' || e === 'sdoor') {
        // jambs either side of a 1.2 m opening, lintel above
        const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2, hx = (x1 - x0) / 2, hz = (z1 - z0) / 2, f = 0.6 / C * 2;
        faces(L, x0, z0, mx - hx * f, mz - hz * f, sd, kind, 0, y1);
        faces(L, mx + hx * f, mz + hz * f, x1, z1, sd, kind, 0, y1);
        faces(L, mx - hx * f, mz - hz * f, mx + hx * f, mz + hz * f, sd, kind, 2.3, y1);
      } else if (e === 'window') {
        const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2, hx = (x1 - x0) / 2, hz = (z1 - z0) / 2, f = 0.7 / C * 2;
        faces(L, x0, z0, mx - hx * f, mz - hz * f, sd, kind, 0, y1);
        faces(L, mx + hx * f, mz + hz * f, x1, z1, sd, kind, 0, y1);
        faces(L, mx - hx * f, mz - hz * f, mx + hx * f, mz + hz * f, sd, kind, 0, 0.95);
        faces(L, mx - hx * f, mz - hz * f, mx + hx * f, mz + hz * f, sd, kind, 2.65, y1);
      }
    }
    if (e === 'door' || e === 'sdoor' || e === 'window' || e === 'glass') props.push({ type: e, L, x0, z0, x1, z1, key });
  }
  // slab edges where a floor meets a void (seen from the atrium and the stairwells)
  for (let L = 1; L < P.NL; L++) for (const [key, e] of P.EDGE) {
    if (e !== 'rail' || +key.split(':')[0] !== L) continue;
    const [, is, ks, s] = key.split(':'); const i = +is, k = +ks; const [j, m] = P.neighbour(i, k, s);
    const voidSide = P.kindAt(L, i, k) === 2 ? -1 : P.kindAt(L, j, m) === 2 ? 1 : 0;
    if (!voidSide) continue;
    let x0, z0, x1, z1;
    if (s === 'E') { x0 = x1 = X0 + (i + 1) * C; z0 = Z0 + k * C; z1 = z0 + C; } else { z0 = z1 = Z0 + (k + 1) * C; x0 = X0 + i * C; x1 = x0 + C; }
    const [mm, sc] = wm('gallery');
    const nx = s === 'E' ? voidSide : 0, nz = s === 'N' ? voidSide : 0;
    const len = C, flip = s === 'E' ? (nx > 0 ? 0 : 1) : (nz > 0 ? 1 : 0);
    const ax = flip ? x1 : x0, az = flip ? z1 : z0, ux = (x1 - x0) / len * (flip ? -1 : 1), uz = (z1 - z0) / len * (flip ? -1 : 1);
    B.quad(mm, L, [ax + nx * 0.02, L * LH - (LH - CEIL), az + nz * 0.02], [ux * len, 0, uz * len], [0, LH - CEIL, 0], [nx, 0, nz], sc);
  }
  const root = new THREE.Group(); root.name = 'building';
  for (const m of B.meshes()) root.add(m);
  // ---------------------------------------------------------------- door frames, door leaves, rails, window glass
  const R = new Batch(), leaves = [];
  for (const q of props) {
    const { L, x0, z0, x1, z1 } = q, y = L * LH, alongX = z0 === z1;
    const len = Math.hypot(x1 - x0, z1 - z0), mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
    const box = (mat, cxx, cy, czz, sx, sy, sz) => {   // axis-aligned box as 6 quads
      const hx = sx / 2, hy = sy / 2, hz = sz / 2;
      R.quad(mat, L, [cxx - hx, cy + hy, czz + hz], [sx, 0, 0], [0, 0, -sz], [0, 1, 0], 1);
      R.quad(mat, L, [cxx - hx, cy - hy, czz - hz], [sx, 0, 0], [0, 0, sz], [0, -1, 0], 1);
      R.quad(mat, L, [cxx - hx, cy - hy, czz + hz], [sx, 0, 0], [0, sy, 0], [0, 0, 1], 1);
      R.quad(mat, L, [cxx + hx, cy - hy, czz - hz], [-sx, 0, 0], [0, sy, 0], [0, 0, -1], 1);
      R.quad(mat, L, [cxx + hx, cy - hy, czz + hz], [0, 0, -sz], [0, sy, 0], [1, 0, 0], 1);
      R.quad(mat, L, [cxx - hx, cy - hy, czz - hz], [0, 0, sz], [0, sy, 0], [-1, 0, 0], 1);
    };
    if (q.type === 'rail') {
      // balustrade: posts, brass handrail, a glass panel
      const n = 3;
      for (let a = 0; a <= n; a++) { const t = a / n; box(steel, x0 + (x1 - x0) * t, y + 0.5, z0 + (z1 - z0) * t, 0.05, 1.0, 0.05); }
      box(brass, mx, y + 1.02, mz, alongX ? len : 0.07, 0.06, alongX ? 0.07 : len);
      R.quad(glassM, L, [x0, y + 0.08, z0], [x1 - x0, 0, z1 - z0], [0, 0.9, 0], alongX ? [0, 0, 1] : [1, 0, 0], 4);
    } else if (q.type === 'door' || q.type === 'sdoor') {
      const w = 1.2, f = 0.09;
      const ox = alongX ? 1 : 0, oz = alongX ? 0 : 1;
      box(trim, mx - ox * (w / 2 + f / 2), y + 1.15, mz - oz * (w / 2 + f / 2), alongX ? f : 0.22, 2.3, alongX ? 0.22 : f);
      box(trim, mx + ox * (w / 2 + f / 2), y + 1.15, mz + oz * (w / 2 + f / 2), alongX ? f : 0.22, 2.3, alongX ? 0.22 : f);
      box(trim, mx, y + 2.3 + f / 2, mz, alongX ? w + 2 * f : 0.22, f, alongX ? 0.22 : w + 2 * f);
      if (q.type === 'sdoor') leaves.push({ key: q.key, L, mx, mz, alongX, y });
    } else if (q.type === 'window' || q.type === 'glass') {
      const w = q.type === 'glass' ? C : 1.4, y0 = q.type === 'glass' ? 0 : 0.95, y1 = q.type === 'glass' ? (L === 0 ? LH : CEIL) : 2.65;
      const ox = alongX ? 1 : 0, oz = alongX ? 0 : 1;
      R.quad(glassM, L, [mx - ox * w / 2, y + y0, mz - oz * w / 2], [ox * w, 0, oz * w], [0, y1 - y0, 0], alongX ? [0, 0, 1] : [1, 0, 0], 4);
      box(trim, mx, y + y0 + 0.02, mz, alongX ? w : 0.2, 0.05, alongX ? 0.2 : w);   // sill
      if (q.type === 'window') box(trim, mx, y + (y0 + y1) / 2, mz, alongX ? 0.05 : 0.08, y1 - y0, alongX ? 0.08 : 0.05);   // mullion
    }
  }
  for (const m of R.meshes()) { if (m.material === glassM) m.renderOrder = 5; root.add(m); }
  // security door leaves: steel panels that slide up into the lintel when open
  const leafGeo = new THREE.BoxGeometry(1.22, 2.3, 0.08);
  const leafMat = patch(new THREE.MeshStandardMaterial({ color: 0x6b2a22, roughness: 0.5, metalness: 0.6 }), 'object', { value: new THREE.Vector2(1, 0) });
  const lampGeo = new THREE.BoxGeometry(0.18, 0.08, 0.04);
  const DOORS = [];
  for (const lf of leaves) {
    const d = P.DOOR_OF.get(lf.key);
    let rec = DOORS.find((q) => q.door === d);
    if (!rec) { rec = { door: d, meshes: [], lamps: [] }; DOORS.push(rec); }
    const mat = leafMat.clone(); patch(mat, 'object', { value: new THREE.Vector2(...lightAt(lf.L, lf.mx, lf.y + 1.2, lf.mz, 0, 1, 0)) });
    const m = new THREE.Mesh(leafGeo, mat); m.position.set(lf.mx, lf.y + 1.15, lf.mz); if (!lf.alongX) m.rotation.y = Math.PI / 2;
    root.add(m); rec.meshes.push(m);
    for (const sd of [-1, 1]) {
      const lm = new THREE.Mesh(lampGeo, new THREE.MeshBasicMaterial({ color: 0x30ff60 }));
      lm.position.set(lf.mx + (lf.alongX ? 0 : sd * 0.13), lf.y + 2.55, lf.mz + (lf.alongX ? sd * 0.13 : 0)); if (!lf.alongX) lm.rotation.y = Math.PI / 2;
      root.add(lm); rec.lamps.push(lm);
    }
  }
  scene.add(root);
  onProgress('building');
  // ---------------------------------------------------------------- sky
  const hdr = await new HDRLoader().loadAsync(`${ASSET}hdri/kloofendal_48d_partly_cloudy_puresky.hdr`);
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  const pm = new THREE.PMREMGenerator(renderer);
  const env = pm.fromEquirectangular(hdr).texture;
  scene.environment = env; scene.background = hdr; scene.backgroundIntensity = 0.9; scene.environmentIntensity = 0.35;
  buildExterior(scene, root, env);
  onProgress('sky');
  // ---------------------------------------------------------------- fixtures and props
  await placeProps(scene, root, onProgress);
  return { root, DOORS, env };
}

// ------------------------------------------------------------------------------------------ outside
// The street and forecourt to the south, lawns either side, the cliff below the terrace, the sea.
function buildExterior(scene, root, env) {
  const tl = new THREE.TextureLoader();
  const ground = (name, w, d, x, z, rep, y = -0.02) => {
    const mk = (kind, srgb) => { const t = tl.load(`${ASSET}textures/${name}/${kind}.jpg`); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(w / rep, d / rep); t.anisotropy = 8; if (srgb) t.colorSpace = THREE.SRGBColorSpace; return t; };
    const m = patch(new THREE.MeshStandardMaterial({ map: mk('diff', true), normalMap: mk('nor'), roughnessMap: mk('rough'), roughness: 1, envMapIntensity: 0.5 }), 'object', { value: new THREE.Vector2(0, 1.0) });
    const g = new THREE.Mesh(new THREE.PlaneGeometry(w, d), m); g.rotation.x = -Math.PI / 2; g.position.set(x, y, z); root.add(g); return g;
  };
  const xa = X0, xb = X0 + NX * C, za = Z0, zt = Z0 + NZ * C;   // building footprint incl. terrace
  ground('cobblestone_floor_08', xb - xa + 24, 14, (xa + xb) / 2, za - 7, 3);          // forecourt
  ground('asphalt_02', 400, 12, 0, za - 20, 6, -0.04);                                  // the coast road
  ground('cobblestone_floor_08', 12, zt - za + 14, xa - 6, (za + zt) / 2 - 7, 3);       // side yards
  ground('cobblestone_floor_08', 12, zt - za + 14, xb + 6, (za + zt) / 2 - 7, 3);
  // grass beyond, out to the cliff line
  const grass = new THREE.MeshStandardMaterial({ color: 0x55603a, roughness: 0.95 });
  for (const [w, d, x, z] of [[160, zt - za + 40, xa - 12 - 80, (za + zt) / 2 - 20], [160, zt - za + 40, xb + 12 + 80, (za + zt) / 2 - 20], [400, 160, 0, za - 26 - 80]]) {
    const g = new THREE.Mesh(new THREE.PlaneGeometry(w, d), grass); g.rotation.x = -Math.PI / 2; g.position.set(x, -0.05, z); root.add(g);
  }
  // the cliff: a rough rock face falling 30 m from the terrace edge (and the lawns) to the sea
  const rock = new THREE.MeshStandardMaterial({ color: 0x6d6255, roughness: 1, flatShading: true });
  const cg = new THREE.PlaneGeometry(400, 34, 120, 12), pa = cg.attributes.position;
  for (let n = 0; n < pa.count; n++) { const x = pa.getX(n), y = pa.getY(n); pa.setZ(n, Math.sin(x * 0.7) * 0.8 + Math.sin(x * 0.23 + y) * 1.6 + (Math.random() - 0.5) * 0.8 - (17 - y) * 0.25); }
  cg.computeVertexNormals();
  const cliff = new THREE.Mesh(cg, rock); cliff.position.set(0, -17, zt + 0.2); cliff.rotation.y = Math.PI; root.add(cliff);
  // the sea
  const sea = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), new THREE.MeshStandardMaterial({ color: 0x1d3a46, roughness: 0.08, metalness: 0.0, envMapIntensity: 1.6 }));
  sea.rotation.x = -Math.PI / 2; sea.position.y = -30; root.add(sea);
}

// ------------------------------------------------------------------------------------------ props
// Each model becomes InstancedMeshes (one per sub-mesh) with a per-instance bake; placements are
// [model, L, x, z, rotY, y?, scale?]. Collision circles for the big ones.
export const OBSTACLES = [];   // {L, x, z, r}
export const BULBS = {};
async function placeProps(scene, root, onProgress) {
  const loader = new GLTFLoader();
  const place = [];   // [name, L, x, z, rot, y, scale, collideR]
  const rnd = mulberry(7);
  const add = (n, L, x, z, rot = 0, y = null, s = 1, cr = 0) => place.push([n, L, x, z, rot, y, s, cr]);
  // fixtures at the lamps
  for (const l of LAMPS) {
    if (l.fixture === 'ceiling') add('modern_ceiling_lamp_01', l.L, l.x, l.z, 0, l.L * LH + CEIL, 1.0);
    if (l.fixture === 'chandelier') add('Chandelier_01', 0, l.x, l.z, 0, l.y - 1.2, 2.2);
    if (l.fixture === 'chandelier3') add('Chandelier_03', l.L, l.x, l.z, 0, l.y - 0.4, 1.0);
  }
  const wx = (i) => X0 + i * C, wz = (k) => Z0 + k * C;   // cell corner
  const R = P.ROOMS;
  for (const r of R) {
    const [i0, k0, i1, k1] = r.rect, L = r.L;
    const xa = wx(i0), xb = wx(i1 + 1), za = wz(k0), zb = wz(k1 + 1), mx = (xa + xb) / 2, mz = (za + zb) / 2;
    if (r.kind === 'lobby') {
      for (const [x, z, rot] of [[mx - 5, mz - 2, Math.PI / 2], [mx + 5, mz + 2, -Math.PI / 2]]) { add('Sofa_01', 0, x, z, rot, null, 1, 1.0); }
      for (const [x, z] of [[mx - 3, mz - 2], [mx + 3, mz + 2]]) add('CoffeeTable_01', 0, x, z, 0, null, 1, 0.6);
      for (const [x, z, rot] of [[mx - 3, mz - 4, 0], [mx + 3, mz + 4, Math.PI], [mx - 3, mz + 0.2, Math.PI], [mx + 3, mz - 0.2, 0]]) add('ArmChair_01', 0, x, z, rot, null, 1, 0.5);
      for (const [x, z] of [[wx(10) + 0.6, wz(6) + 0.6], [wx(18) - 0.6, wz(6) + 0.6], [wx(10) + 0.6, wz(14) - 2.6], [wx(18) - 0.6, wz(12) + 0.4]]) add('potted_plant_02', 0, x, z, rnd() * 6, null, 1.3, 0.4);
    } else if (r.kind === 'reception') {
      add('ClassicConsole_01', 0, mx - 3, zb - 0.6, Math.PI, null, 1.4, 0.7); add('ClassicConsole_01', 0, mx + 3, zb - 0.6, Math.PI, null, 1.4, 0.7);
      add('CashRegister_01', 0, mx - 3, zb - 0.6, Math.PI, L * LH + 0.98, 1);
      add('planter_box_01', 0, xa + 1, za + 1, 0, null, 1, 0.5); add('planter_box_01', 0, xb - 1, za + 1, 0, null, 1, 0.5);
      add('wall_clock', 0, mx, zb - 0.1, Math.PI, L * LH + 2.6, 1.6);
    } else if (r.kind === 'ballroom') {
      for (const x of [mx - 7, mx, mx + 7]) for (const z of [mz - 0.5]) {
        add('round_wooden_table_01', 0, x, z, rnd() * 6, null, 1.2, 0.8);
        add('dining_chair_02', 0, x - 1.1, z, Math.PI / 2, null, 1, 0.35); add('dining_chair_02', 0, x + 1.1, z, -Math.PI / 2, null, 1, 0.35);
      }
    } else if (r.kind === 'bar' || r.kind === 'library') {
      if (r.kind === 'bar') { for (let n = 0; n < 4; n++) add('bar_chair_round_01', 0, xa + 2 + n * 1.2, za + 1.5, 0, null, 1, 0.3); add('coffee_table_round_01', 0, mx + 2, mz + 0.5, 0, null, 1, 0.5); }
      else { add('sofa_02', 0, mx, za + 1.2, 0, null, 1, 0.9); add('mid_century_lounge_chair', 0, mx - 2.5, mz + 1, rnd() * 6, null, 1, 0.5); add('side_table_01', 0, mx + 2, za + 1, 0, null, 1, 0.3); }
      add('fancy_picture_frame_01', 0, mx, (r.kind === 'bar' ? za : zb) + (r.kind === 'bar' ? 0.08 : -0.08), r.kind === 'bar' ? 0 : Math.PI, L * LH + 1.8, 1.5);
    } else if (r.kind === 'kitchen' || r.kind === 'staff') {
      add('wooden_crate_01', 0, xb - 1, za + 1, rnd(), null, 1, 0.5); add('cardboard_box_01', 0, xb - 1.2, za + 2.2, rnd(), null, 1, 0.35);
      add('WetFloorSign_01', 0, mx, mz, rnd() * 6, null, 1, 0); add('korean_fire_extinguisher_01', 0, xa + 0.4, za + 0.4, 0, null, 1, 0);
      if (r.kind === 'staff') { add('Television_01', 0, mx, zb - 0.5, Math.PI, L * LH + 0.75, 1, 0); add('sofa_02', 0, mx, mz - 0.5, 0, null, 1, 0.9); }
    } else if (r.kind === 'terrace') {
      for (const [x, z] of [[-14, 25], [-6, 27], [6, 27], [14, 25], [0, 31]]) add('outdoor_table_chair_set_01', 0, x, z, rnd() * 6, null, 1, 1.0);
      add('lifebuoy', 0, -18, Z0 + 28 * C - 0.15, 0, 1.4, 1.5);
      for (const x of [-20, 20]) add('potted_plant_02', 0, x, Z0 + 20 * C + 1, 0, null, 1.4, 0.4);
    } else if (r.kind === 'guestA' || r.kind === 'guestB' || r.kind === 'suite') {
      const southSide = mz < 0;   // window wall side; bed against the corridor wall
      const bedZ = southSide ? zb - 1.4 : za + 1.4;
      place.push(['__bed', L, mx, bedZ, southSide ? Math.PI : 0, null, 1, 1.0]);
      add('ClassicNightstand_01', L, mx + 1.3, southSide ? zb - 0.35 : za + 0.35, southSide ? Math.PI : 0, null, 1, 0.25);
      add(rnd() < 0.5 ? 'ArmChair_01' : 'modern_arm_chair_01', L, mx - 1.0, southSide ? za + 1.0 : zb - 1.0, rnd() * 6, null, 1, 0.45);
      add('hanging_picture_frame_01', L, mx, southSide ? zb - 0.06 : za + 0.06, southSide ? Math.PI : 0, L * LH + 2.0, 1.2);
      if (r.kind === 'suite') { add('Sofa_01', L, mx + 1.5, mz, rnd() < 0.5 ? Math.PI / 2 : -Math.PI / 2, null, 1, 0.9); add('CoffeeTable_01', L, mx - 0.5, mz, 0, null, 0.8, 0.5); }
    } else if (r.kind === 'gallery') {
      for (const [x, z, rot] of [[wx(8) + 0.1, wz(9) + 1, Math.PI / 2], [wx(20) - 0.1, wz(10) + 1, -Math.PI / 2]]) add('ornate_mirror_01', L, x, z, rot, L * LH + 1.6, 1.0);
      for (const [x, z] of [[wx(8) + 0.5, wz(4) + 0.5], [wx(20) - 0.5, wz(16) - 0.5]]) add('potted_plant_02', L, x, z, rnd() * 6, null, 1.1, 0.35);
    } else if (r.kind === 'hall') {
      add('korean_fire_extinguisher_01', L, xa + 0.25, mz + 1.5, Math.PI / 2, null, 1, 0);
    }
  }
  // glowing bulbs at every lamp (dim with the power)
  {
    const g = new THREE.SphereGeometry(0.07, 10, 8), m = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.0, 0.85, 0.6).multiplyScalar(3), toneMapped: false });
    const im = new THREE.InstancedMesh(g, m, LAMPS.length), M = new THREE.Matrix4();
    LAMPS.forEach((l, n) => { const dy = l.fixture === 'ceiling' ? -0.65 : l.fixture === 'sconce' ? -0.1 : -0.5; im.setMatrixAt(n, M.makeTranslation(l.x, l.y + dy, l.z)); });
    im.userData.glow = true; root.add(im); BULBS.mesh = im; BULBS.color = m.color.clone();
  }
  // stairwell sconces
  for (const l of LAMPS) if (l.fixture === 'sconce') add('industrial_wall_sconce', l.L, l.x, l.z, 0, l.y - 0.2, 1);
  // supplies: ammunition and first aid boxes (picked up by the director)
  // nothing big may stand in a doorway: drop furniture whose footprint reaches within 1 m of a door's middle
  const doorMids = [];
  for (const [key, e] of P.EDGE) {
    if (e !== 'door' && e !== 'sdoor') continue;
    const [L, i, k, sd] = key.split(':'); const x = sd === 'E' ? X0 + (+i + 1) * C : P.cx(+i), z = sd === 'N' ? Z0 + (+k + 1) * C : P.cz(+k);
    doorMids.push([+L, x, z]);
  }
  for (let n = place.length - 1; n >= 0; n--) {
    const q = place[n]; if (!(q[7] > 0)) continue;
    const r = q[7] * (q[6] || 1);
    if (doorMids.some(([L, x, z]) => L === q[1] && Math.hypot(x - q[2], z - q[3]) < r + 1.0)) place.splice(n, 1);
  }
  const names = [...new Set(place.map((q) => q[0]).filter((n) => !n.startsWith('__')))];
  const gl = {};
  let nDone = 0;
  await Promise.all(names.map(async (n) => { gl[n] = await loader.loadAsync(`${ASSET}models/${n}/${n}.gltf`); onProgress(`props ${++nDone}/${names.length}`); }));
  const groups = new Map();
  for (const q of place) { if (!groups.has(q[0])) groups.set(q[0], []); groups.get(q[0]).push(q); }
  // simple beds (box mattress, headboard): procedural, fabric textured
  const tl = new THREE.TextureLoader();
  const bedM = surfMat(tl, 'quatrefoil_jacquard_fabric', { color: 0xe8e0d0 });
  for (const [n, list] of groups) {
    if (n === '__bed') {
      const g = new THREE.BoxGeometry(1.7, 0.55, 2.1); g.translate(0, 0.38, 0);
      const hb = new THREE.BoxGeometry(1.8, 1.1, 0.12); hb.translate(0, 0.6, -1.05);
      const bm = patch(bedM.clone(), 'instance');
      const hm = patch(new THREE.MeshStandardMaterial({ color: 0x3a2618, roughness: 0.6 }), 'instance');
      for (const [geo, mat] of [[g, bm], [hb, hm]]) addInstanced(root, geo, mat, list, new THREE.Matrix4());
      continue;
    }
    const scn = gl[n].scene; scn.updateMatrixWorld(true);
    const bb = new THREE.Box3().setFromObject(scn);
    const hang = /lamp|Chandelier/.test(n) && !/sconce/.test(n), wall = /frame|mirror|clock|lifebuoy|sconce/.test(n);
    const off = new THREE.Matrix4().makeTranslation(0, hang ? -bb.max.y : wall ? -(bb.min.y + bb.max.y) / 2 : -bb.min.y, 0);
    scn.traverse((o) => {
      if (!o.isMesh) return;
      const mat = o.material.clone(); patch(mat, 'instance');
      addInstanced(root, o.geometry, mat, list, off.clone().multiply(o.matrixWorld));
    });
  }
  for (const q of place) if (q[7] > 0) OBSTACLES.push({ L: q[1], x: q[2], z: q[3], r: q[7] * (q[6] || 1) });
}
function addInstanced(root, geo, mat, list, local) {
  const im = new THREE.InstancedMesh(geo, mat, list.length);
  const bake = new Float32Array(list.length * 2), M = new THREE.Matrix4(), Q = new THREE.Quaternion(), S = new THREE.Vector3();
  list.forEach((q, n) => {
    const [, L, x, z, rot, y, s] = q;
    const yy = y == null ? L * LH : y;
    Q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rot); S.setScalar(s || 1);
    M.compose(new THREE.Vector3(x, yy, z), Q, S).multiply(local);
    im.setMatrixAt(n, M);
    const ld = lightAt(L, x, Math.max(yy, L * LH + 0.8), z, 0, 1, 0);
    bake[n * 2] = ld[0] * 0.8; bake[n * 2 + 1] = ld[1] * 0.8;
  });
  geo = geo.clone(); geo.setAttribute('ibake', new THREE.InstancedBufferAttribute(bake, 2));
  im.geometry = geo; im.instanceMatrix.needsUpdate = true; im.computeBoundingSphere();
  root.add(im);
}
export function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
