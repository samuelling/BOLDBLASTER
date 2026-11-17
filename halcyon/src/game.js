// Halcyon's play: the player, the pistol, the people in the corridors, and the director that runs
// the hotel (objectives, encounters, the stairwells' hidden risk, the security doors, the power).
// Everything here is stepped with a fixed dt so a bot can drive it frame by frame; humans drive it
// through main.js. Every event the models need is written to this.log as it happens.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import * as P from './plan.js';
import { los, lightAt, OBSTACLES, BULBS, patch, U, mulberry } from './world.js';

const { C, LH, X0, Z0 } = P;
const EYE = 1.62, RAD = 0.3;
const v3 = () => new THREE.Vector3();

// ------------------------------------------------------------------------------------------ collision
// blocking segments of a cell edge: walls and shut doors whole; doorways leave a 1.2 m gap
function edgeSegs(L, i, k, s) {
  const e = P.edgeAt(L, i, k, s);
  if (!e) return null;
  let x0, z0, x1, z1;
  if (s === 'E') { x0 = x1 = X0 + (i + 1) * C; z0 = Z0 + k * C; z1 = z0 + C; } else { z0 = z1 = Z0 + (k + 1) * C; x0 = X0 + i * C; x1 = x0 + C; }
  const open = e === 'door' || (e === 'sdoor' && P.DOOR_OF.get(P.edgeKey(L, i, k, s)).open);
  if (!open) return [[x0, z0, x1, z1]];
  const f = 0.4 / C;
  return [[x0, z0, x0 + (x1 - x0) * f, z0 + (z1 - z0) * f], [x1 - (x1 - x0) * f, z1 - (z1 - z0) * f, x1, z1]];
}
export function collide(L, pos, r = RAD) {
  const i = P.ci(pos.x), k = P.ck(pos.z);
  for (let a = i - 1; a <= i + 1; a++) for (let b = k - 1; b <= k + 1; b++) for (const s of ['E', 'N']) {
    const segs = edgeSegs(L, a, b, s); if (!segs) continue;
    for (const [x0, z0, x1, z1] of segs) {
      const dx = x1 - x0, dz = z1 - z0, l2 = dx * dx + dz * dz || 1e-9;
      const t = Math.max(0, Math.min(1, ((pos.x - x0) * dx + (pos.z - z0) * dz) / l2));
      const px = x0 + dx * t, pz = z0 + dz * t, ex = pos.x - px, ez = pos.z - pz, d = Math.hypot(ex, ez);
      if (d < r && d > 1e-6) { pos.x = px + ex / d * r; pos.z = pz + ez / d * r; }
      else if (d <= 1e-6) { pos.x += (s === 'E' ? 1 : 0) * r; pos.z += (s === 'N' ? 1 : 0) * r; }
    }
  }
  for (const o of OBSTACLES) {
    if (o.L !== L) continue;
    const dx = pos.x - o.x, dz = pos.z - o.z, d = Math.hypot(dx, dz), m = o.r + r * 0.8;
    if (d < m && d > 1e-6) { pos.x = o.x + dx / d * m; pos.z = o.z + dz / d * m; }
  }
}
// floor height and floor index at a point, given the floor we were on
export function ground(L, x, z) {
  const i = P.ci(x), k = P.ck(z), rp = P.rampAt(L, i, k);
  if (rp < 0) return [L * LH, L];
  const r = P.RAMPS[rp], y = P.rampY(r, x, z);
  return [y, y - r.L * LH > LH / 2 ? r.L + 1 : r.L];
}
export function wallDist(L, x, z) {   // distance to the nearest blocking edge (m), within 6 m
  let best = 6;
  const i = P.ci(x), k = P.ck(z);
  for (let a = i - 3; a <= i + 3; a++) for (let b = k - 3; b <= k + 3; b++) for (const s of ['E', 'N']) {
    const e = P.edgeAt(L, a, b, s); if (!e || e === 'door' || e === 'rail') continue;
    if (e === 'sdoor' && P.DOOR_OF.get(P.edgeKey(L, a, b, s)).open) continue;
    let x0, z0, x1, z1;
    if (s === 'E') { x0 = x1 = X0 + (a + 1) * C; z0 = Z0 + b * C; z1 = z0 + C; } else { z0 = z1 = Z0 + (b + 1) * C; x0 = X0 + a * C; x1 = x0 + C; }
    const dx = x1 - x0, dz = z1 - z0, t = Math.max(0, Math.min(1, ((x - x0) * dx + (z - z0) * dz) / (dx * dx + dz * dz)));
    best = Math.min(best, Math.hypot(x - x0 - dx * t, z - z0 - dz * t));
  }
  return best;
}

// ------------------------------------------------------------------------------------------ people
// Gunmen are mostly in tactical gear (Soldier) but some wear plain clothes (Michelle), as do the
// guests and staff; what tells them apart is what's in the hand, how it's held, and the light.
const ROLE = { gunman: { hostile: true }, guest: { hostile: false }, staff: { hostile: false } };
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _q = new THREE.Quaternion(), _wq = new THREE.Quaternion(), _pq = new THREE.Quaternion(), _m = new THREE.Matrix4();
function aimBone(bone, child, dir, w = 1) {   // turn bone (in world space) so that its child lies along dir
  bone.getWorldPosition(_a); child.getWorldPosition(_b);
  const cur = _b.sub(_a).normalize();
  _q.setFromUnitVectors(cur, dir);
  if (w < 1) _q.slerp(new THREE.Quaternion(), 1 - w);
  bone.getWorldQuaternion(_wq).premultiply(_q);
  bone.parent.getWorldQuaternion(_pq);
  bone.quaternion.copy(_pq.invert().multiply(_wq));
  bone.updateMatrixWorld(true);
}
const UP = new THREE.Vector3(0, 1, 0);
const dirOf = (f, r, a, b, c) => new THREE.Vector3().addScaledVector(f, a).addScaledVector(UP, b).addScaledVector(r, c).normalize();
class Actor {
  constructor(game, role, L, x, z, yaw, opts = {}) {
    Object.assign(this, { game, role, L, yaw, hostile: ROLE[role].hostile, hp: role === 'gunman' ? 100 : 60, alive: true, id: ++game.serial, state: 'idle', t: 0 });
    this.pos = new THREE.Vector3(x, L * LH, z);
    const rnd = game.rnd;
    this.pose = opts.pose || (this.hostile ? (rnd() < 0.55 ? 'aim' : 'low') : (rnd() < 0.5 ? 'hands' : 'phone'));
    this.react = 0.6 + rnd() * 0.9; this.fireT = 0; this.escort = !!opts.escort; this.objective = !!opts.objective;
    this.model = opts.model || (role === 'gunman' && rnd() < 0.7 ? 'soldier' : 'civ');
    const src = this.model === 'soldier' ? game.soldier : game.civ;
    const m = SkeletonUtils.clone(src.scene);
    this.bake = { value: new THREE.Vector2(1, 0) };
    const tint = this.model === 'soldier' ? new THREE.Color(0x2a2c30) : new THREE.Color(1, 1, 1).multiplyScalar(role === 'gunman' ? 0.7 : 0.9 + rnd() * 0.1);
    m.traverse((o) => {
      if (!o.isMesh) return;
      o.frustumCulled = false; o.material = o.material.clone();
      if (!o.name.includes('visor')) o.material.color = tint.clone();
      patch(o.material, 'object', this.bake);
    });
    this.mesh = m; m.rotation.order = 'YXZ';
    this.bones = {}; m.traverse((o) => { if (o.isBone) this.bones[o.name.replace(/^mixamorig:?/, '')] = o; });
    this.mixer = new THREE.AnimationMixer(m);
    this.acts = {}; if (this.model === 'soldier') for (const c of game.clips) this.acts[c.name] = this.mixer.clipAction(c);
    this.mixer.timeScale = 0.9 + rnd() * 0.2; this.mixer.update(rnd() * 3);
    this.play('Idle');
    if (this.pose === 'aim' || this.pose === 'low' || this.pose === 'phone') {
      this.item = this.pose === 'phone' ? new THREE.Mesh(game.phoneGeo, game.phoneMat) : game.pistol.clone();
      game.scene.add(this.item);
    }
    game.scene.add(m); this.sync(0);
  }
  play(n) { if (this.model !== 'soldier') { this.gait = n; return; } if (this.cur === n) return; const a = this.acts[n]; if (!a) return; if (this.cur) this.acts[this.cur].fadeOut(0.25); a.reset().fadeIn(0.25).play(); this.cur = n; }
  sync(dt) {
    this.mixer.update(dt);
    const m = this.mesh;
    m.position.copy(this.pos); m.rotation.set(0, this.yaw + (this.model === 'soldier' ? 0 : Math.PI), 0);
    if (!this.alive) { const k = Math.min(1, (this.deadT || 0) * 2.5); m.rotation.x = k * k * 1.45; m.position.y = this.pos.y + k * 0.12; }
    m.updateMatrixWorld(true);
    const b = this.bones, f = new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw)), r = new THREE.Vector3(-f.z, 0, f.x);
    const pose = this.alive ? (this.hostile && this.state === 'fight' ? 'aim' : this.pose) : 'dead';
    if (this.model !== 'soldier') {   // procedural stance and gait for the plain-clothes rig
      const mv = this.gait === 'Walk' || this.gait === 'Run' ? 1 : 0;
      this.ph = (this.ph || 0) + dt * (this.gait === 'Run' ? 11 : 6.5) * mv;
      const sw = Math.sin(this.ph) * 0.42 * mv, br = Math.sin(this.t * 1.3 + this.id) * 0.02;
      aimBone(b.Spine, b.Spine1 || b.Spine2, dirOf(f, r, 0.04 + br, 1, 0));
      for (const [sd, up, lo, ft] of [[1, b.RightUpLeg, b.RightLeg, b.RightFoot], [-1, b.LeftUpLeg, b.LeftLeg, b.LeftFoot]]) {
        const s2 = sw * sd;
        aimBone(up, lo, dirOf(f, r, s2, -1, sd * 0.06)); aimBone(lo, ft, dirOf(f, r, s2 - 0.3 * mv * Math.max(0, -Math.cos(this.ph) * sd) - 0.02, -1, sd * 0.03));
      }
      if (pose !== 'hands') for (const [sd, up, lo, hd] of [[1, b.RightArm, b.RightForeArm, b.RightHand], [-1, b.LeftArm, b.LeftForeArm, b.LeftHand]]) {
        aimBone(up, lo, dirOf(f, r, -sw * sd * 0.5 + 0.02, -1, sd * 0.16)); aimBone(lo, hd, dirOf(f, r, -sw * sd * 0.5 + 0.18, -1, sd * 0.08));
      }
    }
    if (pose === 'hands') {
      aimBone(b.RightArm, b.RightForeArm, dirOf(f, r, 0.1, 0.8, 0.55)); aimBone(b.RightForeArm, b.RightHand, dirOf(f, r, 0.15, 1, 0.12));
      aimBone(b.LeftArm, b.LeftForeArm, dirOf(f, r, 0.1, 0.8, -0.55)); aimBone(b.LeftForeArm, b.LeftHand, dirOf(f, r, 0.15, 1, -0.12));
    } else if (pose === 'aim') {
      const p = this.game.player, tgt = new THREE.Vector3(p.pos.x, p.pos.y + 1.4, p.pos.z);
      b.RightArm.getWorldPosition(_a); const d = tgt.sub(_a).normalize();
      const dd = this.state === 'fight' || this.state === 'alert' ? d : dirOf(f, r, 1, -0.15, 0);
      aimBone(b.RightArm, b.RightForeArm, dd.clone().addScaledVector(r, -0.12).normalize()); aimBone(b.RightForeArm, b.RightHand, dd);
      aimBone(b.LeftArm, b.LeftForeArm, dd.clone().addScaledVector(r, 0.55).addScaledVector(UP, -0.2).normalize());
      b.RightHand.getWorldPosition(_b); b.LeftForeArm.getWorldPosition(_a); aimBone(b.LeftForeArm, b.LeftHand, _b.sub(_a).normalize());
    } else if (pose === 'low') {
      aimBone(b.RightArm, b.RightForeArm, dirOf(f, r, 0.35, -1, 0.12)); aimBone(b.RightForeArm, b.RightHand, dirOf(f, r, 0.9, -0.6, 0.05));
    } else if (pose === 'phone') {
      aimBone(b.RightArm, b.RightForeArm, dirOf(f, r, 0.25, -1, 0.15)); aimBone(b.RightForeArm, b.RightHand, dirOf(f, r, 0.55, 0.55, -0.35));
    }
    if (this.item) {
      this.item.visible = this.alive;
      // hold it in the right hand: pistol barrel (+x) along the forearm, grip down; phone screen to the face
      const h = b.RightHand.getWorldPosition(new THREE.Vector3()), e = b.RightForeArm.getWorldPosition(new THREE.Vector3());
      const x = h.clone().sub(e).normalize(), y = UP.clone().addScaledVector(x, -x.y).normalize(), z = new THREE.Vector3().crossVectors(x, y);
      _m.makeBasis(x, y, z); this.item.quaternion.setFromRotationMatrix(_m);
      this.item.position.copy(h).addScaledVector(x, 0.06).addScaledVector(y, this.pose === 'phone' ? 0.02 : 0.03);
      if (this.pose === 'phone') this.item.rotateY(-0.6);
    }
    const ld = lightAt(this.L, this.pos.x, this.pos.y + 1.2, this.pos.z, 0, 1, 0);
    this.bake.value.set(ld[0] * 0.9 + 0.04, ld[1] * 0.9);
  }
  dispose() { this.game.scene.remove(this.mesh); if (this.item) this.game.scene.remove(this.item); this.mixer.stopAllAction(); }
  head() { return new THREE.Vector3(this.pos.x, this.pos.y + 1.6, this.pos.z); }
}

// ------------------------------------------------------------------------------------------ the game
export class Game {
  constructor(G) { this.G = G; this.scene = G.scene; this.camera = G.camera; this.serial = 0; this.groups = 0; }
  async init() {
    const gl = new GLTFLoader();
    this.soldier = await gl.loadAsync('assets/characters/Soldier.glb');
    this.civ = await gl.loadAsync('assets/characters/Michelle.glb');
    // the Soldier's clips drive both rigs (same Mixamo bone names); root motion stays put
    this.clips = this.soldier.animations.filter((c) => c.name !== 'TPose').map((c) => { c = c.clone(); c.tracks = c.tracks.filter((t) => !t.name.endsWith('.position')); return c; });
    // Michelle's rig rests differently from the Soldier's, so she is posed procedurally (aimBone) instead of by clips
    const pg = await gl.loadAsync('assets/models/service_pistol/service_pistol.gltf');
    // the asset holds two pistols and spare parts: keep pistol A (frame, slide, hammer, trigger)
    const keep = ['service_pistol_pistol_a', 'service_pistol_slide_a', 'service_pistol_hammer_a', 'service_pistol_trigger_a'];
    const one = new THREE.Group(); for (const n of keep) { const o = pg.scene.getObjectByName(n); if (o) one.add(o.clone()); }
    pg.scene = one; this.pistol = one;
    // supply crates (Poly Haven ammo box, medical box)
    this.crates = {};
    for (const [k, n] of [['ammo', 'ammo_box'], ['aid', 'medical_box']]) {
      const c = (await gl.loadAsync(`assets/models/${n}/${n}.gltf`)).scene;
      const bb = new THREE.Box3().setFromObject(c), sz = bb.getSize(new THREE.Vector3()), sc = 0.5 / Math.max(sz.x, sz.z);
      c.scale.setScalar(sc); c.position.y = -bb.min.y * sc;
      c.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); patch(o.material, 'object', { value: new THREE.Vector2(0.9, 0.2) }); } });
      const grp = new THREE.Group(); grp.add(c); this.crates[k] = grp;
    }
    this.pistol.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); patch(o.material, 'object', { value: new THREE.Vector2(0.6, 0.2) }); } });
    this.phoneGeo = new THREE.BoxGeometry(0.075, 0.012, 0.15);
    this.phoneMat = new THREE.MeshStandardMaterial({ color: 0x101012, roughness: 0.25, metalness: 0.4, emissive: 0x2a3a55, emissiveIntensity: 0.6 });
    // first-person pistol
    this.view = new THREE.Group();
    const vp = pg.scene.clone(); vp.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); patch(o.material, 'object', (this.viewBake = { value: new THREE.Vector2(1, 0) })); } });
    vp.scale.setScalar(1.6); this.slide = vp.getObjectByName('service_pistol_slide_a'); this.slideX = this.slide ? this.slide.position.x : 0; vp.rotation.set(0, Math.PI / 2, 0); vp.position.set(0.17, -0.19, -0.42);
    this.view.add(vp); this.vp = vp;
    // the hand and pistol draw in their own pass (own scene and camera) so they never sink into walls
    this.viewScene = new THREE.Scene(); this.viewCam = new THREE.PerspectiveCamera(60, this.camera.aspect, 0.01, 5);
    this.viewScene.add(this.viewCam); this.viewCam.add(this.view);
    this.viewScene.add(new THREE.HemisphereLight(0xfff0e0, 0x302820, 0.25));
    this.flash = new THREE.PointLight(0xffc070, 0, 9, 2); this.flash.position.set(0.17, -0.12, -0.8);
    this.torch = new THREE.SpotLight(0xfff1dc, 0, 30, 0.42, 0.5, 1.5); this.torch.position.set(0.12, -0.15, 0); this.torch.target.position.set(0, 0, -5);
    this.camera.add(this.flash, this.torch, this.torch.target); this.scene.add(this.camera);
    this.vflash = this.flash.clone(); this.viewCam.add(this.vflash);
    this.flashSprite = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.16), new THREE.MeshBasicMaterial({ color: 0xffd090, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, map: flashTex() }));
    this.flashSprite.position.set(0.17, -0.135, -0.78); this.view.add(this.flashSprite);
    this.reset();
  }
  reset() {
    for (const a of this.actors || []) a.dispose();
    this.actors = []; this.items = [];
    const rnd = (this.rnd = mulberry(this.seed || 1));
    this.t = 0;
    this.player = { pos: new THREE.Vector3(P.cx(13) + 1, 0, P.cz(5)), L: 0, yaw: Math.PI, pitch: 0, vel: new THREE.Vector3(), hp: 100, ammo: 12, reserve: 60, reloadT: -1, cool: 0, kick: 0, lives: 0, torch: false, rampFrom: null, lastCell: -1, cellPath: [] };
    this.power = 1; this.powerUntil = 0; this.pendingDark = null;
    this.score = { rescued: 0, wrong: 0, hostiles: 0 };
    // the stairwells' hidden risk: a slowly drifting chance of an ambush on each flight
    this.risk = P.RAMPS.map(() => 0.2 + rnd() * 0.6);
    this.nextEnc = 3; this.nextDoor = 8; this.nextRisk = 20;
    this.log = { calls: [], choices: [], outcomes: [], doors: [], doorMet: [], blackouts: [], objectives: [], resp: [], shots: [], kills: [], deaths: [], spawns: [], pickups: [], supplies: [] };
    for (const d of P.DOORS) d.open = true;
    this.doorUse = new Map();
    this.objective = null; this.newObjective();
    this.msg('FIND THE GUESTS. WALK THEM OUT.');
  }
  msg(t) { this.lastMsg = [this.t, t]; if (this.G.hud) this.G.hud.msg(t); }

  // -------------------------------------------------------------------------------- objectives
  newObjective() {
    const rooms = P.ROOMS.filter((r) => /guest|suite|bar|library|kitchen|staff|ballroom/.test(r.kind) && (!this.objective || r.id !== this.objective.room));
    const r = rooms[Math.floor(this.rnd() * rooms.length)];
    const [i0, k0, i1, k1] = r.rect;
    let i, k, n = 0;
    do { i = i0 + Math.floor(this.rnd() * (i1 - i0 + 1)); k = k0 + Math.floor(this.rnd() * (k1 - k0 + 1)); n++; } while ((!P.walkable(r.L, i, k) || this.blockedCell(r.L, i, k)) && n < 50);
    const g = new Actor(this, 'guest', r.L, P.cx(i), P.cz(k), this.rnd() * 6.28, { pose: 'hands', objective: true });
    this.actors.push(g);
    this.objective = { room: r.id, name: r.name, guest: g, phase: 'find', t0: this.t };
    this.log.objectives.push([this.t, 'new', r.id, r.L, i, k, g.id]);
  }
  blockedCell(L, i, k) { const x = P.cx(i), z = P.cz(k); return OBSTACLES.some((o) => o.L === L && Math.hypot(o.x - x, o.z - z) < o.r + 0.6); }
  extractionAt(L, x, z) {
    if (L !== 0) return false;
    const i = P.ci(x), k = P.ck(z), r = P.roomAt(0, i, k);
    return (k <= 0 && i >= 12 && i <= 15) || (r >= 0 && P.ROOMS[r].outdoor && k >= 22);
  }

  // -------------------------------------------------------------------------------- the step
  step(dt, inp) {
    this.t += dt;
    const p = this.player;
    this.stepPlayer(dt, inp);
    this.stepWeapon(dt, inp);
    this.stepActors(dt);
    this.stepDirector(dt, inp);
    this.trackCalls();
    // power and the torch
    if (this.power < 0.5 && this.t > this.powerUntil) { this.power = 1; this.msg('POWER BACK'); this.log.blackouts.push([this.t, 'on']); }
    U.uPower.value += ((this.power > 0.5 ? 1 : 0.02) - U.uPower.value) * Math.min(1, dt * 6);
    if (BULBS.mesh) BULBS.mesh.material.color.copy(BULBS.color).multiplyScalar(U.uPower.value);
    this.torch.intensity = p.torch ? 9 : 0;
    this.flash.intensity = Math.max(0, this.flash.intensity - dt * 400); this.vflash.intensity = this.flash.intensity * 0.3;
    this.flashSprite.material.opacity = Math.min(1, this.flash.intensity / 20); this.flashSprite.rotation.z = this.t * 37;
    this.viewScene.environment = this.scene.environment; this.viewScene.environmentIntensity = 0.25 * (0.3 + 0.7 * U.uPower.value);
    this.animateDoors(dt);
    // camera
    const cam = this.camera;
    cam.position.set(p.pos.x, p.pos.y + EYE, p.pos.z);
    cam.rotation.set(p.pitch + p.kick, p.yaw, 0, 'YXZ');
    p.kick *= Math.exp(-dt * 14);
    const ld = lightAt(p.L, p.pos.x, p.pos.y + 1.2, p.pos.z, 0, 1, 0);
    this.viewBake.value.set(ld[0] * 0.8 + 0.05 + (p.torch ? 0.4 : 0), ld[1] * 0.6);
    this.vp.rotation.z = p.kick * 5; if (this.slide) this.slide.position.x = this.slideX - Math.min(0.03, p.kick * 0.9); this.vp.position.z = -0.42 + p.kick * 2; this.vp.position.y = -0.19 + Math.sin(this.t * 9) * 0.004 * Math.min(1, p.vel.length() / 3) - (p.reloadT > 0 ? 0.12 : 0);
  }

  stepPlayer(dt, inp) {
    const p = this.player;
    p.yaw -= inp.lookX || 0; p.pitch = Math.max(-1.3, Math.min(1.3, p.pitch - (inp.lookY || 0)));
    const sp = inp.run ? 5.2 : 3.0;
    const fx = -Math.sin(p.yaw), fz = -Math.cos(p.yaw), rx = Math.cos(p.yaw), rz = -Math.sin(p.yaw);
    const mx = (inp.moveZ || 0) * fx + (inp.moveX || 0) * rx, mz = (inp.moveZ || 0) * fz + (inp.moveX || 0) * rz;
    const ml = Math.hypot(mx, mz) || 1;
    const tv = new THREE.Vector3(mx / Math.max(1, ml) * sp, 0, mz / Math.max(1, ml) * sp);
    p.vel.lerp(tv, Math.min(1, dt * 10));
    const old = p.pos.clone();
    p.pos.x += p.vel.x * dt; p.pos.z += p.vel.z * dt;
    collide(p.L, p.pos);
    const [y, L] = ground(p.L, p.pos.x, p.pos.z);
    if (!P.walkable(L, P.ci(p.pos.x), P.ck(p.pos.z))) { p.pos.copy(old); } else { p.pos.y = y; p.L = L; }
    p.vel.set((p.pos.x - old.x) / dt, 0, (p.pos.z - old.z) / dt);
    if (inp.torchToggle) p.torch = !p.torch;
  }

  stepWeapon(dt, inp) {
    const p = this.player;
    p.cool -= dt;
    if (p.reloadT > 0) { p.reloadT -= dt; if (p.reloadT <= 0) { const n = Math.min(12 - p.ammo, p.reserve); p.ammo += n; p.reserve -= n; p.reloadT = -1; } return; }
    if ((inp.reload && p.ammo < 12 && p.reserve > 0) || (p.ammo === 0 && p.reserve > 0 && inp.fire)) { p.reloadT = 1.4; return; }
    if (inp.fire && p.cool <= 0 && p.ammo > 0) {
      p.ammo--; p.cool = 0.2; p.kick += 0.035; this.flash.intensity = 30;
      const hit = this.hitscan(), aim = this.aimTarget();
      this.log.shots.push([+this.t.toFixed(2), hit ? hit.a.id : -1, hit ? +hit.d.toFixed(2) : -1, hit ? (hit.head ? 2 : 1) : 0, aim ? aim[0].id : -1, aim ? +aim[1].toFixed(2) : -1]);
      if (aim) this.resolveCall(aim[0], 'shoot');
      if (hit) this.damage(hit.a, hit.head ? 100 : 40);
    }
  }
  hitscan() {
    const cam = this.camera, o = cam.position.clone(), d = new THREE.Vector3(0, 0, -1).applyEuler(cam.rotation);
    let best = null;
    const aim = this.aimTarget();
    for (const a of this.actors) {
      if (!a.alive) continue;
      if (Math.abs(a.pos.y - this.player.pos.y) > 2.5) continue;
      if (a.objective && (!aim || aim[0] !== a)) continue;   // the guest you came for keeps out of the line of fire
      // body: segment from hips to neck, radius 0.24; head: sphere radius 0.14
      const parts = [[new THREE.Vector3(a.pos.x, a.pos.y + 0.2, a.pos.z), new THREE.Vector3(a.pos.x, a.pos.y + 1.45, a.pos.z), 0.24, false], [a.head(), a.head(), 0.14, true]];
      for (const [s0, s1, r, head] of parts) {
        const t = rayVsSegment(o, d, s0, s1);
        if (t && t.dist < r && (!best || t.t < best.d)) best = { a, d: t.t, head };
      }
    }
    if (!best) return null;
    const hp = o.clone().addScaledVector(d, best.d);
    if (!los(this.player.L, o.x, o.z, hp.x, hp.z)) return null;
    return best;
  }
  damage(a, n) {
    a.hp -= n; a.state = a.hostile ? 'fight' : a.state; a.hitT = this.t;
    if (a.hp <= 0 && a.alive) {
      a.alive = false; a.deadT = 0; a.play('Idle');
      this.log.kills.push([this.t, a.id, a.hostile ? 1 : 0, a.role]);
      if (a.hostile) this.score.hostiles++; else { this.score.wrong++; this.msg(a.objective ? 'YOU SHOT THE GUEST' : 'NOT A GUNMAN'); }
      if (a.objective && this.objective && this.objective.guest === a) { this.log.objectives.push([this.t, 'failed', this.objective.room]); this.objective = null; this.nextObjT = this.t + 3; }
    }
  }

  stepActors(dt) {
    const p = this.player, eye = new THREE.Vector3(p.pos.x, p.pos.y + EYE, p.pos.z);
    for (const a of this.actors) {
      a.t += dt;
      if (!a.alive) { a.deadT += dt; a.sync(dt); continue; }
      const dx = p.pos.x - a.pos.x, dz = p.pos.z - a.pos.z, d = Math.hypot(dx, dz);
      const sameFloor = a.L === p.L && Math.abs(a.pos.y - p.pos.y) < 2.5;
      const sees = sameFloor && d < 26 && los(a.L, a.pos.x, a.pos.z, p.pos.x, p.pos.z);
      if (a.state === 'approach') {
        const lead = a.leader && a.leader.alive && a.leader.state === 'approach' ? a.leader : null;
        if ((sees && d < 15) || (a.leader && !lead) || d < 4 || a.t > 14) {
          a.state = a.hostile ? (sees ? 'alert' : 'idle') : 'idle'; a.alertT = this.t; a.play('Idle');
        } else {
          let tx, tz;
          if (lead) { tx = lead.pos.x - Math.sin(lead.yaw) * a.off[1] + Math.cos(lead.yaw) * a.off[0]; tz = lead.pos.z - Math.cos(lead.yaw) * a.off[1] - Math.sin(lead.yaw) * a.off[0]; }
          else { const w = a.route[Math.min(a.ri, a.route.length - 1)]; tx = w[1]; tz = w[2]; if (Math.hypot(tx - a.pos.x, tz - a.pos.z) < 0.5 && a.ri < a.route.length - 1) a.ri++; }
          const ddx = tx - a.pos.x, ddz = tz - a.pos.z, dd = Math.hypot(ddx, ddz);
          if (dd > 0.15) {
            const sp = Math.min(lead ? a.walkSp * 1.25 : a.walkSp, dd / dt);
            a.pos.x += ddx / dd * sp * dt; a.pos.z += ddz / dd * sp * dt;
            const ty = Math.atan2(-ddx, -ddz); a.yaw += Math.atan2(Math.sin(ty - a.yaw), Math.cos(ty - a.yaw)) * Math.min(1, dt * 6);
            collide(a.L, a.pos, 0.28); const [y, L] = ground(a.L, a.pos.x, a.pos.z); a.pos.y = y; a.L = L;
            a.play('Walk');
          } else a.play('Idle');
          a.sync(dt); continue;
        }
      }
      if (a.hostile) {
        if (sees && a.state === 'idle') { a.state = 'alert'; a.alertT = this.t; }
        if (a.state === 'alert' && this.t - a.alertT > a.react) a.state = 'fight';
        if (a.state === 'fight' || a.state === 'alert') a.yaw = Math.atan2(-dx, -dz);
        if (a.state === 'fight' && sees) {
          a.fireT -= dt;
          if (a.fireT <= 0) {
            a.fireT = 0.7 + Math.random() * 0.6;
            const dark = this.power < 0.5 ? 0.25 : 0;
            const ph = Math.max(0.08, 0.5 - d * 0.016 - dark - Math.min(0.2, p.vel.length() * 0.03));
            if (Math.random() < ph) { p.hp -= 11; this.hurtT = this.t; }
            a.muzzleT = this.t;
          }
        }
        a.play(a.state === 'idle' ? 'Idle' : 'Idle');
      } else if (a.escort && this.objective && this.objective.guest === a) {
        // follow the player's trail
        const tgt = p.cellPath.length > 3 ? p.cellPath[Math.max(0, p.cellPath.length - 3)] : p.pos;
        const tx = tgt.x - a.pos.x, tz = tgt.z - a.pos.z, td = Math.hypot(tx, tz);
        if (d > 2.2 && td > 0.4) {
          const sp = Math.min(d > 6 ? 5 : 3.2, td / dt);
          a.pos.x += tx / td * sp * dt; a.pos.z += tz / td * sp * dt; a.yaw = Math.atan2(-tx, -tz);
          collide(a.L, a.pos, 0.28);
          const [y, L] = ground(a.L, a.pos.x, a.pos.z); a.pos.y = y; a.L = L;
          a.play(sp > 3.5 ? 'Run' : 'Walk');
        } else a.play('Idle');
        if (d > 14 && !los(a.L, a.pos.x, a.pos.z, p.pos.x, p.pos.z)) { const q = p.cellPath[Math.max(0, p.cellPath.length - 4)] || p.pos; a.pos.set(q.x, q.y, q.z); a.L = p.L; }
      } else {
        if (sees && d < 12) a.yaw += (Math.atan2(-dx, -dz) - a.yaw) * Math.min(1, dt * 2);
        a.play('Idle');
      }
      a.sync(dt);
    }
    // the player's trail (for the escort) and cells walked (for the homing walks)
    const cell = P.nodeId(p.L, P.ci(p.pos.x), P.ck(p.pos.z));
    if (cell !== p.lastCell) { p.cellPath.push(p.pos.clone()); if (p.cellPath.length > 60) p.cellPath.shift(); p.lastCell = cell; }
    // tidy away people far behind
    this.actors = this.actors.filter((a) => {
      if (a.objective && this.objective && this.objective.guest === a) return true;
      const far = Math.hypot(a.pos.x - p.pos.x, a.pos.z - p.pos.z) > 32 || a.L !== p.L;
      const old = (!a.alive && a.deadT > 25) || (far && a.t > 15);
      if (old) a.dispose();
      return !old;
    });
  }

  // -------------------------------------------------------------------------------- the director
  stepDirector(dt, inp) {
    const p = this.player, rnd = this.rnd;
    // death: next life starts in the lobby
    if (p.hp <= 0) {
      this.log.deaths.push([this.t]); p.lives++;
      p.hp = 100; p.ammo = 12; p.reserve = Math.max(p.reserve, 36); p.pos.set(P.cx(13 + Math.floor(rnd() * 2)), 0, P.cz(4)); p.L = 0; p.yaw = Math.PI; p.pitch = 0; p.cellPath = [];
      for (const a of this.actors) if (a.hostile) a.state = 'idle';
      if (this.objective && this.objective.phase === 'escort') { this.objective.guest.escort = false; this.objective.phase = 'find'; }
      this.msg('YOU ARE DOWN. BACK IN THE LOBBY.');
    }
    // objectives
    const o = this.objective;
    if (!o && this.t > (this.nextObjT || 0)) this.newObjective();
    if (o && o.phase === 'find' && o.guest.alive) {
      const g = o.guest, d = Math.hypot(g.pos.x - p.pos.x, g.pos.z - p.pos.z);
      if (g.L === p.L && d < 2.6 && inp.use) {
        o.phase = 'escort'; g.escort = true; g.pose = 'none'; this.log.pickups.push([this.t, o.room, g.id]);
        this.msg('WITH YOU. OUT BY THE FRONT DOORS OR THE TERRACE.');
        if (rnd() < 0.7) this.pendingDark = this.t + 2 + rnd() * 3;
      }
    }
    if (o && o.phase === 'escort' && this.extractionAt(p.L, p.pos.x, p.pos.z)) {
      this.score.rescued++; this.log.objectives.push([this.t, 'rescued', o.room]); this.msg('GUEST OUT SAFE');
      o.guest.dispose(); this.actors = this.actors.filter((a) => a !== o.guest); this.objective = null; this.nextObjT = this.t + 1.5;
    }
    // blackouts
    if (this.pendingDark != null && this.t > this.pendingDark && this.power > 0.5) {
      this.pendingDark = null; this.power = 0; this.powerUntil = this.t + 15 + rnd() * 10;
      this.log.blackouts.push([this.t, 'off', this.powerUntil]); this.msg('THE POWER IS OUT');
    }
    // encounters: someone in the corridor ahead every 4-10 s
    this.nextEnc -= dt;
    if (this.nextEnc <= 0) { if (this.spawnGroup()) this.nextEnc = 4 + rnd() * 6; else this.nextEnc = 0.7; }
    // security doors: the gang's hand on the building's system
    this.nextDoor -= dt;
    if (this.nextDoor <= 0) { this.toggleDoor(); this.nextDoor = 8 + rnd() * 10; }
    // the stairwells' risk drifts
    this.nextRisk -= dt;
    if (this.nextRisk <= 0) { this.nextRisk = 20; this.risk = this.risk.map((r) => Math.min(0.92, Math.max(0.08, r + (rnd() - 0.5) * 0.24))); }
    // stair choices and their outcomes
    const rp = P.rampAt(p.L, P.ci(p.pos.x), P.ck(p.pos.z));
    if (rp >= 0 && p.rampFrom == null) { p.rampFrom = { ramp: rp, t: this.t, L: p.L }; this.log.choices.push([this.t, rp, p.L, +this.risk[rp].toFixed(3)]); }
    if (rp < 0 && p.rampFrom != null) {
      const f = p.rampFrom; p.rampFrom = null;
      if (p.L !== f.L) {   // came out the other end
        const amb = rnd() < this.risk[f.ramp];
        this.log.outcomes.push([this.t, f.ramp, amb ? 1 : 0, p.L]);
        if (amb) this.ambush(); else this.supply();
      }
    }
    // supply pick-ups
    for (const it of this.items) if (!it.taken && it.L === p.L && Math.hypot(it.x - p.pos.x, it.z - p.pos.z) < 1.2) {
      it.taken = true; this.scene.remove(it.mesh);
      if (it.kind === 'ammo') p.reserve += 24; else p.hp = Math.min(100, p.hp + 40);
      this.log.supplies.push([this.t, it.kind]); this.msg(it.kind === 'ammo' ? 'AMMUNITION' : 'FIRST AID');
    }
    // door use, for where the shelling... where the gang shuts doors
    for (const d of P.DOORS) {
      if (d.L !== p.L) continue;
      const dx = P.cx(d.i) + (d.s === 'E' ? C / 2 : d.s === 'W' ? -C / 2 : 0) - p.pos.x, dz = P.cz(d.k) + (d.s === 'N' ? C / 2 : d.s === 'S' ? -C / 2 : 0) - p.pos.z;
      const dd = Math.hypot(dx, dz);
      if (dd < 1.0) this.doorUse.set(d.id, this.t);
      if (!d.open && dd < 2.6 && !d.metAt || (!d.open && dd < 2.6 && this.t - d.metAt > 20)) { d.metAt = this.t; this.log.doorMet.push([this.t, d.id]); }
    }
  }
  // Someone steps into the corridor ahead: a group of 1-5 starts out of sight 5-10 cells away
  // along the walkable grid and walks towards the player until it has them in view, then stops.
  spawnGroup() {
    const p = this.player, rnd = this.rnd;
    if (this.actors.filter((a) => a.alive && !a.objective).length > 9) return false;
    const src = P.nodeId(p.L, P.ci(p.pos.x), P.ck(p.pos.z));
    const dist = P.bfs(src), fx = -Math.sin(p.yaw), fz = -Math.cos(p.yaw), cands = [];
    for (const [n, d] of dist) {
      if (d < 5 || d > 10 || P.nodeL(n) !== p.L) continue;
      const i = P.nodeI(n), k = P.nodeK(n);
      if (P.kindAt(p.L, i, k) !== 1 || this.blockedCell(p.L, i, k)) continue;
      const x = P.cx(i), z = P.cz(k), dx = x - p.pos.x, dz = z - p.pos.z, e = Math.hypot(dx, dz);
      if (los(p.L, p.pos.x, p.pos.z, x, z)) continue;   // out of sight
      if (this.actors.some((a) => a.alive && a.L === p.L && Math.hypot(a.pos.x - x, a.pos.z - z) < 4)) continue;
      const ahead = (dx * fx + dz * fz) / e;
      cands.push([n, i, k, 1 + Math.max(0, ahead) * 3 + (this.objPathSet && this.objPathSet.has(n) ? 6 : 0)]);
    }
    if (!cands.length) return false;
    let u = rnd() * cands.reduce((s, c) => s + c[3], 0), pick = cands[0];
    for (const c of cands) { u -= c[3]; if (u <= 0) { pick = c; break; } }
    const [n0, i, k] = pick;
    const route = P.path(n0, src);
    if (!route) return false;
    const pu = rnd(), num = pu < 0.35 ? 1 : pu < 0.65 ? 2 : pu < 0.85 ? 3 : pu < 0.95 ? 4 : 5;
    const gid = ++this.groups, ids = [];
    let leader = null;
    for (let m = 0; m < num; m++) {
      const off = m ? [(rnd() - 0.5) * 1.6, -0.8 - rnd() * 1.4] : [0, 0];
      const role = rnd() < 0.45 ? 'gunman' : rnd() < 0.72 ? 'guest' : 'staff';
      const a = new Actor(this, role, p.L, P.cx(i) + off[0] * 0.5, P.cz(k) + off[1] * 0.5, 0);
      a.state = 'approach'; a.route = route.slice(1).map((q) => [P.nodeL(q), P.cx(P.nodeI(q)), P.cz(P.nodeK(q))]); a.ri = 0; a.off = off; a.group = gid; a.leader = leader; a.walkSp = 1.4 + rnd() * 0.4;
      if (!m) leader = a;
      a.pos.y = ground(p.L, a.pos.x, a.pos.z)[0];
      this.actors.push(a); ids.push([a.id, a.hostile ? 1 : 0, a.pose]);
    }
    this.log.spawns.push([+this.t.toFixed(2), p.L, i, k, gid, ids]);
    return true;
  }
  ambush() {
    const p = this.player, fx = -Math.sin(p.yaw), fz = -Math.cos(p.yaw);
    for (let m = 0; m < 2; m++) {
      const pos = new THREE.Vector3(p.pos.x + fx * (5 + m * 2) + (this.rnd() - 0.5) * 2, 0, p.pos.z + fz * (5 + m * 2) + (this.rnd() - 0.5) * 2);
      collide(p.L, pos, 0.3);
      if (!P.walkable(p.L, P.ci(pos.x), P.ck(pos.z))) continue;
      const a = new Actor(this, 'gunman', p.L, pos.x, pos.z, 0, { pose: 'aim' }); a.state = 'alert'; a.alertT = this.t; a.react = 0.7 + this.rnd() * 0.5;
      a.pos.y = ground(p.L, pos.x, pos.z)[0]; this.actors.push(a);
    }
    this.msg('AMBUSH');
  }
  supply() {
    const p = this.player, fx = -Math.sin(p.yaw), fz = -Math.cos(p.yaw);
    const x = p.pos.x + fx * 2.2, z = p.pos.z + fz * 2.2;
    if (!P.walkable(p.L, P.ci(x), P.ck(z))) return;
    const kind = this.rnd() < 0.5 ? 'ammo' : 'aid';
    const mesh = this.crates[kind].clone(); mesh.rotation.y = this.rnd() * 6.28;
    mesh.position.set(x, p.pos.y, z); this.scene.add(mesh);
    this.items.push({ kind, x, z, L: p.L, mesh, taken: false });
    this.msg('SUPPLIES');
  }
  toggleDoor() {
    const p = this.player, rnd = this.rnd;
    const closed = P.DOORS.filter((d) => !d.open).length;
    const cands = P.DOORS.map((d) => {
      const x = P.cx(d.i), z = P.cz(d.k), dist = Math.hypot(x - p.pos.x, z - p.pos.z) + Math.abs(d.L - p.L) * 8;
      if (dist < 5 || dist > 40) return [d, 0];
      const used = this.t - (this.doorUse.get(d.id) ?? -1e9) < 120 ? 4 : 1;
      return [d, used * (d.open ? (closed < 9 ? 1 : 0) : 1.2)];
    }).filter((q) => q[1] > 0);
    if (!cands.length) return;
    let u = rnd() * cands.reduce((s, q) => s + q[1], 0), pick = cands[0][0];
    for (const [d, w] of cands) { u -= w; if (u <= 0) { pick = d; break; } }
    pick.open = !pick.open;
    if (!pick.open && !this.connected()) { pick.open = true; return; }
    this.log.doors.push([this.t, pick.id, pick.open ? 1 : 0]);
    if (this.G.world) for (const rec of this.G.world.DOORS) if (rec.door === pick) rec.target = pick.open ? 1 : 0;
  }
  connected() {   // every room still reachable from the lobby
    const d = P.bfs(P.nodeId(0, 13, 9));
    return P.ROOMS.every((r) => { const [i0, k0, i1, k1] = r.rect; for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) if (P.walkable(r.L, i, k) && d.has(P.nodeId(r.L, i, k))) return true; return false; });
  }
  animateDoors(dt) {
    if (!this.G.world) return;
    for (const rec of this.G.world.DOORS) {
      const tgt = rec.door.open ? 1 : 0; rec.k = rec.k == null ? tgt : rec.k + (tgt - rec.k) * Math.min(1, dt * 5);
      for (const m of rec.meshes) { if (m.userData.y0 == null) m.userData.y0 = m.position.y; m.position.y = m.userData.y0 + rec.k * 2.2; m.visible = rec.k < 0.98; }
      for (const l of rec.lamps) l.material.color.set(rec.door.open ? 0x30ff60 : 0xff2a1a);
    }
  }
  // the person nearest the crosshair (within 6 deg), and the angle off
  aimTarget() {
    let best = null;
    for (const v of this.visible()) { if (!v[3]) continue; const e = Math.hypot(v[0], v[1]); if (e < 6 && (!best || e < best[1])) best = [this.actors.find((a) => a.id === v[4]), e]; }
    return best;
  }
  // a call: the first moment someone is plainly in front of you (in view, clear line, < 22 m).
  // It resolves as 'shoot' (you fired at them), or 'hold' after 2.5 s in view without firing.
  trackCalls() {
    const vis = this.visible(), now = new Set();
    for (const v of vis) {
      const [ax, ay, d, clear, id, hostile, light, pose, obj] = v;
      if (!clear || d > 22 || Math.abs(ax) > 30) continue;
      const a = this.actors.find((q) => q.id === id); now.add(id);
      if (a.callT == null) { a.callT = this.t; a.callVis = 0; this.log.calls.push([+this.t.toFixed(2), id, hostile, pose, +d.toFixed(1), light, a.group || 0, obj]); }
      if (!a.resolved) { a.callVis += 1; if (this.t - a.callT > 2.5) this.resolveCall(a, 'hold'); }
    }
    for (const a of this.actors) if (a.callT != null && !a.resolved && !now.has(a.id) && this.t - a.callT > 2.5) this.resolveCall(a, a.alive ? 'hold' : 'gone');
  }
  resolveCall(a, how) {
    if (a.callT == null || a.resolved) return;
    a.resolved = how; this.log.resp.push([+this.t.toFixed(2), a.id, how, +(this.t - a.callT).toFixed(2), a.hostile ? 1 : 0]);
  }
  // people in view: [screen x (deg), screen y (deg), distance, clear line, id, hostile, light, pose]
  visible() {
    const cam = this.camera, p = this.player, out = [];
    cam.updateMatrixWorld();
    const fovY = cam.fov * Math.PI / 180, tanY = Math.tan(fovY / 2), tanX = tanY * cam.aspect;
    const inv = cam.matrixWorldInverse;
    for (const a of this.actors) {
      if (!a.alive) continue;
      const h = new THREE.Vector3(a.pos.x, a.pos.y + 1.3, a.pos.z).applyMatrix4(inv);
      if (h.z > -0.3) continue;
      const sx = -h.x / h.z, sy = -h.y / h.z;
      if (Math.abs(sx) > tanX || Math.abs(sy) > tanY) continue;
      const d = Math.hypot(a.pos.x - p.pos.x, a.pos.z - p.pos.z);
      const clear = Math.abs(a.pos.y - p.pos.y) < 2.5 && los(p.L, p.pos.x, p.pos.z, a.pos.x, a.pos.z) ? 1 : 0;
      const ld = lightAt(a.L, a.pos.x, a.pos.y + 1.3, a.pos.z, 0, 1, 0), light = ld[0] * U.uPower.value + ld[1] + (p.torch && d < 20 ? 0.6 * Math.max(0, 1 - d / 20) : 0);
      out.push([Math.atan(sx) * 180 / Math.PI, Math.atan(sy) * 180 / Math.PI, d, clear, a.id, a.hostile ? 1 : 0, +light.toFixed(2), a.pose, a.objective ? 1 : 0, sx / tanX * 12, sy / tanY * 9, this.t - (a.muzzleT ?? -9) < 1.5 ? 1 : 0]);
    }
    return out;
  }
}
function rayVsSegment(o, d, a, b) {   // closest approach of the ray o + t d (d unit) to the segment ab
  const v = b.clone().sub(a), r = o.clone().sub(a);
  const e = v.dot(v), f = v.dot(r), c = d.dot(r), bb = d.dot(v);
  let t, s;
  if (e < 1e-9) { s = 0; t = -c; }
  else { const den = e - bb * bb; t = den > 1e-9 ? (bb * f - c * e) / den : 0; s = Math.max(0, Math.min(1, (bb * t + f) / e)); t = bb * s - c; }
  if (t <= 0) return null;
  const pt = o.clone().addScaledVector(d, t), q = a.clone().addScaledVector(v, s);
  return { t, dist: pt.distanceTo(q) };
}
function flashTex() {
  const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(255,255,230,1)'); gr.addColorStop(0.3, 'rgba(255,200,90,.8)'); gr.addColorStop(1, 'rgba(255,120,0,0)');
  g.fillStyle = gr; for (let n = 0; n < 6; n++) { g.save(); g.translate(32, 32); g.rotate(n * Math.PI / 3); g.beginPath(); g.moveTo(0, -3); g.lineTo(32, 0); g.lineTo(0, 3); g.fill(); g.restore(); }
  g.beginPath(); g.arc(32, 32, 20, 0, 7); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
