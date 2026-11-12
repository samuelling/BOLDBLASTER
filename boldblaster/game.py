"""Sandboxed first-person game: a vectorized numpy raycaster with textured walls,
floor/ceiling, wandering NPC sprites, 'tag' events with explosions + screen flashes,
a static weapon sprite and crosshair, a scripted player trajectory, and a
fixation/saccade gaze model.

Screen: 192 x 144 px = 24 x 18 deg of visual angle (8 px/deg), 10 Hz.
Rows run top->bottom, so screen y (deg, up positive) = (H/2 - row - 0.5)/PX_PER_DEG.
"""
import numpy as np
from collections import deque

W, H = 192, 144
PX_PER_DEG = 8.0
FPS = 10
FOV = np.deg2rad(70.0)          # in-game horizontal field of view
K = np.tan(FOV / 2)
SCALE = (W / 2) / K              # px per world unit at unit depth
TS = 64                          # texture size
SCREEN_HALF = (W / 2 / PX_PER_DEG, H / 2 / PX_PER_DEG)  # (12, 9) deg


# ----------------------------------------------------------------- textures
def _bandnoise(rng, f0, bw=0.35, size=TS):
    n = rng.standard_normal((size, size))
    fx = np.fft.fftfreq(size)[None, :]
    fy = np.fft.fftfreq(size)[:, None]
    fr = np.hypot(fx, fy)
    fr[0, 0] = 1e-6
    filt = np.exp(-(np.log(fr / f0)) ** 2 / (2 * bw ** 2))
    filt[0, 0] = 0
    out = np.real(np.fft.ifft2(np.fft.fft2(n) * filt))
    return out / out.std()


def make_textures(rng):
    yy, xx = np.mgrid[0:TS, 0:TS]
    tex = np.zeros((7, TS, TS), np.float32)
    brick = 0.6 + 0.06 * _bandnoise(rng, 1 / 4)
    mortar = (yy % 8 == 0) | (((xx + 8 * ((yy // 8) % 2)) % 16) == 0)
    brick[mortar] = 0.28
    tex[1] = brick                                           # mid SF, oriented
    tex[2] = 0.5 + 0.17 * _bandnoise(rng, 1 / 16)            # low SF blobs
    tex[3] = 0.5 + 0.17 * _bandnoise(rng, 1 / 5)             # high SF noise
    tex[4] = 0.55 + 0.12 * np.sign(np.sin(2 * np.pi * (xx + 0.5) / 16)) \
        + 0.05 * _bandnoise(rng, 1 / 3)                      # vertical panels
    tex[5] = 0.42 + 0.12 * (((xx // 32) + (yy // 32)) % 2) + 0.05 * _bandnoise(rng, 1 / 6)  # floor
    tex[6] = 0.68 + 0.05 * _bandnoise(rng, 1 / 20)           # ceiling
    return np.clip(tex, 0, 1)


def make_textures_v3(rng, sf_scale=1.0, mode='bandpass'):
    """Walls: contrast-equalized isotropic band-pass noise at 2, 4, 8, 16 cycles per
    wall unit (x sf_scale), so retinal SF depends on distance, not contrast.
    mode 'mixed' keeps two natural textures; 'natural' keeps all four."""
    tex = make_textures(rng)
    slots = {'bandpass': (1, 2, 3, 4), 'mixed': (3, 4), 'natural': ()}[mode]
    for k, f0 in zip((1, 2, 3, 4), (2, 4, 8, 16)):
        if k in slots:
            n = _bandnoise(rng, min(f0 * sf_scale, 28) / TS, bw=0.25)
            tex[k] = np.clip(0.5 + 0.2 * n, 0, 1)
    return tex


# ----------------------------------------------------------------- map
def make_map(rng, n=40, n_rooms=12, size=(4, 10)):
    m = np.ones((n, n), np.int8)
    rooms = []
    tries = 0
    while len(rooms) < n_rooms and tries < 1000:
        tries += 1
        w, h = rng.integers(size[0], size[1], size=2)
        x, y = rng.integers(1, n - w - 1), rng.integers(1, n - h - 1)
        if all(x + w + 1 < rx or rx + rw + 1 < x or y + h + 1 < ry or ry + rh + 1 < y
               for rx, ry, rw, rh in rooms):
            rooms.append((x, y, w, h))
            m[y:y + h, x:x + w] = 0
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in rooms]
    order = rng.permutation(len(centers))
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    pairs += [tuple(rng.choice(len(centers), 2, replace=False)) for _ in range(3)]
    for a, b in pairs:
        (x1, y1), (x2, y2) = centers[a], centers[b]
        if rng.random() < 0.5:
            m[y1:y1 + 2, min(x1, x2):max(x1, x2) + 2] = 0
            m[min(y1, y2):max(y1, y2) + 2, x2:x2 + 2] = 0
        else:
            m[min(y1, y2):max(y1, y2) + 2, x1:x1 + 2] = 0
            m[y2:y2 + 2, min(x1, x2):max(x1, x2) + 2] = 0
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1
    yy, xx = np.mgrid[0:n, 0:n]
    types = 1 + ((xx // 7 + 2 * (yy // 7)) % 4)
    m = np.where(m > 0, types, 0).astype(np.int8)
    return m, centers


def _bfs(m, start, goal):
    n = m.shape[0]
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        x, y = c
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < n and 0 <= ny < n and m[ny, nx] == 0 and (nx, ny) not in prev:
                prev[(nx, ny)] = c
                q.append((nx, ny))
    if goal not in prev:
        return [start]
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def _gsmooth(x, sigma_frames, axis=0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma_frames, axis=axis, mode='nearest')


def make_trajectory(rng, m, centers, duration, speed=1.8, look_deg=28.0):
    nf = int(duration * FPS)
    t = np.arange(nf) / FPS
    cur = centers[rng.integers(len(centers))]
    cells = [cur]
    while len(cells) < duration * speed * 1.5:
        goal = centers[rng.integers(len(centers))]
        if goal == cur:
            continue
        p = _bfs(m, cur, goal)
        cells += p[1:]
        cur = goal
    pts = np.array(cells, float) + 0.5
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    v = speed * (0.75 + 0.25 * np.sin(2 * np.pi * t / 11 + rng.uniform(0, 6.3)))
    stop = np.zeros(nf, bool)
    tt = rng.exponential(25)
    while tt < duration:
        d = rng.uniform(1.5, 3.0)
        stop[(t >= tt) & (t < tt + d)] = True
        tt += d + rng.exponential(25)
    v[stop] = 0
    s = np.cumsum(v) / FPS
    raw = np.c_[np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])]
    pos = _gsmooth(raw, 6)
    bad = m[pos[:, 1].astype(int), pos[:, 0].astype(int)] > 0
    pos[bad] = raw[bad]
    vel = np.gradient(_gsmooth(raw, 3), axis=0)
    ang = np.arctan2(vel[:, 1], vel[:, 0])
    moving = np.hypot(*vel.T) > 1e-4
    for i in range(1, nf):                       # hold heading while stopped
        if not moving[i]:
            ang[i] = ang[i - 1]
    ang = _gsmooth(np.unwrap(ang), 8)
    look = np.deg2rad(look_deg) * np.sin(2 * np.pi * t / 9 + rng.uniform(0, 6.3)) \
        * (0.5 + 0.5 * np.sin(2 * np.pi * t / 23 + rng.uniform(0, 6.3)))
    pan = _gsmooth(np.where(stop, np.deg2rad(45) * np.sin(2 * np.pi * t / 4), 0.0), 4)
    return pos, ang + look + pan, v


# ----------------------------------------------------------------- renderer
_COLS = np.arange(W)
_CAMX = 2 * (_COLS + 0.5) / W - 1
_ROWS = np.arange(H)[:, None]


def render_world(m, tex, px, py, ang, fog=9.0, cgain=None):
    n = m.shape[0]
    dx, dy = np.cos(ang), np.sin(ang)
    plx, ply = dy * K, -dx * K                 # camera plane: right column = right side
    rdx = dx + plx * _CAMX
    rdy = dy + ply * _CAMX
    rdx = np.where(np.abs(rdx) < 1e-9, 1e-9, rdx)
    rdy = np.where(np.abs(rdy) < 1e-9, 1e-9, rdy)
    mapx = np.full(W, int(np.floor(px)))
    mapy = np.full(W, int(np.floor(py)))
    ddx, ddy = np.abs(1 / rdx), np.abs(1 / rdy)
    stx = np.where(rdx < 0, -1, 1)
    sty = np.where(rdy < 0, -1, 1)
    sdx = np.where(rdx < 0, (px - mapx) * ddx, (mapx + 1 - px) * ddx)
    sdy = np.where(rdy < 0, (py - mapy) * ddy, (mapy + 1 - py) * ddy)
    hit = np.zeros(W, bool)
    side = np.zeros(W, np.int8)
    for _ in range(4 * n):
        nh = ~hit
        if not nh.any():
            break
        gx = nh & (sdx < sdy)
        gy = nh & ~(sdx < sdy)
        sdx[gx] += ddx[gx]; mapx[gx] += stx[gx]; side[gx] = 0
        sdy[gy] += ddy[gy]; mapy[gy] += sty[gy]; side[gy] = 1
        cx, cy = np.clip(mapx, 0, n - 1), np.clip(mapy, 0, n - 1)
        hit |= nh & (m[cy, cx] > 0)
    perp = np.maximum(np.where(side == 0, sdx - ddx, sdy - ddy), 1e-3)
    wt = m[np.clip(mapy, 0, n - 1), np.clip(mapx, 0, n - 1)]
    wallx = np.where(side == 0, py + perp * rdy, px + perp * rdx)
    wallx = wallx - np.floor(wallx)
    tu = (wallx * TS).astype(int) % TS
    lh = SCALE / perp
    tv = (_ROWS + 0.5 - (H / 2 - lh / 2)) / lh                 # (H, W)
    wall = (tv >= 0) & (tv < 1)
    tvi = np.clip((tv * TS).astype(int), 0, TS - 1)
    img = tex[wt[None, :], tvi, tu[None, :]]
    if cgain is not None:                      # v3: random contrast per wall cell
        g = cgain[np.clip(mapy, 0, n - 1), np.clip(mapx, 0, n - 1)]
        img = 0.5 + (img - 0.5) * g[None, :]
    img = np.where(side[None, :] == 1, img * 0.85, img)
    dist = np.broadcast_to(perp[None, :], (H, W)).copy()
    # floor / ceiling
    yoff = np.abs(_ROWS + 0.5 - H / 2)
    rowd = (SCALE * 0.5) / yoff                                 # (H,1)
    fx = px + rowd * rdx[None, :]
    fy = py + rowd * rdy[None, :]
    fu = ((fx - np.floor(fx)) * TS).astype(int) % TS
    fv = ((fy - np.floor(fy)) * TS).astype(int) % TS
    lower = (_ROWS + 0.5 > H / 2)
    fc = np.where(lower, tex[5][fv, fu], tex[6][fv, fu])
    img = np.where(wall, img, fc)
    dist = np.where(wall, dist, np.broadcast_to(rowd, (H, W)))
    img = 0.5 + (img - 0.5) * np.exp(-dist / fog)
    return img.astype(np.float32), perp, (dx, dy, plx, ply), wt, wall


def project(px, py, cam, sx, sy):
    dx, dy, plx, ply = cam
    rx, ry = sx - px, sy - py
    inv = 1.0 / (plx * dy - dx * ply)
    tx = inv * (dy * rx - dx * ry)
    ty = inv * (-ply * rx + plx * ry)
    return tx, ty


def _bbox(cx, cy, hw, hh):
    x0, x1 = int(max(0, np.floor(cx - hw))), int(min(W, np.ceil(cx + hw)))
    y0, y1 = int(max(0, np.floor(cy - hh))), int(min(H, np.ceil(cy + hh)))
    return x0, x1, y0, y1


def draw_npc(img, zbuf, tx, ty, catbuf=None):
    """Returns screen-deg position of the NPC if any part is visible, else None.
    catbuf, if given, gets a matching 1.0 written wherever the NPC is drawn (content-map
    channel for the ventral/temporal population)."""
    if ty < 0.2:
        return None
    scrx = (W / 2) * (1 + tx / ty)
    hpx, wpx = SCALE * 0.75 / ty, SCALE * 0.42 / ty
    top = H / 2 + SCALE * (0.5 - 0.75) / ty
    x0, x1, y0, y1 = _bbox(scrx, top + hpx / 2, wpx / 2, hpx / 2)
    if x0 >= x1 or y0 >= y1:
        return None
    cols, rows = np.arange(x0, x1), np.arange(y0, y1)
    U = ((cols + 0.5 - scrx) / wpx)[None, :]
    V = ((rows + 0.5 - top) / hpx)[:, None]
    head = (U / 0.22) ** 2 + ((V - 0.17) / 0.16) ** 2 < 1
    eyes = (((np.abs(U) - 0.08) / 0.04) ** 2 + ((V - 0.15) / 0.035) ** 2) < 1
    body = (U / 0.45) ** 2 + ((V - 0.62) / 0.36) ** 2 < 1
    belt = body & (V > 0.56) & (V < 0.63)
    lum = np.full(head.shape, np.nan, np.float32)
    lum[body] = 0.16
    lum[belt] = 0.88
    lum[head] = 0.9
    lum[head & eyes] = 0.08
    vis = (ty < zbuf[cols])[None, :] & ~np.isnan(lum)
    if not vis.any():
        return None
    sub = img[y0:y1, x0:x1]
    sub[vis] = lum[vis]
    if catbuf is not None:
        csub = catbuf[y0:y1, x0:x1]
        csub[vis] = 1.0
    ccol = int(np.clip(scrx, 0, W - 1))
    center_visible = (0 <= scrx < W) and ty < zbuf[ccol]
    return ((scrx - W / 2) / PX_PER_DEG, (H / 2 - (top + 0.45 * hpx)) / PX_PER_DEG,
            ty, center_visible)


def draw_explosion(img, zbuf, tx, ty, age, rng, life=0.8):
    if ty < 0.2:
        return None
    fade = max(0.0, 1 - age / life)
    r = 0.15 + 0.9 * age / life
    scrx = (W / 2) * (1 + tx / ty)
    scry = H / 2 + SCALE * (0.5 - 0.35) / ty
    rpx = SCALE * r / ty
    x0, x1, y0, y1 = _bbox(scrx, scry, rpx, rpx)
    if x0 >= x1 or y0 >= y1:
        return None
    cols, rows = np.arange(x0, x1), np.arange(y0, y1)
    D = np.hypot((cols + 0.5 - scrx)[None, :], (rows + 0.5 - scry)[:, None]) / rpx
    inside = (D < 1) & (ty < zbuf[cols])[None, :]
    core = D < 0.45
    spark = rng.random(D.shape) > 0.5
    lum = np.where(core, 0.5 + 0.5 * fade, 0.5 + 0.48 * fade * np.where(spark, 1, -1))
    sub = img[y0:y1, x0:x1]
    sub[inside] = lum[inside]
    return ((scrx - W / 2) / PX_PER_DEG, (H / 2 - scry) / PX_PER_DEG)


def draw_hud(img, bob):
    c = W // 2
    img[H // 2, c - 4:c + 5] = 0.95
    img[H // 2 - 4:H // 2 + 5, c] = 0.95
    b = int(round(bob))
    for r in range(112 + b, H):
        half = 6 + (r - 112 - b) * 0.45
        x0, x1 = int(c + 22 - half), int(c + 22 + half)
        img[r, x0:x1] = 0.22
        img[r, x0:x0 + 2] = 0.7


# ----------------------------------------------------------------- NPCs
class NPCs:
    def __init__(self, rng, m, n=8):
        self.rng, self.m = rng, m
        self.open = np.argwhere(m == 0)[:, ::-1] + 0.5
        self.pos = self.open[rng.integers(len(self.open), size=n)].copy()
        self.dir = rng.uniform(0, 2 * np.pi, n)
        self.alive = np.ones(n, bool)
        self.timer = np.zeros(n)

    def step(self, dt, player):
        rng = self.rng
        for i in range(len(self.pos)):
            if not self.alive[i]:
                self.timer[i] -= dt
                if self.timer[i] <= 0:
                    for _ in range(50):
                        p = self.open[rng.integers(len(self.open))]
                        if np.hypot(*(p - player)) > 5:
                            break
                    self.pos[i] = p
                    self.alive[i] = True
                continue
            if rng.random() < 0.02:
                self.dir[i] = rng.uniform(0, 2 * np.pi)
            nxt = self.pos[i] + 0.5 * dt * np.array([np.cos(self.dir[i]), np.sin(self.dir[i])])
            if self.m[int(nxt[1]), int(nxt[0])] == 0:
                self.pos[i] = nxt
            else:
                self.dir[i] = rng.uniform(0, 2 * np.pi)

    def kill(self, i):
        self.alive[i] = False
        self.timer[i] = self.rng.uniform(3, 6)


# ----------------------------------------------------------------- coins (v2)
class Coins:
    """Clusters of 1-8 floor coins: a designed numerosity lever. Collected (and
    respawned elsewhere) when the player walks into a cluster."""

    def __init__(self, rng, m, n_clusters=28, max_k=8):
        self.rng, self.max_k = rng, max_k
        self.open = np.argwhere(m == 0)[:, ::-1] + 0.5
        self.clusters = [self._new() for _ in range(n_clusters)]

    def _new(self):
        rng = self.rng
        c = self.open[rng.integers(len(self.open))]
        k = int(rng.integers(1, self.max_k + 1))
        ang = rng.uniform(0, 2 * np.pi, k)
        rad = 0.45 * np.sqrt(rng.uniform(0, 1, k))
        return dict(c=c, xy=c + np.c_[rad * np.cos(ang), rad * np.sin(ang)], alive=True, t=0.0)

    def step(self, dt, player):
        for i, cl in enumerate(self.clusters):
            if cl['alive'] and np.hypot(*(cl['c'] - player)) < 0.6:
                cl['alive'], cl['t'] = False, self.rng.uniform(5, 10)
            elif not cl['alive']:
                cl['t'] -= dt
                if cl['t'] <= 0:
                    self.clusters[i] = self._new()


def draw_coin(img, zbuf, tx, ty):
    if ty < 0.2:
        return None
    scrx = (W / 2) * (1 + tx / ty)
    scry = H / 2 + SCALE * (0.5 - 0.12) / ty
    rpx = max(SCALE * 0.07 / ty, 0.6)
    x0, x1, y0, y1 = _bbox(scrx, scry, rpx + 1, rpx + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    cols, rows = np.arange(x0, x1), np.arange(y0, y1)
    D = np.hypot((cols + 0.5 - scrx)[None, :], (rows + 0.5 - scry)[:, None])
    vis = (ty < zbuf[cols])[None, :]
    disk = (D < rpx) & vis
    ring = (D >= rpx) & (D < rpx + 0.8) & vis
    if not disk.any():
        return None
    sub = img[y0:y1, x0:x1]
    sub[ring] = 0.12
    sub[disk] = 0.95
    return ((scrx - W / 2) / PX_PER_DEG, (H / 2 - scry) / PX_PER_DEG)


# ----------------------------------------------------------------- gaze
class GazeModel:
    """Fixation/saccade sampler at frame rate. Targets: crosshair (most often),
    visible NPCs, explosion onsets, and broad 'salient' scene locations."""

    def __init__(self, rng):
        self.rng = rng
        self.target = np.zeros(2)
        self.left = 0
        self.holding = 0

    def hold(self, n):
        """Hold gaze on the crosshair for n frames (covert-attention episode)."""
        self.target = self.rng.normal(0, 0.3, 2)
        self.left = self.holding = n

    def step(self, npcs_vis, onset, coins=None):
        rng = self.rng
        if self.holding > 0:
            self.holding -= 1
            self.left -= 1
            g = self.target + rng.normal(0, 0.08, 2)
            return np.clip(g, [-SCREEN_HALF[0] + 0.3, -SCREEN_HALF[1] + 0.3],
                           [SCREEN_HALF[0] - 0.3, SCREEN_HALF[1] - 0.3])
        if onset is not None and rng.random() < 0.85:
            self._set(np.array(onset[:2]) + rng.normal(0, 0.3, 2))
        elif self.left <= 0:
            r = rng.random()
            vis = [v for v in npcs_vis if v[3]]
            if r < 0.5:
                tgt = rng.normal(0, 0.6, 2)
            elif r < 0.78 and vis:
                v = min(vis, key=lambda v: abs(v[0]))
                tgt = np.array(v[:2]) + rng.normal(0, 0.4, 2)
            elif r < 0.88 and coins:
                tgt = np.array(coins[rng.integers(len(coins))]) + rng.normal(0, 0.3, 2)
            else:
                tgt = np.array([rng.normal(0, 5.0), rng.normal(-0.5, 3.5)])
            self._set(tgt)
        self.left -= 1
        g = self.target + self.rng.normal(0, 0.08, 2)
        return np.clip(g, [-SCREEN_HALF[0] + 0.3, -SCREEN_HALF[1] + 0.3],
                       [SCREEN_HALF[0] - 0.3, SCREEN_HALF[1] - 0.3])

    def _set(self, tgt):
        self.target = tgt
        self.left = max(1, int(round(self.rng.gamma(2.5, 0.13) * FPS)))


# ----------------------------------------------------------------- run
def simulate_run(seed, duration=300.0, frame_sink=None, keep_frames=0):
    """Play one run. frame_sink(chunk) is called with (n,H,W) float32 frames.
    Returns dict with gaze (true, screen deg), events, and optional kept frames."""
    rng = np.random.default_rng(seed)
    tex = make_textures(rng)
    m, centers = make_map(rng)
    pos, ang, speed = make_trajectory(rng, m, centers, duration)
    npcs = NPCs(rng, m)
    gm = GazeModel(rng)
    nf = len(pos)
    gaze = np.zeros((nf, 2), np.float32)
    events, explosions, kept = [], [], []
    buf = []
    for f in range(nf):
        px, py = pos[f]
        img, zbuf, cam, _wt, _wall = render_world(m, tex, px, py, ang[f])
        npcs.step(1 / FPS, pos[f])
        vis = []
        for i in np.where(npcs.alive)[0]:
            tx, ty = project(px, py, cam, *npcs.pos[i])
            r = draw_npc(img, zbuf, tx, ty)
            if r is not None:
                vis.append(r + (i,))
        flash = 0.0
        for e in explosions:
            age = (f - e['f0']) / FPS
            if age <= 0.8:
                tx, ty = project(px, py, cam, *e['xy'])
                draw_explosion(img, zbuf, tx, ty, age, rng)
                if age < 0.3:
                    flash = max(flash, 0.45 * (1 - age / 0.3))
        explosions = [e for e in explosions if (f - e['f0']) / FPS <= 0.8]
        onset = None
        for v in vis:
            if v[3] and abs(v[0]) < 2.5 and abs(v[1]) < 4 and v[2] < 9 and rng.random() < 0.3:
                npcs.kill(v[4])
                explosions.append(dict(f0=f, xy=npcs.pos[v[4]].copy()))
                events.append((f / FPS, v[0], v[1]))
                onset = v
                flash = 0.45
                break
        draw_hud(img, 1.5 * np.sin(2 * np.pi * f / FPS * 1.6) * (speed[f] > 0))
        if flash > 0:
            img = img + flash * (1 - img)
        gaze[f] = gm.step(vis, onset)
        if f < keep_frames:
            kept.append(img.copy())
        buf.append(img)
        if len(buf) == 50 or f == nf - 1:
            if frame_sink is not None:
                frame_sink(np.stack(buf))
            buf = []
    return dict(gaze=gaze, events=np.array(events).reshape(-1, 3), frames=np.array(kept),
                map=m, pos=pos, ang=ang)


def simulate_run_v2(seed, duration=300.0, frame_sink=None, keep_frames=0, levers=False, theme=None,
                    content_sink=None):
    """v2 run: v1 game plus coin clusters, probabilistic shots, and per-frame logs.
    Logs: n_coins (visible coins), n_npcs, fwd (cells/s), yaw (rad/s),
    salient onsets (t, x, y, kind: 0 NPC appears, 1 explosion),
    shots (t, p_hit, outcome, x, y), events (hits = arousal events)."""
    rng = np.random.default_rng(seed)
    th = theme or {}
    if th:
        tex = make_textures_v3(rng, th['sf_scale'], th['tex'])
        m, centers = make_map(rng, n_rooms=th['rooms'][0], size=th['rooms'][1:])
        pos, ang, speed = make_trajectory(rng, m, centers, duration, speed=th['speed'], look_deg=th['look'])
    else:
        tex = make_textures_v3(rng) if levers else make_textures(rng)
        m, centers = make_map(rng)
        pos, ang, speed = make_trajectory(rng, m, centers, duration)
    cgain = rng.uniform(0.25, 1.6, m.shape) if levers else None
    fog = th.get('fog', 25.0 if levers else 9.0)
    light = th.get('light', 1.0)
    cue_rng = th.get('cue', (6, 10))
    shot_p = th.get('shot_p', 0.4)
    flash_amp = th.get('flash', 0.45)
    next_cue = rng.uniform(4, 10) if levers else 1e9
    cue = None
    npcs = NPCs(rng, m, n=th.get('npcs', 12))
    coins = Coins(rng, m, *th.get('coins', (28, 8)))
    gm = GazeModel(rng)
    nf = len(pos)
    gaze = np.zeros((nf, 2), np.float32)
    n_coins = np.zeros(nf, np.int16)
    n_npcs = np.zeros(nf, np.int16)
    events, salient, shots, explosions, kept, buf, cbuf = [], [], [], [], [], [], []
    last_seen = np.full(len(npcs.pos), -1e9)
    cooldown = 0.0
    lower_rows = (_ROWS + 0.5 > H / 2)      # (H,1), frame-independent
    for f in range(nf):
        t = f / FPS
        px, py = pos[f]
        img, zbuf, cam, wt, wall = render_world(m, tex, px, py, ang[f], fog=fog, cgain=cgain)
        cat = None
        if content_sink is not None:
            # content-map channels for the ventral/temporal population: NPC presence (filled
            # below, via draw_npc's catbuf), 4 wall-texture identities, floor, ceiling.
            cat = np.zeros((7, H, W), np.float32)
            for k in range(1, 5):
                cat[k] = wall & (wt[None, :] == k)
            cat[5] = (~wall) & lower_rows
            cat[6] = (~wall) & ~lower_rows
        npcs.step(1 / FPS, pos[f])
        coins.step(1 / FPS, pos[f])
        if t >= next_cue and cue is None:          # covert-attention episode
            th = rng.uniform(0, 2 * np.pi)
            ecc = rng.uniform(4, 9)
            cx, cy = ecc * np.cos(th), np.clip(ecc * np.sin(th), -8, 8)
            cue = dict(t0=t, th=th)
            salient.append((t, cx, cy, 2))
            gm.hold(int(2.0 * FPS))
            next_cue = t + 2.0 + rng.uniform(*cue_rng)
        if cue is not None and t - cue['t0'] >= 2.0:
            cue = None
        coin_pos = []
        for cl in coins.clusters:
            if not cl['alive']:
                continue
            for cx, cy in cl['xy']:
                tx, ty = project(px, py, cam, cx, cy)
                r = draw_coin(img, zbuf, tx, ty)
                if r is not None and abs(r[0]) < SCREEN_HALF[0] and abs(r[1]) < SCREEN_HALF[1]:
                    coin_pos.append(r)
        n_coins[f] = len(coin_pos)
        vis = []
        for i in np.where(npcs.alive)[0]:
            tx, ty = project(px, py, cam, *npcs.pos[i])
            r = draw_npc(img, zbuf, tx, ty, catbuf=(cat[0] if cat is not None else None))
            if r is not None:
                vis.append(r + (i,))
                if r[3]:
                    if t - last_seen[i] > 1.0:
                        salient.append((t, r[0], r[1], 0))
                    last_seen[i] = t
        n_npcs[f] = sum(1 for v in vis if v[3])
        flash = 0.0
        for e in explosions:
            age = (f - e['f0']) / FPS
            if age <= 0.8:
                tx, ty = project(px, py, cam, *e['xy'])
                draw_explosion(img, zbuf, tx, ty, age, rng)
                if age < 0.3:
                    flash = max(flash, flash_amp * (1 - age / 0.3))
        explosions = [e for e in explosions if (f - e['f0']) / FPS <= 0.8]
        onset = None
        cooldown -= 1 / FPS
        for v in vis:
            if cooldown <= 0 and v[3] and abs(v[0]) < 4 and abs(v[1]) < 5 and v[2] < 13 \
                    and rng.random() < shot_p:
                p = float(np.clip(1.05 - v[2] / 13 - abs(v[0]) / 10, 0.1, 0.95))
                hit = rng.random() < p
                shots.append((t, p, float(hit), v[0], v[1]))
                cooldown = 1.0
                if hit:
                    npcs.kill(v[4])
                    explosions.append(dict(f0=f, xy=npcs.pos[v[4]].copy()))
                    events.append((t, v[0], v[1]))
                    salient.append((t, v[0], v[1], 1))
                    onset = v
                    flash = flash_amp
                break
        if light != 1.0:                           # level lighting scales world contrast and mean
            img = 0.5 * light + (img - 0.5) * light
        draw_hud(img, 1.5 * np.sin(2 * np.pi * f / FPS * 1.6) * (speed[f] > 0))
        if cue is not None:                        # small arrow next to the crosshair
            for rr in np.linspace(0.7, 1.6, 9):
                c_ = int(W / 2 + rr * np.cos(cue['th']) * PX_PER_DEG)
                r_ = int(H / 2 - rr * np.sin(cue['th']) * PX_PER_DEG)
                img[max(0, r_ - 1):r_ + 1, max(0, c_ - 1):c_ + 1] = 0.98
        if flash > 0:
            img = img + flash * (1 - img)
        gaze[f] = gm.step(vis, onset, coin_pos)
        if f < keep_frames:
            kept.append(img.copy())
        buf.append(img)
        if cat is not None:
            cbuf.append(cat)
        if len(buf) == 50 or f == nf - 1:
            if frame_sink is not None:
                frame_sink(np.stack(buf))
            buf = []
            if content_sink is not None and cbuf:
                content_sink(np.stack(cbuf))
                cbuf = []
    yaw = np.gradient(np.unwrap(ang)) * FPS
    return dict(gaze=gaze, events=np.array(events).reshape(-1, 3), frames=np.array(kept),
                n_coins=n_coins, n_npcs=n_npcs, fwd=speed.astype(np.float32),
                yaw=yaw.astype(np.float32), salient=np.array(salient).reshape(-1, 4),
                shots=np.array(shots).reshape(-1, 5))
