"""Round 9: navigation (medial temporal and parietal) models and The Palace's cognitive models,
driven by each game's own dynamics, with every parameter attributed to a candidate brain area.

Part A, voxel-level (like rounds 1-8): planted populations in candidate areas respond to logged
navigation variables, BOLD is simulated with area-specific noise, and each voxel's tuning is fitted
back from 30 or 60 min of play.
  grid_move  entorhinal: six-fold (hexadirectional) modulation by movement direction x speed (Doeller 2010)
  grid_gaze  entorhinal: the same code for the direction of eye movements (Nau 2018)
  facing     retrosplenial / superior parietal: von Mises tuning to facing direction
  goal       hippocampus: mix of path and Euclidean distance to the current goal (Howard 2014)
  boundary   subiculum / entorhinal: tuning to distance from the nearest boundary (BVC-like)
  evbound    hippocampus: response at event boundaries (goal reached, death, level or zone change)

Part B, subject-level (model-based fMRI): 40 simulated players per game with planted cognitive
parameters from The Palace / Mind Arcade play the game's recorded decision opportunities. Each model
makes behaviour and a latent signal that drives voxels in its candidate area. Parameters are fitted
from behaviour alone and from BOLD alone, and the two are compared.
  ddm        drift-diffusion for each engage decision: drift scale v, boundary a (pre-SMA / IPS accumulator)
  rl         Rescorla-Wagner value learning of each target type: learning rate alpha, temperature beta
             (ventral striatum RPE)
  novelty    familiarity of each place: encoding e and forgetting f (hippocampus novelty)

Stages: pose | navfit | cog | export | all
"""
import os
import sys
import json
import time

import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt

from boldblaster import game, fit, models as M, sim
import run_v2 as V2

D, p, log = V2.D, V2.p, V2.log
FPS, NF = game.FPS, V2.NF
NTR = 200
N_RUNS = 12
GAMES = ['raycaster', 'betamax', 'brickninja', 'sanctuary']
LABEL = dict(raycaster='Raycaster', betamax='Betamax', brickninja='Brick Ninja', sanctuary='Animal Sanctuary')
PRE = dict(raycaster='game', betamax='betamax', brickninja='brickninja', sanctuary='sanctuary')
RAW = dict(betamax='/home/claude/bbx/runs', brickninja='/home/claude/bbx/runs2d', sanctuary='/home/claude/bbx/runs2d')

# candidate areas, and the noise each area's voxels carry relative to cortex (medial temporal
# voxels lose signal to susceptibility near the sinuses and ear canals, so their noise is higher)
AREA = dict(grid_move=('Entorhinal cortex', 1.6), grid_gaze=('Entorhinal cortex', 1.6),
            facing=('Retrosplenial / superior parietal', 1.0), goal=('Hippocampus', 1.4),
            boundary=('Subiculum / entorhinal', 1.5), evbound=('Hippocampus', 1.4),
            ddm=('Pre-SMA / intraparietal sulcus', 1.0), rl=('Ventral striatum', 1.2),
            novelty=('Hippocampus', 1.4))
NAV = ['grid_move', 'grid_gaze', 'facing', 'goal', 'boundary', 'evbound']
NV = 150


def wrap(a):
    return np.angle(np.exp(1j * a))


def zs(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 0 else 1)


# ------------------------------------------------------------------ poses from each game
def _traj_goals(rng, m, centers, duration, speed=1.8, look_deg=28.0):
    """game.make_trajectory, line for line (same random draws), also returning each frame's goal
    room centre and the remaining path length to it along the BFS route."""
    nf = int(duration * FPS)
    t = np.arange(nf) / FPS
    cur = centers[rng.integers(len(centers))]
    cells, gidx = [cur], [0]
    goals = []
    while len(cells) < duration * speed * 1.5:
        goal = centers[rng.integers(len(centers))]
        if goal == cur:
            continue
        pth = game._bfs(m, cur, goal)
        goals.append(goal)
        cells += pth[1:]
        gidx += [len(goals) - 1] * (len(pth) - 1)
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
    gidx = np.array(gidx)
    k = np.clip(np.searchsorted(seg, s, side='right'), 1, len(seg) - 1)
    g_of = gidx[k]                                   # goal of the segment being walked
    g_end = np.array([seg[np.where(gidx == g)[0][-1]] for g in range(len(goals))])
    path_left = np.maximum(g_end[g_of] - s, 0)
    gxy = np.array(goals, float)[g_of] + 0.5
    return g_of, path_left, gxy


def pose_raycaster(i):
    rng = np.random.default_rng(300 + i)
    game.make_textures(rng)
    m, centers = game.make_map(rng)
    st = rng.bit_generator.state
    pos, ang, speed = game.make_trajectory(rng, m, centers, V2.DUR)
    rng.bit_generator.state = st
    g_of, path_left, gxy = _traj_goals(rng, m, centers, V2.DUR)
    up = 8
    free = np.kron((m == 0).astype(np.uint8), np.ones((up, up), np.uint8))
    edt = distance_transform_edt(free) / up
    bd = edt[np.clip((pos[:, 1] * up).astype(int), 0, edt.shape[0] - 1),
             np.clip((pos[:, 0] * up).astype(int), 0, edt.shape[1] - 1)]
    vel = np.gradient(pos, axis=0) * FPS
    reached = np.where(np.diff(g_of) != 0)[0] + 1
    return dict(pos=pos, vel=vel, face=ang, goal_e=np.hypot(*(gxy - pos).T), goal_p=path_left,
                has_goal=np.ones(NF, bool), bdist=bd, bounds=reached / FPS,
                state=(pos[:, 1].astype(int) * 64 + pos[:, 0].astype(int)), ctx=np.full(NF, 1000 + i),
                valid=np.ones(NF, bool))


def _recs(g, i):
    return [json.loads(l) for l in open(f'{RAW[g]}/{g}{i}_log.jsonl')]


def _invalid_around(frames, n=10):
    bad = np.zeros(NF, bool)
    for f in frames:
        bad[max(0, f - n):min(NF, f + n)] = True
    return bad


def pose_betamax(i):
    R = _recs('betamax', i)
    lg = np.load(p(f'betamax{i}_logs.npz'))
    pos = np.array([[r['x'], r['z']] for r in R], float)
    face = np.array([r['yaw'] for r in R], float)
    body = np.array([r['body'] for r in R])
    tau = np.array([r['tau'] for r in R])
    jumps = sorted(set(np.where((np.diff(body) != 0) | (np.abs(np.diff(tau) - 0.1) > 0.5)
                                | (np.hypot(*np.diff(pos, axis=0).T) > 3))[0] + 1))
    valid = ~_invalid_around(jumps)
    vel = np.gradient(pos, axis=0) * FPS
    vel[~valid] = 0
    # goal: the nearest visible enemy (the bot's target); open beach, so path = Euclidean
    ge = np.array([min([v[2] for v in r['vis'] if v[3] == 1 and v[5] == 1] or [np.nan]) for r in R])
    has = np.isfinite(ge)
    bounds = np.r_[lg['events'][:, 0], lg['deaths'], (np.array(jumps) / FPS)]
    return dict(pos=pos, vel=vel, face=-face, goal_e=np.nan_to_num(ge), goal_p=np.nan_to_num(ge), has_goal=has,
                bdist=None, bounds=np.unique(np.round(bounds, 1)),
                state=(np.floor(pos[:, 0] / 4).astype(int) * 1000 + np.floor(pos[:, 1] / 4).astype(int)),
                ctx=np.zeros(NF, int), valid=valid)


LV_END = {}


def pose_brickninja(i):
    R = _recs('brickninja', i)
    lg = np.load(p(f'brickninja{i}_logs.npz'))
    if not LV_END:
        for j in range(N_RUNS):
            for r in _recs('brickninja', j):
                LV_END[r['lv']] = max(LV_END.get(r['lv'], 0), r['x'])
    x = np.array([r['x'] for r in R], float) / 32.0            # tiles
    lv = np.array([r['lv'] for r in R])
    jumps = list(np.where((np.diff(lv) != 0) | (np.abs(np.diff(x)) > 6))[0] + 1)
    valid = ~_invalid_around(jumps)
    dx = np.gradient(x) * FPS
    dx[~valid] = 0
    face = np.zeros(NF)
    last = 0.0
    for f in range(NF):
        if abs(dx[f]) > 0.3:
            last = 0.0 if dx[f] > 0 else np.pi
        face[f] = last
    pos = np.c_[x, np.zeros(NF)]
    vel = np.c_[dx, np.zeros(NF)]
    end = np.array([LV_END[l] / 32.0 for l in lv])
    ge = np.maximum(end - x, 0)
    lvch = np.where(np.diff(lv) != 0)[0] + 1
    bounds = np.r_[lvch / FPS, lg['deaths']]
    return dict(pos=pos, vel=vel, face=face, goal_e=ge, goal_p=ge, has_goal=np.ones(NF, bool), bdist=None,
                bounds=np.unique(np.round(bounds, 1)), state=lv * 10000 + np.floor(x / 1.5).astype(int),
                ctx=np.zeros(NF, int), valid=valid)


ZONE_EXT = {}


def pose_sanctuary(i):
    R = _recs('sanctuary', i)
    lg = np.load(p(f'sanctuary{i}_logs.npz'))
    if not ZONE_EXT:                                # zone extents from every run (a proxy for its edges)
        for j in range(N_RUNS):
            for r in _recs('sanctuary', j):
                lo, hi = ZONE_EXT.get(r['zone'], ((1e9, 1e9), (-1e9, -1e9)))
                ZONE_EXT[r['zone']] = ((min(lo[0], r['x']), min(lo[1], r['y'])), (max(hi[0], r['x']), max(hi[1], r['y'])))
    pos = np.array([[r['x'], r['y']] for r in R], float) / 16.0   # tiles
    zone = [r['zone'] for r in R]
    zid = np.array([sorted(ZONE_EXT).index(z) for z in zone])
    jumps = list(np.where((np.diff(zid) != 0) | (np.hypot(*np.diff(pos, axis=0).T) > 4))[0] + 1)
    valid = ~_invalid_around(jumps)
    vel = np.gradient(pos, axis=0) * FPS
    vel[~valid] = 0
    sp = np.hypot(*vel.T)
    face = np.zeros(NF)
    last = 0.0
    for f in range(NF):
        if sp[f] > 0.3:
            last = np.arctan2(-vel[f, 1], vel[f, 0])
        face[f] = last
    lo = np.array([ZONE_EXT[z][0] for z in zone]) / 16.0 - 1
    hi = np.array([ZONE_EXT[z][1] for z in zone]) / 16.0 + 1
    bd = np.minimum(pos - lo, hi - pos).min(1)
    ge = np.array([min([v[2] for v in r['vis']] or [np.nan]) for r in R]) / 16.0
    has = np.isfinite(ge)
    bounds = np.r_[lg['events'][:, 0], np.array(jumps) / FPS]
    return dict(pos=pos, vel=vel, face=face, goal_e=np.nan_to_num(ge), goal_p=np.nan_to_num(ge), has_goal=has,
                bdist=bd, bounds=np.unique(np.round(bounds, 1)),
                state=zid * 10000 + np.floor(pos[:, 0] / 2).astype(int) * 100 + np.floor(pos[:, 1] / 2).astype(int),
                ctx=np.zeros(NF, int), valid=valid)


POSE = dict(raycaster=pose_raycaster, betamax=pose_betamax, brickninja=pose_brickninja, sanctuary=pose_sanctuary)


def nav_vars(g, i):
    """Every logged variable the navigation models read, one 5-min run, 10 Hz."""
    f = p(f'nav_{g}{i}.npz')
    if os.path.exists(f):
        d = dict(np.load(f, allow_pickle=True))
        d['bdist'] = None if d['bdist'].ndim == 0 else d['bdist']
        return d
    ps = POSE[g](i)
    lg = np.load(p(f'{PRE[g]}{i}_logs.npz'))
    sp = np.hypot(*ps['vel'].T)
    mdir = np.arctan2(ps['vel'][:, 1], ps['vel'][:, 0])
    out = dict(speed=sp, mdir=mdir, face=wrap(ps['face']), goal_e=ps['goal_e'], goal_p=ps['goal_p'],
               has_goal=ps['has_goal'], bdist=np.array(None) if ps['bdist'] is None else ps['bdist'],
               bounds=ps['bounds'], state=ps['state'], ctx=ps['ctx'], valid=ps['valid'],
               gaze=lg['gaze'], et=lg['et'], events=lg['events'], shots=lg['shots'])
    np.savez(f, **out)
    out['bdist'] = ps['bdist']
    return out


def gaze_motion(gz):
    v = np.gradient(gz.astype(float), axis=0) * FPS          # deg/s on screen
    return np.hypot(*v.T), np.arctan2(v[:, 1], v[:, 0])


# ------------------------------------------------------------------ Part A: planted populations
def nav_population():
    f = p('nav_population.npz')
    if os.path.exists(f):
        return dict(np.load(f))
    rng = np.random.default_rng(909)
    pop = {}
    for name in NAV:
        mult = AREA[name][1]
        pop[f'{name}_hrf_scale'] = np.clip(rng.normal(1, 0.08, NV), 0.8, 1.2)
        pop[f'{name}_amp'] = rng.uniform(0.4, 1.0, NV)       # medial temporal effects are small
        pop[f'{name}_noise_sd'] = rng.uniform(0.6, 1.2, NV) * mult
        pop[f'{name}_sys_amp'] = np.clip(rng.normal(0.5, 0.2, NV), 0.05, None)
    for name in ('grid_move', 'grid_gaze'):
        pop[f'{name}_phi'] = rng.uniform(0, np.pi / 3, NV)   # grid orientation (60 deg period)
        pop[f'{name}_m'] = rng.uniform(0.1, 0.5, NV)         # hexadirectional modulation depth (a minority of the speed response)
    pop['facing_psi'] = rng.uniform(-np.pi, np.pi, NV)
    pop['facing_kappa'] = np.exp(rng.uniform(np.log(1.0), np.log(6.0), NV))
    pop['goal_w'] = rng.uniform(0, 1, NV)                    # weight on path (vs Euclidean) distance
    pop['boundary_d'] = rng.uniform(0.05, 0.8, NV)           # preferred boundary distance (fraction of range)
    pop['evbound_b'] = rng.uniform(0, 1, NV)                 # event-boundary gain
    np.savez(f, **pop)
    return pop


def vox(pop, name):
    return dict(hrf_scale=pop[f'{name}_hrf_scale'], amp=pop[f'{name}_amp'], noise_sd=pop[f'{name}_noise_sd'],
                sys_amp=pop[f'{name}_sys_amp'], x=np.zeros(NV))


def sp_norm(s, valid):
    s = np.where(valid, s, 0)
    q = np.percentile(s[s > 0], 90) if np.any(s > 0) else 1
    return np.clip(s / q, 0, 1.5)


def grid_drive(spn, theta, phi, m):
    return spn[:, None] * (1 + m[None] * np.cos(6 * (theta[:, None] - phi[None])))


def vm(face, psi, kappa):
    return np.exp(kappa[None] * (np.cos(face[:, None] - psi[None]) - 1))


def bdist_norm(g):
    """Each game's boundary distance as a fraction of its own 95th percentile."""
    vals = np.concatenate([nav_vars(g, i)['bdist'] for i in range(N_RUNS)])
    return np.percentile(vals, 95)


def bvc(bdn, d, width=0.12):
    return np.exp(-(bdn[:, None] - d[None]) ** 2 / (2 * (width + 0.3 * d[None]) ** 2))


def goal_regs(v):
    hg = v['has_goal']
    gp, ge = np.where(hg, v['goal_p'], np.nan), np.where(hg, v['goal_e'], np.nan)
    return np.nan_to_num(zs(gp)) * hg, np.nan_to_num(zs(ge)) * hg


def bound_imp(v):
    x = np.zeros(NF)
    for t in v['bounds']:
        k = int(t * FPS)
        x[k:min(NF, k + 10)] = 1
    return x


def nav_truth(g, i, pop, bnorm):
    f = p(f'navY_{g}{i}.npz')
    if os.path.exists(f):
        return dict(np.load(f))
    v = nav_vars(g, i)
    rng = np.random.default_rng(5000 + 97 * GAMES.index(g) + i)
    spn = sp_norm(v['speed'], v['valid'])
    gsp, gdir = gaze_motion(v['gaze'])
    gspn = np.clip(gsp / np.percentile(gsp, 90), 0, 1.5)
    drives = dict(grid_move=grid_drive(spn, v['mdir'], pop['grid_move_phi'], pop['grid_move_m']),
                  grid_gaze=grid_drive(gspn, gdir, pop['grid_gaze_phi'], pop['grid_gaze_m']),
                  facing=vm(v['face'], pop['facing_psi'], pop['facing_kappa']))
    gp, ge = goal_regs(v)
    w = pop['goal_w']
    drives['goal'] = gp[:, None] * w[None] + ge[:, None] * (1 - w[None])
    if v['bdist'] is not None:
        drives['boundary'] = bvc(v['bdist'] / bnorm, pop['boundary_d'])
    drives['evbound'] = bound_imp(v)[:, None] * pop['evbound_b'][None]
    out = {}
    for name, d in drives.items():
        vis = M.bold_from_drive(d.astype(np.float32), pop[f'{name}_hrf_scale'], pop[f'{name}_amp'])
        out[name] = M.add_nuisance(rng, vis, vox(pop, name), v['events'])
    np.savez(f, **out)
    return out


def circ_corr(a, b):
    """Phase recovery: length of the mean resultant of the angular errors (1 = every voxel's
    preferred angle recovered exactly, 0 = errors uniform, i.e. chance)."""
    return float(np.abs(np.mean(np.exp(1j * (a - b)))))


def hrf_tr(x):
    """(NF,) or (NF,K) 10 Hz -> (NTR,K) HRF-convolved."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    return sim.to_tr(sim._conv(x, sim.hrf(1 / FPS), axis=0))


def resid_runs(Xs, Ys, Qs):
    X = np.concatenate([X - Q @ (Q.T @ X) for X, Q in zip(Xs, Qs)])
    Y = np.concatenate([Y - Q @ (Q.T @ Y) for Y, Q in zip(Ys, Qs)])
    return X, Y


def fit_nav(g, n, pop, bnorm, realistic=True):
    runs = []
    for i in range(n):
        v = nav_vars(g, i)
        Y = nav_truth(g, i, pop, bnorm)
        Q = fit.nuisance_basis(NTR, v['events'], arousal=True)
        runs.append((v, Y, Q))
    res = {}
    Qs = [Q for _, _, Q in runs]
    # grid codes: quadrature regression on speed x cos/sin(6 theta)
    for name in ('grid_move', 'grid_gaze'):
        Xs = []
        for v, Y, Q in runs:
            if name == 'grid_move':
                spn, th = sp_norm(v['speed'], v['valid']), v['mdir']
            else:
                gsp, th = gaze_motion(v['et'] if realistic else v['gaze'])
                spn = np.clip(gsp / np.percentile(gsp, 90), 0, 1.5)
            Xs.append(hrf_tr(np.c_[spn, spn * np.cos(6 * th), spn * np.sin(6 * th)]))
        X, Yc = resid_runs(Xs, [Y[name] for _, Y, _ in runs], Qs)
        B = np.linalg.lstsq(X, Yc, rcond=None)[0]
        phi_hat = np.mod(np.arctan2(B[2], B[1]) / 6, np.pi / 3)
        m_hat = np.hypot(B[1], B[2]) / np.maximum(np.abs(B[0]), 1e-6)
        res[name] = dict(phi=circ_corr(6 * pop[f'{name}_phi'], 6 * phi_hat),
                         phi_err_deg=float(np.degrees(np.median(np.abs(wrap(6 * (phi_hat - pop[f'{name}_phi'])) / 6)))),
                         m=float(spearmanr(pop[f'{name}_m'], m_hat).correlation),
                         dir_coverage=float(direction_coverage(runs, name, realistic)))
    # facing direction: grid search over preferred direction and width, linear amplitude
    psis = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    kaps = np.exp(np.linspace(np.log(1.0), np.log(6.0), 6))
    Yc = None
    best = np.full(NV, -np.inf)
    bpsi, bkap = np.zeros(NV), np.zeros(NV)
    for kp in kaps:
        Xs = [hrf_tr(vm(v['face'], psis, np.full(len(psis), kp))) for v, _, _ in runs]
        X, Yc = resid_runs(Xs, [Y['facing'] for _, Y, _ in runs], Qs)
        Xn = X / np.maximum(np.linalg.norm(X, axis=0), 1e-9)
        Yn = Yc / np.maximum(np.linalg.norm(Yc, axis=0), 1e-9)
        C = Xn.T @ Yn                                   # (36, NV) correlation
        j = np.argmax(C, 0)
        cmax = C[j, np.arange(NV)]
        upd = cmax > best
        best[upd], bpsi[upd], bkap[upd] = cmax[upd], psis[j[upd]], kp
    res['facing'] = dict(psi=circ_corr(pop['facing_psi'], bpsi),
                         psi_err_deg=float(np.degrees(np.median(np.abs(wrap(bpsi - pop['facing_psi']))))),
                         kappa=float(spearmanr(pop['facing_kappa'], bkap).correlation))
    # goal distance: two regressors
    Xs = [hrf_tr(np.c_[goal_regs(v)]) for v, _, _ in runs]
    X, Yc = resid_runs(Xs, [Y['goal'] for _, Y, _ in runs], Qs)
    rpe = np.corrcoef(X.T)[0, 1]
    B = np.linalg.lstsq(X, Yc, rcond=None)[0]
    w_hat = B[0] / np.maximum(np.abs(B[0]) + np.abs(B[1]), 1e-9)
    res['goal'] = dict(w=float(spearmanr(pop['goal_w'], w_hat).correlation) if abs(rpe) < 0.999 else np.nan,
                       path_eucl_corr=float(rpe),
                       goal_time=float(np.mean(np.concatenate([v['has_goal'] for v, _, _ in runs]))))
    # boundary: grid over preferred distance
    if runs[0][0]['bdist'] is not None:
        ds = np.linspace(0.05, 0.8, 16)
        Xs = [hrf_tr(bvc(v['bdist'] / bnorm, ds)) for v, _, _ in runs]
        X, Yc = resid_runs(Xs, [Y['boundary'] for _, Y, _ in runs], Qs)
        Xn = X / np.maximum(np.linalg.norm(X, axis=0), 1e-9)
        C = Xn.T @ (Yc / np.maximum(np.linalg.norm(Yc, axis=0), 1e-9))
        res['boundary'] = dict(d=float(spearmanr(pop['boundary_d'], ds[np.argmax(C, 0)]).correlation))
    else:
        res['boundary'] = dict(d=np.nan)
    # event boundaries: one regressor, gain
    Xs = [hrf_tr(bound_imp(v)) for v, _, _ in runs]
    X, Yc = resid_runs(Xs, [Y['evbound'] for _, Y, _ in runs], Qs)
    B = np.linalg.lstsq(X, Yc, rcond=None)[0][0]
    res['evbound'] = dict(b=float(spearmanr(pop['evbound_b'] * pop['evbound_amp'], B).correlation),
                          per_min=float(sum(len(v['bounds']) for v, _, _ in runs) / (5 * n)))
    return res


def direction_coverage(runs, name, realistic):
    """How evenly the runs sample direction in the 60-deg grid cycle: 1 - |mean resultant| of 6 theta,
    weighted by speed (1 = perfectly balanced, 0 = one direction only)."""
    num, den = 0j, 0.0
    for v, _, _ in runs:
        if name == 'grid_move':
            w, th = sp_norm(v['speed'], v['valid']), v['mdir']
        else:
            w, th = gaze_motion(v['et'] if realistic else v['gaze'])
        num += np.sum(w * np.exp(6j * th))
        den += np.sum(w)
    return 1 - abs(num) / max(den, 1e-9)


def navfit():
    pop = nav_population()
    out = {}
    for g in GAMES:
        bn = bdist_norm(g) if nav_vars(g, 0)['bdist'] is not None else 1.0
        for mins, n in ((30, 6), (60, 12)):
            out[f'{g}_{mins}'] = fit_nav(g, n, pop, bn, realistic=True)
            out[f'{g}_{mins}_oraclegaze'] = dict(grid_gaze=fit_nav(g, n, pop, bn, realistic=False)['grid_gaze'])
            log(g, mins, json.dumps({k: {a: round(b, 2) for a, b in d.items()} for k, d in out[f'{g}_{mins}'].items()}))
    json.dump(out, open(p('results_round9_nav.json'), 'w'), indent=1)


# ------------------------------------------------------------------ Part B: cognitive models
NS = 40
GAP = 60.0          # seconds between 5-min runs (cross-run forgetting for the novelty model)


def opportunities(g, n):
    """Decision opportunities from the game's own logs: time (s, session clock), expected hit
    probability, outcome. Each is one engage-or-hold decision."""
    rows = []
    for i in range(n):
        sh = np.load(p(f'{PRE[g]}{i}_logs.npz'))['shots']
        for t, ph, o, *_ in sh:
            rows.append((i, t, ph, o))
    return np.array(rows, float)


def ddm_sim(rng, v, a, t0, dt=0.005, tmax=4.0):
    """Unbiased two-boundary diffusion (sigma 1). v: drifts (n,). Returns rt, correct."""
    n = len(v)
    x = np.zeros(n)
    rt = np.full(n, tmax)
    done = np.zeros(n, bool)
    up = np.zeros(n, bool)
    for k in range(int(tmax / dt)):
        live = ~done
        if not live.any():
            break
        x[live] += v[live] * dt + np.sqrt(dt) * rng.standard_normal(live.sum())
        hit = live & (np.abs(x) >= a / 2)
        rt[hit] = (k + 1) * dt
        up[hit] = x[hit] > 0
        done |= hit
    up[~done] = x[~done] > 0
    return rt + t0, up


def ez(pc, vrt, mrt, n):
    """EZ-diffusion (Wagenmakers 2007), s = 1."""
    pc = np.clip(pc, 1 / (2 * n), 1 - 1 / (2 * n))
    if abs(pc - 0.5) < 1e-3:
        pc = 0.5 + 1e-3
    L = np.log(pc / (1 - pc))
    xx = L * (L * pc ** 2 - L * pc + pc - 0.5) / max(vrt, 1e-6)
    v = np.sign(pc - 0.5) * xx ** 0.25
    a = L / v
    y = -v * a
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    return v, a, mrt - mdt


def edt_ddm(v, a):
    """Expected decision time, unbiased start, sigma 1."""
    v = np.maximum(np.abs(v), 1e-4)
    return (a / (2 * v)) * np.tanh(a * v / 2)


def strength(ph):
    return np.clip(2 * ph - 1, -1, 1)


def region_bold(rng, drive_runs, n_vox, mult, events_runs):
    """Region of n_vox voxels, each a gain on the shared drive; returns region-mean BOLD per run."""
    gains = rng.uniform(0.5, 1.5, n_vox)
    vx = dict(hrf_scale=np.ones(n_vox), amp=np.ones(n_vox), noise_sd=rng.uniform(0.6, 1.2, n_vox) * mult,
              sys_amp=np.clip(rng.normal(0.5, 0.2, n_vox), 0.05, None), x=np.zeros(n_vox))
    out = []
    for d, ev in zip(drive_runs, events_runs):
        vis = hrf_tr(d) * gains[None] * 0.6
        Yv = M.add_nuisance(rng, vis.astype(np.float32), vx, ev).mean(1)
        # noise shared by neighbouring voxels (physiological, motion residue): AR(1), half the
        # per-voxel SD, so averaging a region cannot beat it down
        e = rng.normal(0, 1, NTR)
        for k in range(1, NTR):
            e[k] = 0.3 * e[k - 1] + np.sqrt(1 - 0.09) * e[k]
        out.append(Yv + 0.5 * mult * 0.9 * e)
    return out


def frames_of(times, n=10, val=None):
    x = np.zeros(NF)
    for k, t in enumerate(times):
        i = int(t * FPS)
        if i < NF:
            x[i:min(NF, i + n)] += 1 if val is None else val[k]
    return x


def boxcars(times, durs):
    x = np.zeros(NF)
    for t, d in zip(times, durs):
        i = int(t * FPS)
        x[i:min(NF, i + max(1, int(round(d * FPS))))] += 1
    return x


def best_by_r2(pred_runs_list, Yruns, Qs):
    """pred_runs_list: candidates, each a list over runs of (NTR,K). Returns argmax R^2 index."""
    Yc = np.concatenate([y - Q @ (Q.T @ y) for y, Q in zip(Yruns, Qs)])
    best, bi = -np.inf, 0
    for c, pr in enumerate(pred_runs_list):
        X = np.concatenate([x - Q @ (Q.T @ x) for x, Q in zip(pr, Qs)])
        b = np.linalg.lstsq(X, Yc, rcond=None)[0]
        r2 = 1 - np.sum((Yc - X @ b) ** 2) / np.sum(Yc ** 2)
        if r2 > best:
            best, bi = r2, c
    return bi


def cog_game(g, n):
    rng = np.random.default_rng(31 + GAMES.index(g) * 7 + n)
    op = opportunities(g, n)
    ev_runs = [np.load(p(f'{PRE[g]}{i}_logs.npz'))['events'] for i in range(n)]
    Qs = [fit.nuisance_basis(NTR, e, arousal=True) for e in ev_runs]
    s = strength(op[:, 2])
    corr_choice = s > 0                                  # engaging is right when a hit is more likely than not
    nav = [nav_vars(g, i) for i in range(n)]
    true = {k: [] for k in ('v', 'a', 't0', 'alpha', 'beta', 'e', 'f')}
    est = {k: [] for k in ('v_beh', 'a_beh', 'va_beh', 'v_bold', 'a_bold', 'va_bold', 'alpha_beh', 'beta_beh',
                           'alpha_bold', 'e_bold', 'f_bold')}
    # candidate grids for BOLD-only fits
    vg = np.linspace(0.8, 4.2, 12)
    ag = np.linspace(0.7, 2.1, 12)
    alg = np.linspace(0.03, 0.65, 16)
    eg = np.linspace(0.15, 0.85, 8)
    fg = np.exp(np.linspace(np.log(0.01), np.log(0.8), 10))       # per minute
    # novelty state visits, pre-computed: entries into a new state (time, state key)
    entries = []
    for i, v in enumerate(nav):
        stt = v['state'].astype(np.int64) + v['ctx'].astype(np.int64) * 10 ** 7
        ch = np.r_[0, np.where(np.diff(stt) != 0)[0] + 1]
        ch = ch[v['valid'][ch]]
        entries.append((ch / FPS, stt[ch]))
    ddm_cands, ddm_keys = [], []
    for vv in vg:
        for aa in ag:
            dtp = edt_ddm(vv * np.abs(s), aa)
            ddm_cands.append([hrf_tr(boxcars(op[op[:, 0] == i, 1], dtp[op[:, 0] == i])) for i in range(n)])
            ddm_keys.append((vv, aa))

    def novelty(e, f):
        F, last = {}, {}
        out = []
        for i, (tt, st) in enumerate(entries):
            T = tt + i * (V2.DUR + GAP)
            nv = np.zeros(len(tt))
            for k in range(len(tt)):
                key = st[k]
                if key in F:
                    F[key] *= np.exp(-f * (T[k] - last[key]) / 60)
                fam = F.get(key, 0.0)
                nv[k] = 1 - fam
                F[key] = fam + e * (1 - fam)
                last[key] = T[k]
            out.append(frames_of(tt, 10, nv))
        return out
    nov_cands, nov_keys = [], []
    for ee in eg:
        for ff in fg:
            nov_cands.append([hrf_tr(d) for d in novelty(ee, ff)])
            nov_keys.append((ee, ff))
    for sj in range(NS):
        v_, a_, t0_ = rng.uniform(1.0, 4.0), rng.uniform(0.8, 2.0), rng.uniform(0.15, 0.4)
        al_, be_ = rng.uniform(0.05, 0.6), rng.uniform(2, 12)
        e_, f_ = rng.uniform(0.2, 0.8), np.exp(rng.uniform(np.log(0.02), np.log(0.5)))
        for k, x in zip(true, (v_, a_, t0_, al_, be_, e_, f_)):
            true[k].append(x)
        # ---- DDM: behaviour and accumulator BOLD
        rt, up = ddm_sim(rng, v_ * np.abs(s) + 1e-3, a_, t0_)
        dt_true = rt - t0_
        # behaviour as recorded: motor jitter on every response, and 4% lapses (a guess at a
        # random time) that the model does not know about
        rt = rt + rng.normal(0, 0.06, len(rt))
        lapse = rng.random(len(rt)) < 0.04
        rt[lapse] = rng.uniform(0.3, 2.0, lapse.sum())
        up[lapse] = rng.random(lapse.sum()) < 0.5
        correct = up                                       # upper boundary = the correct choice
        bins = np.digitize(np.abs(s), np.quantile(np.abs(s), [1 / 3, 2 / 3]))
        vb, ab, sb = [], [], []
        for b in range(3):
            k = bins == b
            if k.sum() < 10:
                continue
            vv, aa, tt = ez(correct[k].mean(), rt[k][correct[k]].var() if correct[k].sum() > 2 else rt[k].var(),
                            rt[k][correct[k]].mean() if correct[k].sum() > 2 else rt[k].mean(), k.sum())
            vb.append(vv), ab.append(aa), sb.append(np.abs(s[k]).mean())
        vslope = float(np.polyfit(sb, vb, 1)[0] * np.mean(sb) / np.mean(sb)) if len(sb) > 1 else np.nan
        vslope = float(np.sum(np.array(vb) * np.array(sb)) / np.sum(np.array(sb) ** 2)) if sb else np.nan
        est['v_beh'].append(vslope)
        est['a_beh'].append(float(np.median(ab)) if ab else np.nan)
        est['va_beh'].append(vslope * est['a_beh'][-1])
        drive = []
        for i in range(n):
            k = op[:, 0] == i
            drive.append(boxcars(op[k, 1], dt_true[k]))
        Yd = region_bold(rng, drive, 60, AREA['ddm'][1], ev_runs)
        vv, aa = ddm_keys[best_by_r2(ddm_cands, Yd, Qs)]
        est['v_bold'].append(vv), est['a_bold'].append(aa), est['va_bold'].append(vv * aa)
        # ---- Rescorla-Wagner: engage-or-hold choices, outcomes, striatal RPE
        ctx = np.digitize(op[:, 2], np.quantile(op[:, 2], [0.25, 0.5, 0.75]))
        def rw(al, be=None, choices=None, rr=None):
            V = np.full(4, 0.5)
            rpe = np.zeros(len(op))
            ch = np.zeros(len(op), bool)
            ll = 0.0
            for k in range(len(op)):
                c = ctx[k]
                pf = 1 / (1 + np.exp(-(be if be is not None else 5) * (V[c] - 0.5)))
                if choices is None:
                    ch[k] = rr.random() < (pf if rr.random() > 0.04 else 0.5)
                else:
                    ch[k] = choices[k]
                    ll += np.log(pf if ch[k] else 1 - pf + 1e-12) if be is not None else 0
                if ch[k]:
                    rpe[k] = op[k, 3] - V[c]
                    V[c] += al * rpe[k]
            return ch, rpe, ll
        choices, rpe_true, _ = rw(al_, be_, rr=rng)
        bestll, ab_ = -np.inf, (0, 0)
        for al in alg:
            for be in np.linspace(1, 14, 14):
                ll = rw(al, be, choices=choices)[2]
                if ll > bestll:
                    bestll, ab_ = ll, (al, be)
        est['alpha_beh'].append(ab_[0]), est['beta_beh'].append(ab_[1])
        drive = []
        for i in range(n):
            k = (op[:, 0] == i) & choices
            drive.append(frames_of(op[k, 1] + 0.3, 10, rpe_true[k]))
        Yr = region_bold(rng, drive, 60, AREA['rl'][1], ev_runs)
        cands = []
        for al in alg:
            _, rp, _ = rw(al, None, choices=choices)
            cands.append([hrf_tr(np.c_[frames_of(op[(op[:, 0] == i) & choices, 1] + 0.3, 10,
                                                  rp[(op[:, 0] == i) & choices]),
                                       frames_of(op[(op[:, 0] == i) & choices, 1] + 0.3, 10)]) for i in range(n)])
        est['alpha_bold'].append(alg[best_by_r2(cands, Yr, Qs)])
        # ---- novelty: familiarity of each place, encoded at rate e, forgotten at rate f per minute
        Yn = region_bold(rng, novelty(e_, f_), 60, AREA['novelty'][1], ev_runs)
        ee, ff = nov_keys[best_by_r2(nov_cands, Yn, Qs)]
        est['e_bold'].append(ee), est['f_bold'].append(ff)
        if sj % 10 == 9:
            log(g, n * 5, 'subjects', sj + 1)
    T = {k: np.array(x) for k, x in true.items()}
    E = {k: np.array(x, float) for k, x in est.items()}

    def sr(a, b):
        k = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[k], b[k]).correlation) if k.sum() > 5 else np.nan
    res = dict(
        ddm=dict(v=dict(behaviour=sr(T['v'], E['v_beh']), bold=sr(T['v'], E['v_bold']), agree=sr(E['v_beh'], E['v_bold'])),
                 a=dict(behaviour=sr(T['a'], E['a_beh']), bold=sr(T['a'], E['a_bold']), agree=sr(E['a_beh'], E['a_bold'])),
                 va=dict(behaviour=sr(T['v'] * T['a'], E['va_beh']), bold=sr(T['v'] * T['a'], E['va_bold']),
                         agree=sr(E['va_beh'], E['va_bold'])),
                 trials=len(op), per_min=len(op) / (5 * n),
                 median_gap=float(np.median(np.diff(np.sort(op[:, 1] + op[:, 0] * 1e4))[np.diff(np.sort(op[:, 1] + op[:, 0] * 1e4)) < 300])),
                 strength_sd=float(np.std(np.abs(s)))),
        rl=dict(alpha=dict(behaviour=sr(T['alpha'], E['alpha_beh']), bold=sr(T['alpha'], E['alpha_bold']),
                           agree=sr(E['alpha_beh'], E['alpha_bold'])),
                beta=dict(behaviour=sr(T['beta'], E['beta_beh'])),
                hit_rate=float(op[:, 3].mean())),
        novelty=dict(e=dict(bold=sr(T['e'], E['e_bold'])), f=dict(bold=sr(T['f'], E['f_bold'])),
                     entries_per_min=float(sum(len(t) for t, _ in entries) / (5 * n)),
                     revisit_frac=float(revisit_fraction(entries))))
    return res


def revisit_fraction(entries):
    seen, rev, tot = set(), 0, 0
    for _, st in entries:
        for k in st:
            rev += k in seen
            tot += 1
            seen.add(k)
    return rev / max(tot, 1)


def cog():
    f = p('results_round9_cog.json')
    out = json.load(open(f)) if os.path.exists(f) else {}
    for g in GAMES:
        for mins, n in ((30, 6), (60, 12)):
            key = f'{g}_{mins}'
            if key in out:
                continue
            out[key] = cog_game(g, n)
            log(key, json.dumps(out[key])[:400])
            json.dump(out, open(f, 'w'), indent=1)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('pose', 'all'):
        for g in GAMES:
            for i in range(N_RUNS):
                nav_vars(g, i)
            log('poses', g)
    if stage in ('navfit', 'all'):
        navfit()
    if stage in ('cog', 'all'):
        cog()
