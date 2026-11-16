"""Round 10: Betamax with the Diggings -- does adding puzzle elements to the game help the models?

The Diggings (samuelling/betamax, branch `diggings`) adds tunnels and bunkers under the parking lot
and the back of the dunes: a braided maze with several routes, magazine rooms whose ammunition is
carried back up to the line, cave-ins that change the layout, sector blackouts lit only by a
lantern, and friend-or-foe figures. The bot plays the same beach battle and now and then takes a
tunnel mission. Here every round-9 model is refitted on an hour of it and compared with the
original Betamax, and three models that no earlier game could test are added:

  pathint   path integration on the blackout homing trials (heading noise, distance gain, leak;
            The Palace's lantern model), from behaviour (where each player thinks home is) and from
            BOLD (entorhinal / hippocampal positional uncertainty while walking in the dark)
  mapupd    map updating after cave-ins (update rate alpha, route-choice temperature tau), from the
            junction choices and from hippocampal BOLD (believed distance to goal at each junction
            and the surprise at each cave-in)
  ddm_id    drift diffusion for the friend-or-foe decisions (spaced, and varied in difficulty by
            distance and lighting), set against the drift diffusion fitted to the firefight bursts

The round-8 visual models are refitted by run_round8.py (GAME=betamaxt) as before.
Stages: pose | nav | cog | new | all
"""
import os
import sys
import json
from collections import deque

import numpy as np
from scipy.stats import spearmanr

import run_round9 as R9
import run_v2 as V2
from boldblaster import fit, models as M

D, p, log = V2.D, V2.p, V2.log
FPS, NF, NTR = R9.FPS, R9.NF, R9.NTR
# R10_GAME=betamaxt: the first tunnels; R10_GAME=betamaxu: the tuned tunnels (shorter trips, cave-ins
# that stay down, figures ahead, blackouts after pickups, brighter)
G = os.environ.get('R10_GAME', 'betamaxt')
SUF = {'betamaxt': '', 'betamaxu': 'u'}[G]
R9.RAW[G] = {'betamaxt': '/home/claude/bbx/runs_dig', 'betamaxu': '/home/claude/bbx/runs_dig2'}[G]
R9.PRE[G] = G
R9.LABEL[G] = {'betamaxt': 'Betamax + tunnels', 'betamaxu': 'Tuned tunnels'}[G]
R9.AREA.update(pathint=('Entorhinal cortex / hippocampus', 1.5), mapupd=('Hippocampus', 1.4), ddm_id=('Pre-SMA / intraparietal sulcus', 1.0))
N_RUNS = int(os.environ.get('R10_RUNS', 12))
if G not in R9.GAMES:
    R9.GAMES.append(G)


def raw(i):
    recs = [json.loads(l) for l in open(f'{R9.RAW[G]}/{G}{i}_log.jsonl')]
    ev = json.load(open(f'{R9.RAW[G]}/{G}{i}_events.json'))
    t0 = recs[0]['t']
    fr = lambda t: int(np.clip(round((t - t0) * FPS), 0, NF - 1))
    return recs, ev, t0, fr


def plan_graph(plan):
    adj = {k: [] for k in plan['cells']}
    for e in plan['edges']:
        a, b = map(int, e.split(':'))
        adj[a].append(b), adj[b].append(a)
    return adj


def cell_xz(plan, k):
    DG = plan['DG']
    return DG['x0'] + (k % DG['nx']) * DG['cell'], DG['z0'] + (k // DG['nx']) * DG['cell']


def bfs(adj, src, dead=frozenset()):
    d = {src: 0}
    q = deque([src])
    while q:
        a = q.popleft()
        for b in adj[a]:
            if (min(a, b), max(a, b)) in dead or b in d:
                continue
            d[b] = d[a] + 1
            q.append(b)
    return d


def pose_betamaxt(i):
    R, ev, t0, fr = raw(i)
    lg = np.load(p(f'{G}{i}_logs.npz'))
    plan = ev['plan']
    adj = plan_graph(plan)
    hat = {h['id']: h for h in plan['hatches']}
    pos = np.array([[r['x'], r['z']] for r in R], float)
    face = np.array([r['yaw'] for r in R], float)
    body = np.array([r['body'] for r in R])
    tau = np.array([r['tau'] for r in R])
    under = np.array([r['under'] for r in R], bool)
    jumps = sorted(set(np.where((np.diff(body) != 0) | (np.abs(np.diff(tau) - 0.1) > 0.5) | (np.hypot(*np.diff(pos, axis=0).T) > 3))[0] + 1))
    valid = ~R9._invalid_around(jumps)
    vel = np.gradient(pos, axis=0) * FPS
    vel[~valid] = 0
    # goals: the tunnel mission's (hatch on the way down, the magazine or the line below ground),
    # otherwise the nearest visible enemy as before; path distance through the warren, cave-ins included
    phase = [r['phase'] for r in R]
    cells = np.array([r['cell'] for r in R])
    goal_cell = np.full(NF, -1)
    for j in ev['junctions']:
        goal_cell[fr(j[0]):] = j[7]
    col_t = [(fr(e[0]), tuple(sorted(map(int, e[2].split(':'))))) for e in ev['digev'] if e[1] == 'collapses']
    life_start = [0] + list(jumps)
    ge, gp, has = np.zeros(NF), np.zeros(NF), np.zeros(NF, bool)
    mis = {}
    for m in ev['missions']:
        for f in range(fr(m[0]), fr(m[1]) + 1):
            mis[f] = m[2]
    cache = {}
    for f in range(NF):
        if phase[f] == 'toHatch' and f in mis and mis[f] in hat:
            h = hat[mis[f]]
            ge[f] = gp[f] = np.hypot(h['x'] - pos[f, 0], h['z'] - pos[f, 1]); has[f] = True
        elif phase[f] == 'route' and under[f] and goal_cell[f] >= 0 and cells[f] >= 0:
            ls = max(s for s in life_start if s <= f)
            dead = frozenset(e for (tf, e) in col_t if ls <= tf <= f)
            key = (goal_cell[f], dead)
            if key not in cache:
                cache[key] = bfs(adj, int(goal_cell[f]), dead)
            gx, gz = cell_xz(plan, int(goal_cell[f]))
            cx, cz = cell_xz(plan, int(cells[f]))
            ge[f] = np.hypot(gx - pos[f, 0], gz - pos[f, 1])
            gp[f] = cache[key].get(int(cells[f]), 30) * 6 + np.hypot(cx - pos[f, 0], cz - pos[f, 1])
            has[f] = True
        else:
            v = [x[2] for x in R[f]['vis'] if x[3] == 1 and x[5] == 1 and x[4] < 1000]
            if v:
                ge[f] = gp[f] = min(v); has[f] = True
    bd = np.array([r['wall'] if r['under'] else np.nan for r in R], float)
    dig_b = [e[0] for e in ev['digev'] if e[1] in ('descents', 'ascents', 'pickups', 'deliveries', 'collapses', 'lightsOut')]
    bounds = np.r_[lg['events'][:, 0], lg['deaths'], np.array(jumps) / FPS, np.array([fr(t) / FPS for t in dig_b])]
    state = np.where(under, 10 ** 6 + np.maximum(cells, 0), np.floor(pos[:, 0] / 4).astype(int) * 1000 + np.floor(pos[:, 1] / 4).astype(int))
    return dict(pos=pos, vel=vel, face=-face, goal_e=ge, goal_p=gp, has_goal=has, bdist=bd,
                bounds=np.unique(np.round(bounds, 1)), state=state, ctx=np.zeros(NF, int), valid=valid)


R9.POSE[G] = pose_betamaxt
_bvc0 = R9.bvc


def bvc_nan(bdn, d, width=0.12):                  # boundary tuning only below ground (no walls logged on the beach)
    out = _bvc0(np.nan_to_num(bdn, nan=99.0), d, width)
    out[~np.isfinite(bdn)] = 0
    return out


R9.bvc = bvc_nan
_bnorm0 = R9.bdist_norm
R9.bdist_norm = lambda g: np.nanpercentile(np.concatenate([R9.nav_vars(g, i)['bdist'] for i in range(N_RUNS)]), 95) if g == G else _bnorm0(g)


# ------------------------------------------------------------------ navigation and round-9 cognition
def nav():
    pop = R9.nav_population()
    out = {}
    bn = R9.bdist_norm(G)
    for mins, n in ((30, 6), (60, 12)):
        if n > N_RUNS:
            continue
        out[f'{G}_{mins}'] = R9.fit_nav(G, n, pop, bn, realistic=True)
        out[f'{G}_{mins}_oraclegaze'] = dict(grid_gaze=R9.fit_nav(G, n, pop, bn, realistic=False)['grid_gaze'])
        log(G, mins, json.dumps({k: {a: round(b, 2) for a, b in d.items()} for k, d in out[f'{G}_{mins}'].items()}))
    json.dump(out, open(p(f'results_round10{SUF}_nav.json'), 'w'), indent=1)


def cog():
    out = {}
    for mins, n in ((30, 6), (60, 12)):
        if n > N_RUNS:
            continue
        out[f'{G}_{mins}'] = R9.cog_game(G, n)
        log(G, mins, json.dumps(out[f'{G}_{mins}'])[:300])
    json.dump(out, open(p(f'results_round10{SUF}_cog.json'), 'w'), indent=1)


# ------------------------------------------------------------------ new models
NS = 40
SAVE = {}                                 # (model, n) -> (planted, fitted) per simulated player


def sr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[k], b[k]).correlation) if k.sum() > 5 and np.std(a[k]) > 0 and np.std(b[k]) > 0 else np.nan


def session(n):
    """All runs as one session: per-run frame offsets, events and decision/trial lists."""
    runs = []
    for i in range(n):
        R, ev, t0, fr = raw(i)
        runs.append(dict(i=i, ev=ev, fr=fr, lg=np.load(p(f'{G}{i}_logs.npz')), plan=ev['plan']))
    return runs


def homing_trials(runs):
    out = []
    for r in runs:
        for h in r['ev']['homing']:
            t, x, z, hid, hx, hz, outb = h[:7]
            t1, x1, z1, code = h[7:11]
            if code == 0 or len(outb) < 10 or t1 - t < 3:     # a real walk in the dark, ended at the ladder or by the lights
                continue
            ob = np.array(outb, float)
            if np.hypot(ob[0, 1] - hx, ob[0, 2] - hz) > 3:     # the outbound path must start at the ladder
                continue
            seg = np.diff(ob[:, 1:], axis=0)
            L = np.hypot(*seg.T)
            keep = L > 0.02
            if keep.sum() < 5 or L.sum() < 10:
                continue
            # dark walk: frames from t to t1 (distance walked in the dark), for the BOLD signal
            out.append(dict(run=r['i'], f0=r['fr'](t), f1=r['fr'](t1), seg=seg[keep], L=L[keep], start=(x, z), home=(hx, hz), Lout=float(L.sum())))
    return out


def pi_response(tr, sh, gd, lam, rng, n=1):
    """Believed home position(s) after integrating the outbound path with heading noise sh (rad per
    sqrt m), distance gain gd and leak lam (per m)."""
    ang = np.arctan2(tr['seg'][:, 1], tr['seg'][:, 0])
    L = tr['L']
    eps = np.cumsum(rng.normal(0, 1, (n, len(L))) * sh * np.sqrt(L)[None], axis=1)
    v = np.zeros((n, 2))
    for k in range(len(L)):
        v *= np.exp(-lam * L[k])
        v[:, 0] += gd * L[k] * np.cos(ang[k] + eps[:, k]); v[:, 1] += gd * L[k] * np.sin(ang[k] + eps[:, k])
    return np.array(tr['start'])[None] - v


def pathint(runs, n, rng):
    trials = homing_trials(runs)
    if len(trials) < 3:
        return dict(trials=len(trials), note='too few blackout homing trials')
    ev_runs = [r['lg']['events'] for r in runs]
    Qs = [fit.nuisance_basis(NTR, e, arousal=True) for e in ev_runs]
    SH = np.linspace(0.02, 0.16, 8); GD = np.linspace(0.7, 1.3, 7); LA = np.array([0, 0.005, 0.01, 0.02, 0.03])
    # response distributions per trial and grid point (Monte Carlo, Gaussian approximation)
    mr = np.random.default_rng(5)
    stats = {}
    for a, sh in enumerate(SH):
        for b, gd in enumerate(GD):
            for c, la in enumerate(LA):
                mus, covs = [], []
                for tr in trials:
                    s = pi_response(tr, sh, gd, la, mr, n=48)
                    mus.append(s.mean(0)); covs.append(np.cov(s.T) + np.eye(2) * 0.5)
                stats[(a, b, c)] = (mus, covs)

    def ll(resp, key):
        mus, covs = stats[key]
        tot = 0.0
        for y, mu, C in zip(resp, mus, covs):
            dlt = y - mu
            tot += -0.5 * dlt @ np.linalg.solve(C, dlt) - 0.5 * np.log(np.linalg.det(C))
        return tot

    def unc_drive(sh):
        # positional uncertainty while walking in the dark: grows with the outbound path and every dark metre
        out = [np.zeros(NF) for _ in runs]
        for tr in trials:
            r = [k for k, q in enumerate(runs) if q['i'] == tr['run']][0]
            lg = np.load(p(f'nav_{G}{tr["run"]}.npz'))
            sp = lg['speed'] / FPS
            dd = np.cumsum(sp[tr['f0']:tr['f1'] + 1])
            out[r][tr['f0']:tr['f1'] + 1] = 1 - np.exp(-sh ** 2 * (tr['Lout'] + dd) / 0.25)
        return out
    cands = [[R9.hrf_tr(d) for d in unc_drive(sh)] for sh in SH]
    T = dict(sh=[], gd=[], la=[]); E = dict(sh_beh=[], gd_beh=[], la_beh=[], sh_bold=[])
    for sj in range(NS):
        sh, gd, la = rng.uniform(0.03, 0.15), rng.uniform(0.75, 1.25), rng.uniform(0, 0.03)
        T['sh'].append(sh), T['gd'].append(gd), T['la'].append(la)
        resp = [pi_response(tr, sh, gd, la, rng)[0] + rng.normal(0, 0.5, 2) for tr in trials]   # + where they stop
        best = max(stats, key=lambda k: ll(resp, k))
        E['sh_beh'].append(SH[best[0]]), E['gd_beh'].append(GD[best[1]]), E['la_beh'].append(LA[best[2]])
        Y = R9.region_bold(rng, unc_drive(sh), 60, R9.AREA['pathint'][1], ev_runs)
        E['sh_bold'].append(SH[R9.best_by_r2(cands, Y, Qs)])
    SAVE[('pathint', n)] = (T, E)
    return dict(trials=len(trials),
                sh=dict(behaviour=sr(T['sh'], E['sh_beh']), bold=sr(T['sh'], E['sh_bold']), agree=sr(E['sh_beh'], E['sh_bold'])),
                gd=dict(behaviour=sr(T['gd'], E['gd_beh'])), la=dict(behaviour=sr(T['la'], E['la_beh'])))


def mapupd(runs, n, rng):
    """Junction choices after cave-ins. Each simulated player believes the plan; meeting a cave-in
    lowers their belief that the link is open by a factor (1 - alpha). Route lengths are computed
    with believed-closed links costing extra, and choices are a softmax with temperature tau."""
    J, S = [], []
    for k, r in enumerate(runs):
        ev, fr = r['ev'], r['fr']
        adj = plan_graph(r['plan'])
        for j in ev['junctions']:
            J.append(dict(run=k, f=fr(j[0]), cell=j[1], opts=j[3], goal=j[7], adj=adj))
        for s in ev['stale']:
            S.append(dict(run=k, f=fr(s[0]), edge=(min(s[1], s[2]), max(s[1], s[2]))))
    if len(J) < 20 or not S:
        return dict(junctions=len(J), stale=len(S), note='too few cave-ins met')
    # the order of events in session time
    evs = sorted([(j['run'], j['f'], 'j', j) for j in J] + [(s['run'], s['f'], 's', s) for s in S], key=lambda e: (e[0], e[1]))
    ev_runs = [r['lg']['events'] for r in runs]
    Qs = [fit.nuisance_basis(NTR, e, arousal=True) for e in ev_runs]

    def walk(alpha):
        """Believed route lengths at each junction and surprise at each cave-in, for this alpha."""
        bel = {}
        Lj, Lmin, sup = [], [], []
        run_prev = -1
        for run, f, kind, e in evs:
            if run != run_prev:
                bel = {}; run_prev = run      # a new run: the plan is fresh again (cave-ins are cleared between lives)
            if kind == 's':
                b = bel.get(e['edge'], 1.0); sup.append((run, f, b)); bel[e['edge']] = b * (1 - alpha)
            else:
                adj = e['adj']
                cost = lambda a, b2: 1 + 20 * (1 - bel.get((min(a, b2), max(a, b2)), 1.0))
                # Dijkstra from the goal with believed costs
                dist = {e['goal']: 0.0}; q = [(0.0, e['goal'])]
                import heapq
                while q:
                    d0, a = heapq.heappop(q)
                    if d0 > dist[a]:
                        continue
                    for b2 in adj[a]:
                        nd = d0 + cost(a, b2)
                        if nd < dist.get(b2, 1e9):
                            dist[b2] = nd; heapq.heappush(q, (nd, b2))
                Lo = np.array([cost(e['cell'], o) + dist.get(o, 60) for o in e['opts']])
                Lj.append(Lo); Lmin.append((run, f, Lo.min()))
        return Lj, Lmin, sup
    AL = np.linspace(0.05, 0.95, 10); TA = np.array([0.3, 0.5, 0.8, 1.2, 1.8, 2.7, 4.0])
    walks = {a: walk(a) for a in AL}

    def regs(a):
        Lj, Lmin, sup = walks[a]
        out = []
        for k in range(len(runs)):
            x1 = R9.frames_of([f / FPS for (r, f, v) in Lmin if r == k], 10, [v for (r, f, v) in Lmin if r == k])
            x2 = R9.frames_of([f / FPS for (r, f, v) in sup if r == k], 10, [v for (r, f, v) in sup if r == k])
            x3 = R9.frames_of([f / FPS for (r, f, v) in Lmin if r == k], 10)
            out.append(np.c_[x1, x2, x3])
        return out
    cands = [[R9.hrf_tr(x) for x in regs(a)] for a in AL]
    T = dict(alpha=[], tau=[]); E = dict(alpha_beh=[], tau_beh=[], alpha_bold=[])
    for sj in range(NS):
        al, ta = rng.uniform(0.1, 0.9), np.exp(rng.uniform(np.log(0.4), np.log(3.0)))
        T['alpha'].append(al), T['tau'].append(ta)
        a_near = AL[np.argmin(np.abs(AL - al))]
        Lj = walks[a_near][0]
        ch = []
        for Lo in Lj:
            pr = np.exp(-(Lo - Lo.min()) / ta); pr /= pr.sum()
            ch.append(rng.choice(len(Lo), p=pr) if rng.random() > 0.04 else rng.integers(len(Lo)))
        best, bl = None, -1e18
        for a in AL:
            for t in TA:
                tot = 0.0
                for Lo, c in zip(walks[a][0], ch):
                    pr = np.exp(-(Lo - Lo.min()) / t); pr /= pr.sum(); tot += np.log(0.96 * pr[c] + 0.04 / len(Lo))
                if tot > bl:
                    bl, best = tot, (a, t)
        E['alpha_beh'].append(best[0]), E['tau_beh'].append(best[1])
        drive = [x @ np.array([0.03, 1.0, 0.0]) for x in regs(a_near)]
        Y = R9.region_bold(rng, drive, 60, R9.AREA['mapupd'][1], ev_runs)
        E['alpha_bold'].append(AL[R9.best_by_r2(cands, Y, Qs)])
    SAVE[('mapupd', n)] = (T, E)
    return dict(junctions=len(J), stale=len(S),
                alpha=dict(behaviour=sr(T['alpha'], E['alpha_beh']), bold=sr(T['alpha'], E['alpha_bold']), agree=sr(E['alpha_beh'], E['alpha_bold'])),
                tau=dict(behaviour=sr(T['tau'], E['tau_beh'])))


def ddm_id(runs, n, rng):
    """Drift diffusion on the friend-or-foe decisions: evidence strength from distance and lighting."""
    rows = []
    for k, r in enumerate(runs):
        for f in r['ev']['figures']:
            q = f[6] * (1 if f[3] else -1)
            rows.append((k, r['fr'](f[0]) / FPS, 0.5 + 0.5 * q, 1.0))
    if len(rows) < 20:
        return dict(trials=len(rows), note='too few figure decisions')
    op = np.array(rows, float)
    # reuse the round-9 drift-diffusion machinery with these opportunities
    save = R9.opportunities
    R9.opportunities = lambda g, n2: op
    try:
        res = R9.cog_game(G, n)['ddm']
    finally:
        R9.opportunities = save
    res['trials'] = len(rows)
    return res


def new_models():
    out = {}
    for mins, n in ((30, 6), (60, 12)):
        if n > N_RUNS:
            continue
        runs = session(n)
        rng = np.random.default_rng(1000 + n)
        out[f'{G}_{mins}'] = dict(pathint=pathint(runs, n, rng), mapupd=mapupd(runs, n, rng), ddm_id=ddm_id(runs, n, rng))
        log(G, mins, json.dumps(out[f'{G}_{mins}'])[:600])
    json.dump(out, open(p(f'results_round10{SUF}_new.json'), 'w'), indent=1)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('pose', 'all'):
        for i in range(N_RUNS):
            R9.nav_vars(G, i)
        log('poses', G)
    if stage in ('nav', 'all'):
        nav()
    if stage in ('cog', 'all'):
        cog()
    if stage in ('new', 'all'):
        new_models()
