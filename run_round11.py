"""Round 11: Halcyon -- a game built from scratch around the models.

Rounds 8-10 fitted the model battery on games that were not made for it. Halcyon
(samuelling/Halcyon) is a first-person shooter designed so that every model's events happen inside
the same fight, at rates set by the earlier rounds' power analyses (DESIGN.md there):

  calls     people step into the corridor every 4-10 s in groups of 1-5; each is a shoot-or-hold
            decision with graded evidence (what is in the hand, the light, the distance). These
            replace the firefight bursts as the drift-diffusion and Rescorla-Wagner opportunities:
            spaced, varied in difficulty, and with a right answer
  bandit    each flight of stairs has an ambush risk that drifts (a restless bandit); the stair
            choices and their outcomes are a reinforcement-learning task in its own right
  mapupd    the gang opens and shuts security doors; meeting one shut that was believed open is a
            map update (the round-10 model, on the hotel's zone graph)
  pathint   after a pickup the power often fails and the player walks back by torchlight (the
            round-10 model, on the hotel's homing walks)

The visual models are refitted by run_round8.py (GAME=halcyon) as before; here the navigation and
cognitive models (round 9), the round-10 models and the new bandit model are fitted on 30 and 60
minutes of play and compared with Betamax and both tunnel versions.

Stages: pose | nav | cog | new | all
"""
import os
import sys
import json
import pickle

import numpy as np

import run_round9 as R9
import run_round10 as R10
import run_v2 as V2
from boldblaster import fit

D, p, log = V2.D, V2.p, V2.log
FPS, NF, NTR = R9.FPS, R9.NF, R9.NTR
G = 'halcyon'
R10.G = G                                   # the round-10 model code reads its game from here
R9.RAW[G] = '/home/claude/bbx/runs_hx'
R9.PRE[G] = G
R9.LABEL[G] = 'Halcyon'
R9.AREA.update(bandit=('Ventral striatum', 1.3))
N_RUNS = int(os.environ.get('R11_RUNS', 12))
R10.N_RUNS = N_RUNS
if G not in R9.GAMES:
    R9.GAMES.append(G)
raw = R10.raw
LH = 4.2
EXITS = [(-1.0, -19.0), (1.0, -19.0), (-7.0, 25.0), (7.0, 25.0)]   # front doors and the terrace (ground floor)


def pose_halcyon(i):
    R, ev, t0, fr = raw(i)
    lg = np.load(p(f'{G}{i}_logs.npz'))
    plan = ev['plan']
    gx0, gz0, C = plan['grid']['x0'], plan['grid']['z0'], plan['grid']['cell']
    pos = np.array([[r['x'], r['z']] for r in R], float)
    Lf = np.array([r['L'] for r in R])
    face = np.array([r['yaw'] for r in R], float)
    body = np.array([r['body'] for r in R])
    jumps = sorted(set(np.where((np.diff(body) != 0) | (np.hypot(*np.diff(pos, axis=0).T) > 3))[0] + 1))
    valid = ~R9._invalid_around(jumps)
    vel = np.gradient(pos, axis=0) * FPS
    vel[~valid] = 0
    # goals: the guest being looked for (its cell), then the nearest way out; floors apart add a
    # flight's length. Path length is approximated as 1.3 x the straight line plus the stairs.
    tgt = np.full((NF, 3), np.nan)
    for o in ev['objectives']:
        if o[1] == 'new':
            f0 = fr(o[0]); tgt[f0:] = [gx0 + (o[4] + 0.5) * C, gz0 + (o[5] + 0.5) * C, o[3]]
    phase = [r['phase'] for r in R]
    ge, gp, has = np.zeros(NF), np.zeros(NF), np.zeros(NF, bool)
    for f in range(NF):
        if phase[f] == 'find' and np.isfinite(tgt[f, 0]):
            dz = abs(tgt[f, 2] - Lf[f])
            d = np.hypot(tgt[f, 0] - pos[f, 0], tgt[f, 1] - pos[f, 1])
            ge[f] = np.hypot(d, dz * LH); gp[f] = 1.3 * d + dz * 14.0; has[f] = True
        elif phase[f] == 'escort':
            d = min(np.hypot(x - pos[f, 0], z - pos[f, 1]) for x, z in EXITS)
            ge[f] = np.hypot(d, Lf[f] * LH); gp[f] = 1.3 * d + Lf[f] * 14.0; has[f] = True
    bd = np.array([r['wall'] for r in R], float)
    bt = [o[0] for o in ev['objectives']] + [q[0] for q in ev['pickups']] + [q[0] for q in ev['outcomes']]
    lchg = np.where(np.diff(Lf) != 0)[0] + 1
    bounds = np.r_[lg['events'][:, 0], lg['deaths'], np.array(jumps) / FPS, np.array([fr(t) / FPS for t in bt]), lchg / FPS]
    cells = np.array([r['cell'] for r in R])
    return dict(pos=pos, vel=vel, face=-face, goal_e=ge, goal_p=gp, has_goal=has, bdist=bd,
                bounds=np.unique(np.round(bounds, 1)), state=cells, ctx=Lf, valid=valid)


R9.POSE[G] = pose_halcyon


# ------------------------------------------------------------------ calls as the decision opportunities
SIG = dict(aim=1.0, low=0.75, phone=-0.4, hands=-1.0, none=-1.0)


def call_rows(n):
    """(run, t, ph, hostile): one per call. ph = 0.5 + 0.5 x signed evidence, where the evidence is
    what the person holds (pistol aimed or low, phone, raised hands) times how well it can be seen
    (light, distance); its sign is the right answer (shoot a gunman, hold for anyone else)."""
    rows = []
    for i in range(n):
        R, ev, t0, fr = raw(i)
        for c in ev['calls']:
            t, aid, hostile, pose, d, light, grp, obj = c[:8]
            q = min(1.0, light / 0.55) * max(0.15, 1 - d / 26)
            s = abs(SIG.get(pose, -1.0)) * q * (1 if hostile else -1)
            rows.append((i, fr(t) / FPS, 0.5 + 0.5 * s, float(hostile)))
    return np.array(rows, float).reshape(-1, 4)


_opp0 = R9.opportunities
R9.opportunities = lambda g, n: call_rows(n) if g == G else _opp0(g, n)


# ------------------------------------------------------------------ the stairwell bandit
def bandit_trials(runs):
    """Each crossing of a flight: its time, the flights on offer with their route lengths, and the
    flight taken (from the most recent stair choice for that flight)."""
    out = []
    for k, r in enumerate(runs):
        ev, fr = r['ev'], r['fr']
        ch = ev['stairs']
        for o in ev['outcomes']:
            t, ramp = o[0], o[1]
            prev = [c for c in ch if c[0] <= t and c[1] == ramp]
            if not prev:
                continue
            c = prev[-1]
            if ramp not in c[2]:
                continue
            out.append(dict(run=k, f=fr(t), opts=list(c[2]), cost=np.array(c[3], float), chosen=c[2].index(ramp)))
    return out


def risk_paths(rng, n_ramps, t_end, step=20.0):
    """The game's rule for the flights' ambush risk: a bounded random walk, a step every 20 s."""
    nst = int(t_end / step) + 2
    r = np.empty((nst, n_ramps)); r[0] = 0.2 + rng.random(n_ramps) * 0.6
    for k in range(1, nst):
        r[k] = np.clip(r[k - 1] + (rng.random(n_ramps) - 0.5) * 0.24, 0.08, 0.92)
    return r


def bandit(runs, n, rng):
    tr = bandit_trials(runs)
    if len(tr) < 20:
        return dict(trials=len(tr), note='too few stair crossings')
    ev_runs = [r['lg']['events'] for r in runs]
    Qs = [fit.nuisance_basis(NTR, e, arousal=True) for e in ev_runs]
    n_ramps = 6
    T_SOFT = 2.2
    times = np.array([t['run'] * (V2.DUR + R9.GAP) + t['f'] / FPS for t in tr])

    def play(al, lam, rr, risk=None, choices=None, outcomes=None):
        """Rescorla-Wagner on each flight's risk; softmax over route length + lam x risk."""
        Q = np.full(n_ramps, 0.5)
        ch, oc, rpe, ll = np.zeros(len(tr), int), np.zeros(len(tr)), np.zeros(len(tr)), 0.0
        for k, t in enumerate(tr):
            u = -((t['cost'] - t['cost'].min()) + lam * Q[t['opts']]) / T_SOFT
            pr = np.exp(u - u.max()); pr /= pr.sum(); pr = 0.96 * pr + 0.04 / len(pr)
            if choices is None:
                c = rr.choice(len(pr), p=pr)
                fl = t['opts'][c]
                o = float(rr.random() < risk[min(len(risk) - 1, int(times[k] / 20.0)), fl])
            else:
                c, o = choices[k], outcomes[k]
                fl = t['opts'][c]
                ll += np.log(pr[c])
            ch[k], oc[k] = c, o
            rpe[k] = o - Q[fl]; Q[fl] += al * rpe[k]
        return ch, oc, rpe, ll
    ALG = np.linspace(0.05, 0.85, 12); LAG = np.linspace(0, 16, 9)
    T = dict(alpha=[], lam=[]); E = dict(alpha_beh=[], lam_beh=[], alpha_bold=[])
    for sj in range(R10.NS):
        al, lam = rng.uniform(0.08, 0.8), rng.uniform(1, 15)
        T['alpha'].append(al), T['lam'].append(lam)
        risk = risk_paths(rng, n_ramps, times.max() + 60)
        ch, oc, rpe, _ = play(al, lam, rng, risk=risk)
        best, bl = (ALG[0], LAG[0]), -1e18
        for a in ALG:
            for l in LAG:
                ll = play(a, l, None, choices=ch, outcomes=oc)[3]
                if ll > bl:
                    bl, best = ll, (a, l)
        E['alpha_beh'].append(best[0]), E['lam_beh'].append(best[1])
        # striatal BOLD: the prediction error at each landing
        drive = []
        for k in range(len(runs)):
            m = np.array([t['run'] == k for t in tr])
            drive.append(R9.frames_of(np.array([t['f'] / FPS for t in tr])[m], 10, rpe[m]))
        Y = R9.region_bold(rng, drive, 60, R9.AREA['bandit'][1], ev_runs)
        cands = []
        for a in ALG:
            rp = play(a, best[1], None, choices=ch, outcomes=oc)[2]
            cands.append([R9.hrf_tr(np.c_[R9.frames_of(np.array([t['f'] / FPS for t in tr])[np.array([t['run'] == k for t in tr])], 10,
                                                         rp[np.array([t['run'] == k for t in tr])]),
                                          R9.frames_of(np.array([t['f'] / FPS for t in tr])[np.array([t['run'] == k for t in tr])], 10)])
                          for k in range(len(runs))])
        E['alpha_bold'].append(ALG[R9.best_by_r2(cands, Y, Qs)])
    R10.SAVE[('bandit', n)] = (T, E)
    sr = R10.sr
    return dict(trials=len(tr), per_min=len(tr) / (5 * n),
                alpha=dict(behaviour=sr(T['alpha'], E['alpha_beh']), bold=sr(T['alpha'], E['alpha_bold']), agree=sr(E['alpha_beh'], E['alpha_bold'])),
                lam=dict(behaviour=sr(T['lam'], E['lam_beh'])))


# ------------------------------------------------------------------ stages
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
    json.dump(out, open(p('results_round11_nav.json'), 'w'), indent=1)


def cog():
    out = {}
    for mins, n in ((30, 6), (60, 12)):
        if n > N_RUNS:
            continue
        out[f'{G}_{mins}'] = R9.cog_game(G, n)
        log(G, mins, json.dumps(out[f'{G}_{mins}'])[:300])
    json.dump(out, open(p('results_round11_cog.json'), 'w'), indent=1)
    pickle.dump({k: v for k, v in R9.SAVE.items() if k[0] == G}, open(p('diag_round11_cog.pkl'), 'wb'))   # every player's planted and fitted values


def new_models():
    out = {}
    for mins, n in ((30, 6), (60, 12)):
        if n > N_RUNS:
            continue
        runs = R10.session(n)
        rng = np.random.default_rng(1100 + n)
        out[f'{G}_{mins}'] = dict(pathint=R10.pathint(runs, n, rng), mapupd=R10.mapupd(runs, n, rng), bandit=bandit(runs, n, rng))
        log(G, mins, json.dumps(out[f'{G}_{mins}'])[:600])
    json.dump(out, open(p('results_round11_new.json'), 'w'), indent=1)
    pickle.dump(dict(R10.SAVE), open(p('diag_round11_new.pkl'), 'wb'))


def rates(n=N_RUNS):
    """Event rates per hour of play, against the design targets."""
    tot = dict(calls=0, stairs=0, stale=0, homing=0, pickups=0, rescued=0, deaths=0, kills=0, junctions=0, dark=0.0, outdoors=0.0)
    for i in range(n):
        R, ev, t0, fr = raw(i)
        tot['calls'] += len(ev['calls']); tot['stairs'] += len(ev['outcomes']); tot['stale'] += len(ev['stale'])
        tot['homing'] += len(R10.homing_trials([dict(i=i, ev=ev, fr=fr)])); tot['pickups'] += len(ev['pickups'])
        tot['rescued'] += sum(1 for o in ev['objectives'] if o[1] == 'rescued'); tot['deaths'] += len(ev['deaths'])
        tot['kills'] += len(ev['kills']); tot['junctions'] += len(ev['junctions'])
        tot['dark'] += np.mean([r['dark'] for r in R]); tot['outdoors'] += np.mean([r['room'] >= 0 and r['L'] == 0 and r['z'] > 20 for r in R])
    h = n * 5 / 60
    out = {k: (v / h if k not in ('dark', 'outdoors') else v / n) for k, v in tot.items()}
    json.dump(out, open(p('results_round11_rates.json'), 'w'), indent=1)
    log('rates per hour', json.dumps({k: round(v, 3) for k, v in out.items()}))
    return out


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('pose', 'all'):
        for i in range(N_RUNS):
            R9.nav_vars(G, i)
        log('poses', G)
        rates()
    if stage in ('nav', 'all'):
        nav()
    if stage in ('cog', 'all'):
        cog()
    if stage in ('new', 'all'):
        new_models()
