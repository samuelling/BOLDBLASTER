"""BOLDBLASTER round 4: one subject, ten 2-hour sessions of varied gameplay (20 h).

Each session is a different 'level' of the v3 game (levers on): lighting, fog, room scale,
texture SF scale and type, pace, enemy and coin density, covert-cue rate, explosion
strength. Sessions also differ in response gain and noise, as real scan sessions do; the
fitter estimates one gain per session.

Questions: (1) how V1 parameter recovery scales from 2 to 20 h; (2) at equal time (4 h),
does varied gameplay beat 4 h of a single level type?
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr

from boldblaster import game, bands, models as M, sim, fit
import run_v2 as V2
import run_experiment as R1

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_long')
os.makedirs(D, exist_ok=True)
RUNS_PER_SESSION = 24
NV = 120                                      # V1 voxels used (first 120 of the round-2 population)
NF = V2.NF

THEMES = [
    dict(name='Warehouse', tex='natural', light=1.0, fog=12, rooms=(12, 4, 10), speed=1.8, look=28,
         npcs=12, coins=(28, 8), cue=(6, 10), shot_p=0.4, flash=0.45, sf_scale=1.0),
    dict(name='Night ops', tex='bandpass', light=0.4, fog=8, rooms=(12, 4, 10), speed=1.4, look=35,
         npcs=6, coins=(20, 8), cue=(3, 5), shot_p=0.3, flash=0.6, sf_scale=1.0),
    dict(name='Open arena', tex='mixed', light=1.0, fog=30, rooms=(5, 9, 14), speed=2.6, look=25,
         npcs=18, coins=(28, 8), cue=(8, 12), shot_p=0.5, flash=0.5, sf_scale=1.0),
    dict(name='Cathedral', tex='bandpass', light=0.9, fog=30, rooms=(4, 10, 15), speed=1.2, look=30,
         npcs=8, coins=(24, 8), cue=(6, 10), shot_p=0.4, flash=0.4, sf_scale=2.0),
    dict(name='Sprint', tex='bandpass', light=1.0, fog=20, rooms=(12, 4, 10), speed=3.2, look=12,
         npcs=10, coins=(28, 8), cue=(10, 14), shot_p=0.4, flash=0.45, sf_scale=0.5),
    dict(name='Treasure hunt', tex='natural', light=1.0, fog=15, rooms=(10, 5, 11), speed=1.6, look=30,
         npcs=4, coins=(40, 16), cue=(8, 12), shot_p=0.3, flash=0.4, sf_scale=1.0),
    dict(name='Fog bank', tex='mixed', light=0.8, fog=5, rooms=(10, 5, 11), speed=1.6, look=30,
         npcs=10, coins=(28, 8), cue=(4, 6), shot_p=0.4, flash=0.45, sf_scale=1.0),
    dict(name='Mixed day', tex='mixed', light=1.0, fog=18, rooms=(12, 4, 10), speed=2.0, look=28,
         npcs=12, coins=(28, 8), cue=(6, 10), shot_p=0.4, flash=0.45, sf_scale=1.0),
    dict(name='Boss rush', tex='bandpass', light=1.0, fog=25, rooms=(6, 8, 13), speed=2.2, look=32,
         npcs=20, coins=(20, 8), cue=(8, 12), shot_p=0.6, flash=0.7, sf_scale=1.0),
    dict(name='Dusk maze', tex='natural', light=0.6, fog=10, rooms=(16, 3, 7), speed=1.6, look=45,
         npcs=10, coins=(28, 8), cue=(5, 8), shot_p=0.4, flash=0.45, sf_scale=1.0),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def p(name):
    return os.path.join(D, name)


def run_tag(s, i):
    return f's{s}_{i:02d}'


def session_effects(s):
    rng = np.random.default_rng(4000 + (s if isinstance(s, int) else 99))
    return rng.normal(1.0, 0.1), rng.uniform(0.85, 1.15)       # gain, noise scale


def one_run(task):
    s, i = task                         # s: session index, or 'c' for the single-theme control
    tag = run_tag(s, i)
    if os.path.exists(p(f'{tag}.npz')):
        return
    pop, G = _G['pop'], _G['G']
    theme = THEMES[0] if s == 'c' else THEMES[s]
    seed = 20000 + (999 if s == 'c' else s) * 100 + i
    sink = bands.BandSink()
    r = game.simulate_run_v2(seed, duration=V2.DUR, frame_sink=sink, levers=True, theme=theme)
    maps = sink.result()
    gz = r['gaze']
    et = R1.eyetracker(gz, np.random.default_rng(seed + 7))
    Gc_t, Gs_t, Gc_f, Gs_f = G
    Ec, Es, Tc = bands.pool(maps, gz, Gc_t, Gs_t)
    P = {k: v[:NV] for k, v in V2.v1_params(pop).items()}
    att_t = M.attention_trace(r['salient'], gz, pop['v1_x'][:NV], pop['v1_y'][:NV], sigma=2.0)
    gain, nscale = session_effects(s)
    d = M.v1_drive(Ec, Es, Tc, att_t, P)
    vis = M.bold_from_drive(d, pop['v1_hrf_scale'][:NV], pop['v1_amp'][:NV] * gain)
    vx = dict(hrf_scale=pop['v1_hrf_scale'][:NV], amp=pop['v1_amp'][:NV], x=np.zeros(NV),
              noise_sd=pop['v1_noise_sd'][:NV] * nscale, sys_amp=pop['v1_sys_amp'][:NV])
    Y = M.add_nuisance(np.random.default_rng(seed + 11), vis, vx, r['events'])
    Ec, Es, Tc = bands.pool(maps, et, Gc_f, Gs_f)
    att_f = M.attention_trace(r['salient'], et, pop['v1_fx'][:NV], pop['v1_fy'][:NV], sigma=2.5)
    np.savez(p(tag), Ec=Ec.astype(np.float16), Es=Es.astype(np.float16), Tc=Tc.astype(np.float16),
             att=att_f.astype(np.float16), Y=Y, events=r['events'],
             n_cov=np.sum(r['salient'][:, 3] == 2) if len(r['salient']) else 0)
    log(f'{tag} ({theme["name"]}): {len(r["events"])} hits, {len(r["shots"])} shots')


_G = {}


def generate():
    pop = V2.populations()
    G = V2.gauss_sets(pop)
    _G['pop'], _G['G'] = pop, tuple(g[:NV] for g in G)
    tasks = [(s, i) for i in range(RUNS_PER_SESSION) for s in range(len(THEMES))]  # interleave sessions
    tasks += [('c', i) for i in range(RUNS_PER_SESSION)]
    with Pool(2) as pool:
        pool.map(one_run, tasks, chunksize=1)


# ------------------------------------------------------------------ voxel-major store + fits
def runs_for(name):
    S = len(THEMES)
    if name == 'long_2h':
        return [(0, i) for i in range(24)]
    if name == 'long_5h':
        return [(s, i) for s in range(2) for i in range(24)] + [(2, i) for i in range(12)]
    if name == 'long_10h':
        return [(s, i) for s in range(5) for i in range(24)]
    if name == 'long_20h':
        return [(s, i) for s in range(S) for i in range(24)]
    if name == 'single_4h':
        return [(0, i) for i in range(24)] + [('c', i) for i in range(24)]
    if name == 'varied_4h':     # 48 runs spread evenly over the ten levels
        return [(s, i) for s in range(S) for i in range(5)][:48]
    raise ValueError(name)


ORDER = [(s, i) for s in range(len(THEMES)) for i in range(RUNS_PER_SESSION)] + \
    [('c', i) for i in range(RUNS_PER_SESSION)]
INDEX = {k: j for j, k in enumerate(ORDER)}


def build_store():
    """Transpose per-run files into voxel-major memmaps: (NV, n_runs*NF, NB)."""
    from numpy.lib.format import open_memmap
    if os.path.exists(p('store_done')):
        return
    n = len(ORDER)
    st = {k: open_memmap(p(f'store_{k}.npy'), 'w+', np.float16, (NV, n * NF, bands.NB)) for k in ('Ec', 'Es', 'Tc')}
    att = open_memmap(p('store_att.npy'), 'w+', np.float16, (NV, n * NF))
    Y = np.zeros((NV, n * 200), np.float32)
    events = {}
    for j, (s, i) in enumerate(ORDER):
        d = np.load(p(f'{run_tag(s, i)}.npz'))
        for k in st:
            st[k][:, j * NF:(j + 1) * NF] = d[k].transpose(1, 0, 2)
        att[:, j * NF:(j + 1) * NF] = d['att'].T
        Y[:, j * 200:(j + 1) * 200] = d['Y'].T
        events[f'{j}'] = d['events']
    for k in st:
        st[k].flush()
    att.flush()
    np.save(p('store_Y.npy'), Y)
    np.savez(p('store_events'), **events)
    open(p('store_done'), 'w').write('ok')
    log('voxel-major store built')


_F = {}


def _fit_voxel(v):
    Ec, Es, Tc = (np.load(p(f'store_{k}.npy'), mmap_mode='r')[v] for k in ('Ec', 'Es', 'Tc'))
    att = np.load(p('store_att.npy'), mmap_mode='r')[v]
    Y = _F['Y'][v]
    runs = []
    for s, i in _F['runs']:
        j = INDEX[(s, i)]
        sl = slice(j * NF, (j + 1) * NF)
        runs.append(dict(Ec=np.asarray(Ec[sl], np.float32)[:, None], Es=np.asarray(Es[sl], np.float32)[:, None],
                         Tc=np.asarray(Tc[sl], np.float32)[:, None], att=np.asarray(att[sl], np.float32)[:, None],
                         Y=Y[j * 200:(j + 1) * 200, None], Q=_F['Q'][j], group=s))
    fz = M.V1Fitter(runs, M.PNAMES)
    return fz.fit_voxel(0, dict(M.DEFAULT), _F['starts'])


def fit_condition(name):
    build_store()
    runs = runs_for(name)
    ev = np.load(p('store_events.npz'))
    _F['runs'] = runs
    _F['Y'] = np.load(p('store_Y.npy'))
    _F['Q'] = {INDEX[r]: fit.nuisance_basis(200, ev[f'{INDEX[r]}'], arousal=True) for r in runs}
    _F['starts'] = M.starts_for(M.PNAMES, np.random.default_rng(5), n=2)
    with Pool(2) as pool:
        res = pool.map(_fit_voxel, range(NV), chunksize=2)
    return {k: np.array([r[k] for r in res]) for k in M.PNAMES + ['hrf']}


def score(pop, est):
    sub = {f'v1_{k}': pop[f'v1_{k}'][:NV] for k in M.PNAMES}
    return V2.score_v1(sub, est, M.PNAMES)


def fits(names):
    pop = V2.populations()
    res = json.load(open(p('results.json'))) if os.path.exists(p('results.json')) else {}
    for name in names:
        if name in res:
            continue
        t0 = time.time()
        est = fit_condition(name)
        np.savez(p(f'fit_{name}'), **est)
        res[name] = dict(hours=len(runs_for(name)) * 5 / 60, scores=score(pop, est), seconds=time.time() - t0)
        json.dump(res, open(p('results.json'), 'w'), indent=1)
        log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()}, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    V2.calibrate()
    if sys.argv[1] == 'data':
        generate()
    else:
        fits(sys.argv[2:] or ['long_2h', 'single_4h', 'varied_4h', 'long_5h', 'long_10h', 'long_20h'])
    log('done')
