"""BOLDBLASTER: can model-based pRF fits survive free gameplay?

Stages (each cached in data/):
  1. play 8 game runs + 4 bar-retinotopy runs (300 s each), store CE + true gaze + events
  2. ground-truth V1 population (500 voxels) -> simulated BOLD
  3. eye-tracker model -> fit stimuli under different gaze assumptions
  4. fit conditions, cross-validate on held-out game runs and a held-out bar run
"""
import json
import os
import sys
import time
import numpy as np

from boldblaster import game, stimuli as st, sim, fit

OUT = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT, exist_ok=True)
DUR = 300.0
N_GAME, N_BAR, N_VOX = 8, 4, 500
GAME_TRAIN, GAME_TEST = list(range(6)), [6, 7]
BAR_TRAIN, BAR_TEST = [0, 1, 2], [3]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def p(name):
    return os.path.join(OUT, name)


# ------------------------------------------------------------------ stage 1
def stage_play():
    for kind, n, fn, seed0 in (('game', N_GAME, game.simulate_run, 100),
                               ('bar', N_BAR, st.simulate_bar_run, 200)):
        for i in range(n):
            f = p(f'{kind}{i}_ce.npy')
            if os.path.exists(f):
                continue
            sink = st.CESink()
            r = fn(seed0 + i, duration=DUR, frame_sink=sink, keep_frames=600 if i == 0 else 0)
            np.save(f, sink.result())
            np.save(p(f'{kind}{i}_gaze.npy'), r['gaze'])
            np.save(p(f'{kind}{i}_events.npy'), r['events'])
            if i == 0:
                np.save(p(f'{kind}0_frames.npy'), (r['frames'] * 255).astype(np.uint8))
            log(f'played {kind}{i}: {len(r["events"])} events')


def extend_game_runs(n_total):
    """Add game runs N_GAME..n_total-1 (play, truth, fit stimuli) without touching
    the original runs. Used for the 'how many hours?' curve."""
    truth = dict(np.load(p('truth.npz')))
    vox = {k: truth[k] for k in ('x', 'y', 'ecc', 'pa', 'sigma', 'n', 'hrf_scale', 'amp',
                                 'noise_sd', 'sys_amp')}
    G = sim.gaussians(vox['x'], vox['y'], vox['sigma'], step=1)
    if 'ref' not in truth:
        bar_d = np.concatenate([np.maximum(sim.linear_drive(*load_run('bar', i)[:2], G), 0) ** vox['n']
                                for i in range(N_BAR)])
        truth['ref'] = np.percentile(bar_d, 95, axis=0)
    rng = np.random.default_rng(23 + 1000)
    for i in range(N_GAME, n_total):
        if f'game{i}_Y' in truth:
            continue
        if not os.path.exists(p(f'game{i}_ce.npy')):
            sink = st.CESink()
            r = game.simulate_run(100 + i, duration=DUR, frame_sink=sink)
            np.save(p(f'game{i}_ce.npy'), sink.result())
            np.save(p(f'game{i}_gaze.npy'), r['gaze'])
            np.save(p(f'game{i}_events.npy'), r['events'])
        ce, gz, ev = load_run('game', i)
        vis = sim.visual_bold(sim.linear_drive(ce, gz, G), vox, truth['ref'])
        sysm, noise = sim.nuisance_bold(np.random.default_rng(5000 + i), vox, vis.shape[0], ev, DUR)
        truth[f'game{i}_vis'] = vis.astype(np.float32)
        truth[f'game{i}_Y'] = (vis + sysm + noise).astype(np.float32)
        truth[f'game{i}_sys'] = sysm.astype(np.float32)
        et = eyetracker(gz, rng)
        np.save(p(f'game{i}_gaze_et.npy'), et)
        np.save(p(f'S_game{i}_et.npy'), st.fit_stimulus(ce, et))
        log(f'extended game{i}: {len(ev)} events')
        np.savez(p('truth.npz'), **truth)


def load_run(kind, i):
    return (np.load(p(f'{kind}{i}_ce.npy'), mmap_mode='r'),
            np.load(p(f'{kind}{i}_gaze.npy')), np.load(p(f'{kind}{i}_events.npy')))


# ------------------------------------------------------------------ stage 2
def stage_truth():
    if os.path.exists(p('truth.npz')):
        return dict(np.load(p('truth.npz')))
    rng = np.random.default_rng(7)
    vox = sim.sample_voxels(rng, N_VOX)
    G = sim.gaussians(vox['x'], vox['y'], vox['sigma'], step=1)
    lins = {}
    for kind, n in (('game', N_GAME), ('bar', N_BAR)):
        for i in range(n):
            ce, gz, _ = load_run(kind, i)
            lins[(kind, i)] = sim.linear_drive(ce, gz, G)
            log(f'drive {kind}{i}')
    # gain reference: 95th pct of the nonlinear drive across bar runs (bookkeeping only;
    # the same gain applies to every stimulus)
    bar_d = np.concatenate([np.maximum(lins[('bar', i)], 0) ** vox['n'] for i in range(N_BAR)])
    ref = np.percentile(bar_d, 95, axis=0)
    out = {k: v for k, v in vox.items()}
    nrng = np.random.default_rng(11)
    for (kind, i), lin in lins.items():
        vis = sim.visual_bold(lin, vox, ref)
        ev = load_run(kind, i)[2]
        sysm, noise = sim.nuisance_bold(nrng, vox, vis.shape[0], ev if kind == 'game' else [], DUR)
        out[f'{kind}{i}_vis'] = vis.astype(np.float32)
        out[f'{kind}{i}_Y'] = (vis + sysm + noise).astype(np.float32)
        out[f'{kind}{i}_sys'] = sysm.astype(np.float32)
    np.savez(p('truth.npz'), **out)
    return out


# ------------------------------------------------------------------ stage 3
def eyetracker(gaze, rng):
    """In-bore tracker: per-run calibration offset, linear drift, gain error,
    sample noise, and blink dropouts (held at last valid sample)."""
    n = len(gaze)
    tt = np.linspace(0, 1, n)[:, None]
    est = gaze * (1 + rng.normal(0, 0.04, 2)) + rng.normal(0, 0.5, 2) \
        + tt * rng.normal(0, 0.3, 2) + rng.normal(0, 0.3, (n, 2))
    i = 0
    while i < n:
        if rng.random() < 0.012:
            L = rng.integers(2, 4)
            est[i:i + L] = est[i - 1] if i > 0 else 0
            i += L
        i += 1
    return est.astype(np.float32)


def stage_stimuli():
    rng = np.random.default_rng(23)
    for i in range(N_GAME):
        _, gz, _ = load_run('game', i)
        et = eyetracker(gz, rng)
        np.save(p(f'game{i}_gaze_et.npy'), et)
        for mode, g in (('true', gz), ('et', et), ('center', np.zeros_like(gz))):
            f = p(f'S_game{i}_{mode}.npy')
            if not os.path.exists(f):
                np.save(f, st.fit_stimulus(load_run('game', i)[0], g))
        log(f'fit stimuli game{i}')
    for i in range(N_BAR):
        f = p(f'S_bar{i}_center.npy')
        if not os.path.exists(f):
            ce, gz, _ = load_run('bar', i)
            np.save(f, st.fit_stimulus(ce, np.zeros_like(gz)))
    log('fit stimuli bars')


# ------------------------------------------------------------------ stage 4
def runset(truth, spec):
    """spec: list of (kind, idx, gaze_mode, arousal_nuisance)"""
    S, Y, Q, Yv = [], [], [], []
    for kind, i, mode, ar in spec:
        S.append(np.load(p(f'S_{kind}{i}_{mode}.npy')))
        Yr = truth[f'{kind}{i}_Y']
        Y.append(Yr)
        Yv.append(truth[f'{kind}{i}_vis'])
        ev = np.load(p(f'{kind}{i}_events.npy'))
        Q.append(fit.nuisance_basis(Yr.shape[0], ev, arousal=ar))
    return S, Y, Q, Yv


def game_spec(idx, mode='et', ar=True):
    return [('game', i, mode, ar) for i in idx]


def bar_spec(idx):
    return [('bar', i, 'center', False) for i in idx]


CONDITIONS = {
    # name: (train spec, game-test spec)
    'bars': (bar_spec(BAR_TRAIN), game_spec(GAME_TEST, 'et')),
    'game_truegaze': (game_spec(GAME_TRAIN, 'true'), game_spec(GAME_TEST, 'true')),
    'game_eyetracker': (game_spec(GAME_TRAIN, 'et'), game_spec(GAME_TEST, 'et')),
    'game_nogaze': (game_spec(GAME_TRAIN, 'center'), game_spec(GAME_TEST, 'center')),
    'game_et_noarousal': (game_spec(GAME_TRAIN, 'et', False), game_spec(GAME_TEST, 'et', False)),
    'bars+game': (bar_spec(BAR_TRAIN) + game_spec(GAME_TRAIN, 'et'), game_spec(GAME_TEST, 'et')),
    # data-quantity curve
    'game_et_1run': (game_spec([0]), game_spec(GAME_TEST)),
    'game_et_2run': (game_spec([0, 1]), game_spec(GAME_TEST)),
    'game_et_4run': (game_spec([0, 1, 2, 3]), game_spec(GAME_TEST)),
    'bars_1run': (bar_spec([0]), game_spec(GAME_TEST)),
    'bars_2run': (bar_spec([0, 1]), game_spec(GAME_TEST)),
    # long-gameplay point (runs 8-23 are added by extend_game_runs; 2 h with runs 0-5)
    'game_et_2h': (game_spec(list(range(6)) + list(range(8, 26))), game_spec(GAME_TEST)),
}


def eff_size(sig, n):
    return sig / np.sqrt(n)


def summarize(truth, prm, r2g, r2b, ceil_g, ceil_b):
    ecc = truth['ecc']
    pe = np.hypot(prm[:, 0] - truth['x'], prm[:, 1] - truth['y'])
    se = np.log2(eff_size(prm[:, 2], prm[:, 3]) / eff_size(truth['sigma'], truth['n']))
    out = dict(pos_err_median=float(np.median(pe)), size_logratio_median=float(np.median(se)),
               size_abs_log2_median=float(np.median(np.abs(se))),
               frac_pos_err_lt_1deg=float(np.mean(pe < 1)),
               r2_game_test_median=float(np.median(r2g)), r2_bar_test_median=float(np.median(r2b)),
               frac_ceiling_game=float(np.median(r2g / ceil_g)),
               frac_ceiling_bar=float(np.median(r2b / ceil_b)), by_ecc={})
    for lo, hi in ((0, 2), (2, 5), (5, 9), (9, 12.1)):
        m = (ecc >= lo) & (ecc < hi)
        out['by_ecc'][f'{lo}-{hi:g}'] = dict(n=int(m.sum()), pos_err=float(np.median(pe[m])),
                                            size_log2=float(np.median(se[m])))
    return out


def stage_fit(names):
    truth = stage_truth()
    res = json.load(open(p('results.json'))) if os.path.exists(p('results.json')) else {}
    Sb, Yb, Qb, Yvb = runset(truth, bar_spec(BAR_TEST))
    for name in names:
        if name in res:
            continue
        tr_spec, te_spec = CONDITIONS[name]
        t0 = time.time()
        S, Y, Q, _ = runset(truth, tr_spec)
        log(f'[{name}] grid fit on {len(S)} runs')
        prm0, r0 = fit.grid_fit(S, Y, Q)
        prm, r = fit.refine(prm0, S, Y, Q)
        g = fit.gains(prm, S, Y, Q)
        St, Yt, Qt, Yvt = runset(truth, te_spec)
        r2g = fit.cv_r2(prm, g, St, Yt, Qt)
        r2b = fit.cv_r2(prm, g, Sb, Yb, Qb)
        cg = fit.ceiling_r2(Yvt, Yt, Qt)
        cb = fit.ceiling_r2(Yvb, Yb, Qb)
        np.savez(p(f'fit_{name}.npz'), grid=prm0, params=prm, r_train=r, gain=g,
                 r2_game=r2g, r2_bar=r2b, ceil_game=cg, ceil_bar=cb)
        res[name] = summarize(truth, prm, r2g, r2b, cg, cb)
        res[name]['minutes_train'] = len(S) * DUR / 60
        res[name]['fit_seconds'] = time.time() - t0
        json.dump(res, open(p('results.json'), 'w'), indent=1)
        log(f'[{name}] pos err {res[name]["pos_err_median"]:.2f} deg, '
            f'size log2 {res[name]["size_logratio_median"]:+.2f}, '
            f'R2 game {res[name]["r2_game_test_median"]:.3f}, bar {res[name]["r2_bar_test_median"]:.3f} '
            f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    stage_play()
    stage_truth()
    stage_stimuli()
    stage_fit(sys.argv[1:] or list(CONDITIONS))
    log('done')
