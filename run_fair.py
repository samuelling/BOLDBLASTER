"""BOLDBLASTER round 5: a fair dedicated baseline.

Round 2 compared gameplay against dedicated experiments of only 10 min each (2 x 5-min runs).
Real pRF / pSFT / pCRF mapping uses 30-60+ min. Here each dedicated V1 experiment is extended
to 12 runs (60 min) and fit at 10 / 30 / 60 min, one experiment at a time (as in round 2) and as
a joint battery. The full battery at 60 min per experiment is 5 h, the same budget as round-4
long_5h gameplay. Multi-session batteries get the same session gain / noise jitter as round 4.

stages:  data   - generate dedicated runs j = 2..11 for each kind
         fit    - fit all conditions -> data_v2/results_fair.json
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

from boldblaster import bands, controlled as ctl, models as M, fit
import run_v2 as V2

NJ = 12                 # runs per experiment (12 x 5 min = 60 min)
NV_BATTERY = 120        # battery fits use the round-4 voxel subset (first 120) to match long_*
RUNS_PER_SESSION = 24   # 2-h sessions, as in round 4
p, log, V1_KINDS = V2.p, V2.log, V2.V1_KINDS


def seeds(kind, j):
    k = V1_KINDS.index(kind)
    if j < 2:           # round-2 runs keep their seeds
        return 500 + 10 * k + j, 1100 + 10 * k + j
    return 2000 + 100 * k + j, 3000 + 100 * k + j


def dedicated(kind, j, pop, G):
    tag = f'{kind}{j}'
    if os.path.exists(p(f'{tag}_Y.npz')):
        return
    Gc_t, Gs_t, Gc_f, Gs_f = G
    s_stim, s_noise = seeds(kind, j)
    sink = bands.BandSink()
    r = ctl.run_v1_experiment(kind, s_stim, V2.DUR, frame_sink=sink)
    maps = sink.result()
    side = r['design']['attn_side']
    Ec, Es, Tc = bands.pool(maps, r['gaze'], Gc_t, Gs_t)
    vis = V2.v1_truth_bold(Ec, Es, Tc, M.attention_blocks(side, pop['v1_x']), pop)
    Ec, Es, Tc = bands.pool(maps, np.zeros_like(r['gaze']), Gc_f, Gs_f)
    V2.save_feats(f'{tag}_feat', Ec, Es, Tc, M.attention_blocks(side, pop['v1_fx']))
    rng = np.random.default_rng(s_noise)
    np.savez(p(f'{tag}_Y'), v1_vis=vis, v1_Y=M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), []))
    log('dedicated', tag)


_G = {}


def _task(t):
    dedicated(t[0], t[1], _G['pop'], _G['G'])


def battery_order(nj):
    """Interleave experiments so each session holds a mix (as a real protocol would)."""
    return [(k, j) for j in range(nj) for k in V1_KINDS]


def load(order, nv, sessions):
    """sessions=True: runs are split into 2-h sessions with session gain / noise jitter."""
    rng = np.random.default_rng(77)
    nsess = (len(order) - 1) // RUNS_PER_SESSION + 1
    gain = np.r_[1.0, rng.normal(1, 0.1, nsess - 1)] if sessions else np.ones(nsess)
    nsc = np.r_[1.0, rng.uniform(0.85, 1.15, nsess - 1)] if sessions else np.ones(nsess)
    runs = []
    for idx, (kind, j) in enumerate(order):
        s = idx // RUNS_PER_SESSION
        fe = np.load(p(f'{kind}{j}_feat.npz'))
        d = np.load(p(f'{kind}{j}_Y.npz'))
        vis, Y = d['v1_vis'][:, :nv], d['v1_Y'][:, :nv]
        Y = (vis * gain[s] + (Y - vis) * nsc[s]).astype(np.float32)
        runs.append(dict(Ec=fe['Ec'][:, :nv].astype(np.float32), Es=fe['Es'][:, :nv].astype(np.float32),
                         Tc=fe['Tc'][:, :nv].astype(np.float32), att=fe['att'][:, :nv].astype(np.float32),
                         Y=Y, Q=fit.nuisance_basis(Y.shape[0], [], arousal=False), group=s))
    return runs


def _fit_one(v):
    return _G['fz'].fit_voxel(v, dict(M.DEFAULT), _G['starts'])


def fit_condition(order, free, nv, sessions):
    runs = load(order, nv, sessions)
    _G['fz'] = M.V1Fitter(runs, free)
    _G['starts'] = M.starts_for(free, np.random.default_rng(5), n=2)
    with Pool(2) as pool:
        res = pool.map(_fit_one, range(nv), chunksize=5)
    _G.pop('fz')
    return {k: np.array([r[k] for r in res]) for k in M.PNAMES + ['hrf']}


def conditions():
    C = {}
    for kind in V1_KINDS:
        for mins, nj in ((10, 2), (30, 6), (60, 12)):
            C[f'ded_{kind}_{mins}m'] = dict(order=[(kind, j) for j in range(nj)], free=V2.DEDICATED_FREE[kind],
                                            nv=300, sessions=False, minutes=mins, kind=kind)
    for mins, nj in ((10, 2), (30, 6), (60, 12)):
        C[f'battery_{mins}m'] = dict(order=battery_order(nj), free=list(M.PNAMES), nv=NV_BATTERY,
                                     sessions=True, minutes=5 * mins, kind='battery')
    return C


# ------------------------------------------------------------------ non-V1 dedicated (numerosity, flow, reward)
def other_runs(pop):
    """j = 0, 1 from round 2 (ded_other.npz); j = 2..11 generated here with fresh seeds."""
    path = p('ded_other_fair.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    out = dict(np.load(p('ded_other.npz')))
    for j in range(2, NJ):
        rng = np.random.default_rng(4300 + j)
        N = ctl.numerosity_design(4600 + j)
        nv = M.bold_from_drive(M.num_tuning(N, pop['num_mu'], pop['num_sig']), pop['num_hrf_scale'], pop['num_amp'])
        out[f'num{j}_Y'], out[f'num{j}_N'] = M.add_nuisance(rng, nv, V2.vox(pop, 'num'), []), N
        fwd, yaw = ctl.flow_design(4610 + j)
        mf = np.c_[fwd / 1.8, np.maximum(yaw, 0), np.maximum(-yaw, 0)]
        mv = M.bold_from_drive(mf @ np.stack([pop['mst_wf'], pop['mst_wl'], pop['mst_wr']]),
                               pop['mst_hrf_scale'], pop['mst_amp'])
        out[f'mst{j}_Y'], out[f'mst{j}_fwd'], out[f'mst{j}_yaw'] = \
            M.add_nuisance(rng, mv, V2.vox(pop, 'mst'), []), fwd, yaw
        sh = ctl.reward_design(4620 + j)
        on = M.events_1s(sh[:, 0], np.ones(len(sh)), V2.NF)
        rp = M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], V2.NF)
        rv = M.bold_from_drive(on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None],
                               pop['rpe_hrf_scale'], pop['rpe_amp'])
        out[f'rpe{j}_Y'] = M.add_nuisance(np.random.default_rng(4700 + j), rv, V2.vox(pop, 'rpe'),
                                          sh[sh[:, 2] > 0][:, [0, 3, 4]])
        out[f'rpe{j}_shots'] = sh
    np.savez(path, **out)
    log('non-V1 dedicated runs generated')
    return out


def fit_other(pop):
    ded, out = other_runs(pop), {}
    q0 = lambda Y: fit.nuisance_basis(Y.shape[0])
    for mins, nj in ((10, 2), (30, 6), (60, 12)):
        J = range(nj)
        mu, sg = M.fit_numerosity([dict(N=ded[f'num{j}_N'], Y=ded[f'num{j}_Y'], Q=q0(ded[f'num{j}_Y'])) for j in J])
        out[f'num_ded_{mins}m'] = V2.score_num(pop, mu, sg, mins)
        X = [np.stack([M.hrf_regressor(ded[f'mst{j}_fwd'] / 1.8), M.hrf_regressor(np.maximum(ded[f'mst{j}_yaw'], 0)),
                       M.hrf_regressor(np.maximum(-ded[f'mst{j}_yaw'], 0))], 1) for j in J]
        B = M.glm(X, [ded[f'mst{j}_Y'] for j in J], [q0(ded[f'mst{j}_Y']) for j in J])
        out[f'mst_ded_{mins}m'] = V2.score_mst(pop, B, mins)
        X = [V2.rpe_regs(ded[f'rpe{j}_shots']) for j in J]
        Qs = [fit.nuisance_basis(200, ded[f'rpe{j}_shots'][ded[f'rpe{j}_shots'][:, 2] > 0][:, [0, 3, 4]],
                                 arousal=True) for j in J]
        B = M.glm(X, [ded[f'rpe{j}_Y'] for j in J], Qs)
        out[f'rpe_ded_{mins}m'] = V2.score_rpe(pop, B, mins, sum(len(ded[f'rpe{j}_shots']) for j in J))
    for k, v in out.items():
        log(k, {a: round(b['spearman'], 2) for a, b in v['scores'].items()})
    return out


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    V2.calibrate()
    pop = V2.populations()
    if stage in ('all', 'data'):
        _G['pop'], _G['G'] = pop, V2.gauss_sets(pop)
        tasks = [(k, j) for j in range(2, NJ) for k in V1_KINDS]
        with Pool(2) as pool:
            pool.map(_task, tasks, chunksize=1)
        _G.clear()
    if stage in ('all', 'fit'):
        out_path = p('results_fair.json')
        res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        C = conditions()
        for name in (sys.argv[2:] or list(C)):
            if name in res:
                continue
            c, t0 = C[name], time.time()
            est = fit_condition(c['order'], c['free'], c['nv'], c['sessions'])
            np.savez(p(f'fitfair_{name}'), **est)
            sub = {f'v1_{k}': pop[f'v1_{k}'][:c['nv']] for k in M.PNAMES}
            res[name] = dict(minutes=c['minutes'], kind=c['kind'], free=c['free'], n_vox=c['nv'],
                             scores=V2.score_v1(sub, est, c['free']), seconds=time.time() - t0)
            json.dump(res, open(out_path, 'w'), indent=1)
            log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()},
                f'{time.time() - t0:.0f}s')
    if stage in ('all', 'other'):
        out_path = p('results_fair_other.json')
        json.dump(fit_other(pop), open(out_path, 'w'), indent=1)
    log('done')
