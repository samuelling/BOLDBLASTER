"""BOLDBLASTER v2: does gameplay recover many models at once?

Populations (all simulated in the same 'subject'):
  V1 rich (300 voxels): pRF (from v1 truth) + pSFT (mu, sig) + pCRF (c50, q)
                        + surround normalization (k) + transient weight (tau) + attention gain (a)
  numerosity (150), self-motion/MST (150), striatal RPE (100)
Data: 24 v2 game runs (2 h) and dedicated experiments (2 x 5 min each).
Fits: game 1 h / 2 h (eye-tracker gaze + bar-fitted pRFs), game 2 h oracle (true gaze + true
pRFs), dedicated one-model-at-a-time, and the full dedicated battery fit jointly.
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr

from boldblaster import game, bands, controlled as ctl, models as M, sim, fit
import run_experiment as R1

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v2')
os.makedirs(D, exist_ok=True)
N_GAME = 24
DUR = 300.0
NF = int(DUR * game.FPS)
V1_KINDS = ['psft', 'crf', 'size', 'temporal', 'attn']
DEDICATED_FREE = dict(psft=['mu', 'sig'], crf=['c50', 'q'], size=['k'], temporal=['tau'], attn=['a'])


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def p(name):
    return os.path.join(D, name)


# ------------------------------------------------------------------ calibration + populations
def calibrate():
    rng = np.random.default_rng(0)
    frames = np.stack([ctl.noise(rng, 'pink') for _ in range(20)])
    bands.E_SCALE = 1.0
    e = bands.band_energy(frames)
    c = e[:, :, bands.MH // 2 - 8:bands.MH // 2 + 8, bands.MW // 2 - 8:bands.MW // 2 + 8]
    scale = 1.0 / float(c.mean())
    bands.E_SCALE = scale
    return scale


def populations():
    f = p('populations.npz')
    if os.path.exists(f):
        return {k: v for k, v in np.load(f).items()}
    t1 = dict(np.load(os.path.join(R1.OUT, 'truth.npz')))
    bar_fit = np.load(os.path.join(R1.OUT, 'fit_bars.npz'))['params']
    rng = np.random.default_rng(99)
    idx = np.sort(rng.choice(len(t1['x']), 300, replace=False))
    pop = dict(v1_idx=idx)
    for k in ('x', 'y', 'sigma', 'ecc', 'hrf_scale', 'amp', 'noise_sd', 'sys_amp'):
        pop[f'v1_{k}'] = t1[k][idx]
    pop['v1_fx'], pop['v1_fy'], pop['v1_fsigma'] = bar_fit[idx, 0], bar_fit[idx, 1], bar_fit[idx, 2]
    planted = M.sample_v1(rng, 300)
    planted['mu'] = M.plant_mu(rng, pop['v1_ecc'])
    for k, v in planted.items():
        pop[f'v1_{k}'] = v
    for name, n in (('num', 150), ('mst', 150), ('rpe', 100)):
        pop[f'{name}_hrf_scale'] = np.clip(rng.normal(1, 0.08, n), 0.8, 1.2)
        pop[f'{name}_amp'] = rng.uniform(1.0, 2.0, n)
        pop[f'{name}_noise_sd'] = rng.uniform(0.6, 1.2, n)
        pop[f'{name}_sys_amp'] = np.clip(rng.normal(0.5, 0.2, n), 0.05, None)
    pop['num_mu'] = np.exp(rng.uniform(np.log(1), np.log(7), 150))
    pop['num_sig'] = rng.uniform(0.3, 0.8, 150)
    for k in ('wf', 'wl', 'wr'):
        pop[f'mst_{k}'] = rng.uniform(0, 1, 150)
    pop['rpe_g'] = rng.uniform(0, 1, 100)
    pop['rpe_b'] = rng.uniform(-0.2, 1.0, 100)
    np.savez(f, **pop)
    return pop


def vox(pop, name):
    return dict(hrf_scale=pop[f'{name}_hrf_scale'], amp=pop[f'{name}_amp'],
                noise_sd=pop[f'{name}_noise_sd'], sys_amp=pop[f'{name}_sys_amp'],
                x=np.zeros(len(pop[f'{name}_amp'])))


def v1_params(pop):
    return {k: pop[f'v1_{k}'] for k in M.PNAMES}


def gauss_sets(pop):
    Gc_t = sim.gaussians(pop['v1_x'], pop['v1_y'], pop['v1_sigma'], step=2)
    Gs_t = sim.gaussians(pop['v1_x'], pop['v1_y'], 3 * pop['v1_sigma'], step=2)
    fs = np.clip(pop['v1_fsigma'], 0.15, None)
    Gc_f = sim.gaussians(pop['v1_fx'], pop['v1_fy'], fs, step=2)
    Gs_f = sim.gaussians(pop['v1_fx'], pop['v1_fy'], 3 * fs, step=2)
    out = []
    for G in (Gc_t, Gs_t, Gc_f, Gs_f):      # zero far tails: subnormal floats stall BLAS
        G = np.where(G < 1e-9, 0, G).astype(np.float32)
        out.append(G / G.sum(1, keepdims=True))
    return tuple(out)


def save_feats(prefix, Ec, Es, Tc, att):
    # kept at the 10 Hz frame rate: the fitter applies its nonlinearity before binning
    np.savez(p(prefix), Ec=Ec.astype(np.float16), Es=Es.astype(np.float16),
             Tc=Tc.astype(np.float16), att=att.astype(np.float16))


def v1_truth_bold(Ec, Es, Tc, att, pop):
    d = M.v1_drive(Ec, Es, Tc, att, v1_params(pop))
    return M.bold_from_drive(d, pop['v1_hrf_scale'], pop['v1_amp'])


# ------------------------------------------------------------------ game runs
def game_run(i, pop, G, prefix='game', levers=False):
    if os.path.exists(p(f'{prefix}{i}_Y.npz')):
        return
    Gc_t, Gs_t, Gc_f, Gs_f = G
    sink = bands.BandSink()
    r = game.simulate_run_v2((800 if levers else 300) + i, duration=DUR, frame_sink=sink, levers=levers)
    maps = sink.result()
    gz = r['gaze']
    et = R1.eyetracker(gz, np.random.default_rng(700 + i))
    # truth + oracle features (true gaze, true pRFs)
    Ec, Es, Tc = bands.pool(maps, gz, Gc_t, Gs_t)
    att_t = M.attention_trace(r['salient'], gz, pop['v1_x'], pop['v1_y'], sigma=2.0)
    vis = v1_truth_bold(Ec, Es, Tc, att_t, pop)
    save_feats(f'{prefix}{i}_feat_oracle', Ec, Es, Tc, att_t)
    del Ec, Es, Tc
    # realistic features (eye tracker, bar-fitted pRFs, assumed attention width 2.5 deg)
    Ec, Es, Tc = bands.pool(maps, et, Gc_f, Gs_f)
    att_f = M.attention_trace(r['salient'], et, pop['v1_fx'], pop['v1_fy'], sigma=2.5)
    save_feats(f'{prefix}{i}_feat', Ec, Es, Tc, att_f)
    del Ec, Es, Tc, maps
    ev = r['events']
    rng = np.random.default_rng(900 + i)
    out = dict(v1_vis=vis, v1_Y=M.add_nuisance(rng, vis, vox(pop, 'v1'), ev))
    # numerosity
    nd = M.num_tuning(r['n_coins'], pop['num_mu'], pop['num_sig'])
    nvis = M.bold_from_drive(nd, pop['num_hrf_scale'], pop['num_amp'])
    out['num_Y'] = M.add_nuisance(rng, nvis, vox(pop, 'num'), ev)
    # self-motion
    mf = np.c_[r['fwd'] / 1.8, np.maximum(r['yaw'], 0), np.maximum(-r['yaw'], 0)]
    md = mf @ np.stack([pop['mst_wf'], pop['mst_wl'], pop['mst_wr']])
    mvis = M.bold_from_drive(md, pop['mst_hrf_scale'], pop['mst_amp'])
    out['mst_Y'] = M.add_nuisance(rng, mvis, vox(pop, 'mst'), ev)
    # striatal RPE
    sh = r['shots']
    on = M.events_1s(sh[:, 0], np.ones(len(sh)), NF)
    rp = M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
    rd = on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None]
    rvis = M.bold_from_drive(rd, pop['rpe_hrf_scale'], pop['rpe_amp'])
    out['rpe_Y'] = M.add_nuisance(rng, rvis, vox(pop, 'rpe'), ev)
    np.savez(p(f'{prefix}{i}_logs'), gaze=gz, et=et, events=ev, salient=r['salient'], shots=sh,
             n_coins=r['n_coins'], fwd=r['fwd'], yaw=r['yaw'])
    np.savez(p(f'{prefix}{i}_Y'), **out)
    log(f'{prefix}{i}: {len(sh)} shots, {len(ev)} hits, coins visible {np.mean(r["n_coins"] > 0):.2f}')


# ------------------------------------------------------------------ dedicated runs
def v1_dedicated(kind, j, pop, G):
    tag = f'{kind}{j}'
    if os.path.exists(p(f'{tag}_Y.npz')):
        return
    Gc_t, Gs_t, Gc_f, Gs_f = G
    sink = bands.BandSink()
    r = ctl.run_v1_experiment(kind, 500 + 10 * V1_KINDS.index(kind) + j, DUR, frame_sink=sink)
    maps = sink.result()
    side = r['design']['attn_side']
    Ec, Es, Tc = bands.pool(maps, r['gaze'], Gc_t, Gs_t)
    att_t = M.attention_blocks(side, pop['v1_x'])
    vis = v1_truth_bold(Ec, Es, Tc, att_t, pop)
    Ec, Es, Tc = bands.pool(maps, np.zeros_like(r['gaze']), Gc_f, Gs_f)
    save_feats(f'{tag}_feat', Ec, Es, Tc, M.attention_blocks(side, pop['v1_fx']))
    rng = np.random.default_rng(1100 + 10 * V1_KINDS.index(kind) + j)
    np.savez(p(f'{tag}_Y'), v1_vis=vis, v1_Y=M.add_nuisance(rng, vis, vox(pop, 'v1'), []))
    log(f'dedicated {tag}')


def other_dedicated(pop):
    if os.path.exists(p('ded_other.npz')):
        return
    out = {}
    for j in range(2):
        rng = np.random.default_rng(1300 + j)
        N = ctl.numerosity_design(600 + j)
        nv = M.bold_from_drive(M.num_tuning(N, pop['num_mu'], pop['num_sig']),
                               pop['num_hrf_scale'], pop['num_amp'])
        out[f'num{j}_Y'], out[f'num{j}_N'] = M.add_nuisance(rng, nv, vox(pop, 'num'), []), N
        fwd, yaw = ctl.flow_design(610 + j)
        mf = np.c_[fwd / 1.8, np.maximum(yaw, 0), np.maximum(-yaw, 0)]
        mv = M.bold_from_drive(mf @ np.stack([pop['mst_wf'], pop['mst_wl'], pop['mst_wr']]),
                               pop['mst_hrf_scale'], pop['mst_amp'])
        out[f'mst{j}_Y'], out[f'mst{j}_fwd'], out[f'mst{j}_yaw'] = \
            M.add_nuisance(rng, mv, vox(pop, 'mst'), []), fwd, yaw
        sh = ctl.reward_design(620 + j)
        on = M.events_1s(sh[:, 0], np.ones(len(sh)), NF)
        rp = M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
        rv = M.bold_from_drive(on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None],
                               pop['rpe_hrf_scale'], pop['rpe_amp'])
        wins = sh[sh[:, 2] > 0][:, [0, 3, 4]]
        out[f'rpe{j}_Y'] = M.add_nuisance(rng, rv, vox(pop, 'rpe'), wins)
        out[f'rpe{j}_shots'] = sh
    np.savez(p('ded_other'), **out)
    log('dedicated numerosity / flow / reward')


# ------------------------------------------------------------------ fitting
_FIT = {}


def _data_task(task):
    kind, a, b = task
    if kind == 'ded':
        v1_dedicated(a, b, _FIT['pop'], _FIT['G'])
    elif kind == 'g3':
        game_run(a, _FIT['pop'], _FIT['G'], prefix='g3_', levers=True)
    else:
        game_run(a, _FIT['pop'], _FIT['G'])


def _fit_one(v):
    fz, free, starts = _FIT['fitter'], _FIT['free'], _FIT['starts']
    fixed = dict(M.DEFAULT)
    return fz.fit_voxel(v, fixed, starts)


def load_v1_runs(specs, oracle=False):
    runs = []
    for tag, events in specs:
        fe = np.load(p(f'{tag}_feat_oracle.npz' if oracle else f'{tag}_feat.npz'))
        Y = np.load(p(f'{tag}_Y.npz'))['v1_Y']
        runs.append(dict(Ec=fe['Ec'].astype(np.float32), Es=fe['Es'].astype(np.float32),
                         Tc=fe['Tc'].astype(np.float32), att=fe['att'].astype(np.float32),
                         Y=Y, Q=fit.nuisance_basis(Y.shape[0], events, arousal=len(events) > 0)))
    return runs


def game_specs(idx, prefix='game'):
    return lambda: [(f'{prefix}{i}', np.load(p(f'{prefix}{i}_logs.npz'))['events']) for i in idx]


def fit_v1(name, specs, free, oracle=False):
    runs = load_v1_runs(specs, oracle)
    _FIT['fitter'] = M.V1Fitter(runs, free)
    _FIT['free'] = free
    _FIT['starts'] = M.starts_for(free, np.random.default_rng(5), n=2)
    with Pool(2) as pool:
        res = pool.map(_fit_one, range(runs[0]['Y'].shape[1]), chunksize=10)
    return {k: np.array([r[k] for r in res]) for k in M.PNAMES + ['hrf']}


def score_v1(pop, est, free):
    out = {}
    if 'c50' in free:   # identifiable quantity: half-saturation for full-field broadband noise
        t = pop['v1_c50'] / (1 + pop['v1_k']) ** (1 / pop['v1_q'])
        e = est['c50'] / (1 + est['k']) ** (1 / est['q'])
        out['c50_eff'] = dict(spearman=float(spearmanr(t, e).correlation),
                              median_abs_err=float(np.median(np.abs(np.log2(e / t)))))
    for k in free:
        t, e = pop[f'v1_{k}'], est[k]
        if k == 'c50':
            err = np.abs(np.log2(e / t))
        else:
            err = np.abs(e - t)
        out[k] = dict(spearman=float(spearmanr(t, e).correlation), median_abs_err=float(np.median(err)))
    return out


def run_fits(names):
    pop = populations()
    res = json.load(open(p('results.json'))) if os.path.exists(p('results.json')) else {}
    allp = list(M.PNAMES)
    V1 = {
        'game_1h': (game_specs(range(12)), allp, False, 60),
        'game_2h': (game_specs(range(24)), allp, False, 120),
        'game_2h_oracle': (game_specs(range(24)), allp, True, 120),
        'battery_joint': ([(f'{k}{j}', []) for k in V1_KINDS for j in range(2)], allp, False, 50),
        'levers_1h': (game_specs(range(12), 'g3_'), allp, False, 60),
        'levers_2h': (game_specs(range(24), 'g3_'), allp, False, 120),
        'levers_2h_oracle': (game_specs(range(24), 'g3_'), allp, True, 120),
    }
    for kind in V1_KINDS:
        V1[f'dedicated_{kind}'] = ([(f'{kind}{j}', []) for j in range(2)], DEDICATED_FREE[kind], False, 10)
    for name in names:
        if name in res or name not in V1:
            continue
        t0 = time.time()
        specs, free, oracle, minutes = V1[name]
        est = fit_v1(name, specs() if callable(specs) else specs, free, oracle)
        np.savez(p(f'fit_{name}'), **est)
        res[name] = dict(population='v1', minutes=minutes, free=free, scores=score_v1(pop, est, free),
                         seconds=time.time() - t0)
        json.dump(res, open(p('results.json'), 'w'), indent=1)
        log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()},
            f'{time.time() - t0:.0f}s')
    if 'nonv1' in names:
        res.update(fit_nonv1(pop))
        json.dump(res, open(p('results.json'), 'w'), indent=1)


def fit_nonv1(pop):
    out = {}
    ded = np.load(p('ded_other.npz'))

    def game_runs(n):
        rs = []
        for i in range(n):
            lg, Y = np.load(p(f'game{i}_logs.npz')), np.load(p(f'game{i}_Y.npz'))
            Q = fit.nuisance_basis(200, lg['events'], arousal=True)
            rs.append((lg, Y, Q))
        return rs

    def q0(Yr):
        return fit.nuisance_basis(Yr.shape[0])

    for label, n in (('game_1h', 12), ('game_2h', 24)):
        rs = game_runs(n)
        # numerosity
        mu, sg = M.fit_numerosity([dict(N=lg['n_coins'], Y=Y['num_Y'], Q=Q) for lg, Y, Q in rs])
        out[f'num_{label}'] = score_num(pop, mu, sg, n * 5)
        # self-motion
        X = [np.stack([M.hrf_regressor(lg['fwd'] / 1.8), M.hrf_regressor(np.maximum(lg['yaw'], 0)),
                       M.hrf_regressor(np.maximum(-lg['yaw'], 0))], 1) for lg, Y, Q in rs]
        B = M.glm(X, [Y['mst_Y'] for _, Y, _ in rs], [Q for _, _, Q in rs])
        out[f'mst_{label}'] = score_mst(pop, B, n * 5)
        # RPE
        X = [rpe_regs(lg['shots']) for lg, Y, Q in rs]
        B = M.glm(X, [Y['rpe_Y'] for _, Y, _ in rs], [Q for _, _, Q in rs])
        out[f'rpe_{label}'] = score_rpe(pop, B, n * 5, sum(len(lg['shots']) for lg, _, _ in rs))
    # dedicated
    runsN = [dict(N=ded[f'num{j}_N'], Y=ded[f'num{j}_Y'], Q=q0(ded[f'num{j}_Y'])) for j in range(2)]
    mu, sg = M.fit_numerosity(runsN)
    out['num_dedicated'] = score_num(pop, mu, sg, 10)
    X = [np.stack([M.hrf_regressor(ded[f'mst{j}_fwd'] / 1.8), M.hrf_regressor(np.maximum(ded[f'mst{j}_yaw'], 0)),
                   M.hrf_regressor(np.maximum(-ded[f'mst{j}_yaw'], 0))], 1) for j in range(2)]
    B = M.glm(X, [ded[f'mst{j}_Y'] for j in range(2)], [q0(ded[f'mst{j}_Y']) for j in range(2)])
    out['mst_dedicated'] = score_mst(pop, B, 10)
    X = [rpe_regs(ded[f'rpe{j}_shots']) for j in range(2)]
    Qs = [fit.nuisance_basis(200, ded[f'rpe{j}_shots'][ded[f'rpe{j}_shots'][:, 2] > 0][:, [0, 3, 4]], arousal=True)
          for j in range(2)]
    B = M.glm(X, [ded[f'rpe{j}_Y'] for j in range(2)], Qs)
    out['rpe_dedicated'] = score_rpe(pop, B, 10, sum(len(ded[f'rpe{j}_shots']) for j in range(2)))
    for k, v in out.items():
        log(k, v)
    return out


def rpe_regs(sh):
    on = M.events_1s(sh[:, 0], np.ones(len(sh)), NF)
    rp = M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
    return np.stack([M.hrf_regressor(on), M.hrf_regressor(rp)], 1)


def score_num(pop, mu, sg, minutes):
    return dict(population='numerosity', minutes=minutes, scores=dict(
        mu=dict(spearman=float(spearmanr(pop['num_mu'], mu).correlation),
                median_abs_err=float(np.median(np.abs(np.log2(mu / pop['num_mu']))))),
        sig=dict(spearman=float(spearmanr(pop['num_sig'], sg).correlation),
                 median_abs_err=float(np.median(np.abs(sg - pop['num_sig']))))))


def score_mst(pop, B, minutes):
    Bp = np.maximum(B, 1e-6)
    wt = np.stack([pop['mst_wf'], pop['mst_wl'], pop['mst_wr']]) + 1e-6
    di = lambda w: (w[1] - w[2]) / (w[1] + w[2])
    ef = lambda w: w[0] / w.sum(0)
    return dict(population='self-motion', minutes=minutes, scores=dict(
        rotation_pref=dict(spearman=float(spearmanr(di(wt), di(Bp)).correlation),
                           median_abs_err=float(np.median(np.abs(di(wt) - di(Bp))))),
        expansion_frac=dict(spearman=float(spearmanr(ef(wt), ef(Bp)).correlation),
                            median_abs_err=float(np.median(np.abs(ef(wt) - ef(Bp)))))))


def score_rpe(pop, B, minutes, n_events):
    tb = pop['rpe_b'] * pop['rpe_amp']
    return dict(population='striatal RPE', minutes=minutes, n_outcomes=int(n_events), scores=dict(
        rpe_beta=dict(spearman=float(spearmanr(tb, B[1]).correlation),
                      median_abs_err=float(np.median(np.abs(B[1] - tb))))))


def redo_rpe_truth(pop):
    """Regenerate striatal data with 1 s outcome events (game runs + reward task)."""
    for i in range(N_GAME):
        lg = np.load(p(f'game{i}_logs.npz'))
        Y = dict(np.load(p(f'game{i}_Y.npz')))
        sh = lg['shots']
        on, rp = M.events_1s(sh[:, 0], np.ones(len(sh)), NF), M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
        rv = M.bold_from_drive(on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None],
                               pop['rpe_hrf_scale'], pop['rpe_amp'])
        Y['rpe_Y'] = M.add_nuisance(np.random.default_rng(1900 + i), rv, vox(pop, 'rpe'), lg['events'])
        np.savez(p(f'game{i}_Y'), **Y)
    ded = dict(np.load(p('ded_other.npz')))
    for j in range(2):
        sh = ded[f'rpe{j}_shots']
        on, rp = M.events_1s(sh[:, 0], np.ones(len(sh)), NF), M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
        rv = M.bold_from_drive(on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None],
                               pop['rpe_hrf_scale'], pop['rpe_amp'])
        ded[f'rpe{j}_Y'] = M.add_nuisance(np.random.default_rng(2300 + j), rv, vox(pop, 'rpe'),
                                          sh[sh[:, 2] > 0][:, [0, 3, 4]])
    np.savez(p('ded_other'), **ded)
    log('striatal data regenerated with 1 s outcome events')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    log('E_SCALE', calibrate())
    pop = populations()
    G = gauss_sets(pop)
    if stage in ('all', 'data'):
        other_dedicated(pop)
        _FIT['pop'], _FIT['G'] = pop, G
        tasks = [('ded', k, j) for k in V1_KINDS for j in range(2)] + [('game', i, 0) for i in range(N_GAME)]
        with Pool(2) as pool:
            pool.map(_data_task, tasks, chunksize=1)
    if stage == 'data3':
        _FIT['pop'], _FIT['G'] = pop, G
        with Pool(2) as pool:
            pool.map(_data_task, [('g3', i, 0) for i in range(N_GAME)], chunksize=1)
    if stage in ('all', 'fit'):
        names = sys.argv[2:] or (['dedicated_' + k for k in V1_KINDS] + ['battery_joint', 'game_1h',
                                                                          'game_2h', 'game_2h_oracle', 'nonv1'])
        run_fits(names)
    log('done')
