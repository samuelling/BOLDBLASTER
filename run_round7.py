"""BOLDBLASTER round 7: compressive spatiotemporal (CST) pRF, after Kim/Kupers/Grill-Spector
(Kim, Kupers, Lerma-Usabiaga & Grill-Spector 2023 J Neurosci; Kupers, Kim & Grill-Spector 2024
Nat Commun).

Replaces the single instantaneous tau*Dt^q term in the V1 drive with two channels that each
carry their own memory: a sustained channel (causal low-pass of the centre drive Dc, time
constant taus) and a transient channel (causal low-pass of the frame-difference drive Dt,
time constant taut), mixed by a per-voxel weight ws (models.cst_drive / models.CSTFitter).
Ground truth and gameplay features reuse the already-cached game logs + band energy (no
re-rendering), exactly as round 6 did for attention dynamics. Isolates (taus, taut, ws) with
mu/sig/c50/q/k plugged in from the existing 2h joint fit and attention switched off -- the same
"one model at a time, generic elsewhere" convention as round 6's attention-dynamics test.

A new dedicated SEQ-SIM design (controlled.run_seqsim_experiment) renders two spatially-
separate probes shown sequentially vs simultaneously at several SOAs -- the fair,
timing-controlled counterpart to gameplay's uncontrolled transient statistics, mirroring
Kim/Kupers' own real-data paradigm -- and is rendered + fit the same way.

stages: pop | cst_data | cst_fit | seqsim_data | seqsim_fit | diagnostic | all
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

from boldblaster import bands, controlled as ctl, models as M, fit
import run_v2 as V2
import run_round6 as R6

D = V2.D
p = V2.p
log = V2.log
NV = 300


def patch_population():
    pop = V2.populations()
    if 'v1_taus' in pop:
        return pop
    rng = np.random.default_rng(717)
    n = len(pop['v1_x'])
    pop = dict(pop)
    pop['v1_taus'] = rng.uniform(0.03, 0.4, n).astype(np.float32)    # sustained-channel time constant, s
    pop['v1_taut'] = rng.uniform(0.03, 0.4, n).astype(np.float32)    # transient-channel time constant, s
    pop['v1_ws'] = rng.uniform(0.15, 0.85, n).astype(np.float32)     # sustained-channel mix weight
    np.savez(p('populations.npz'), **pop)
    log('planted v1_taus / v1_taut / v1_ws on', n, 'voxels')
    return pop


# ------------------------------------------------------------------ Part A: gameplay recoverability
N_CST_RUNS = 12   # reuse game0..11 (60 min)


def regen_cst_truth(pop):
    """Recompute V1 truth BOLD for game0..N_CST_RUNS-1 using the two-channel CST drive (no
    attention term -- that mechanism is characterized separately in round 6), reusing the
    already-cached oracle Ec/Es/Tc. No re-rendering."""
    for i in range(N_CST_RUNS):
        out_path = p(f'game{i}_Y_cst.npz')
        if os.path.exists(out_path):
            continue
        fe = np.load(p(f'game{i}_feat_oracle.npz'))
        drive = M.cst_drive(fe['Ec'].astype(np.float32), fe['Es'].astype(np.float32), fe['Tc'].astype(np.float32),
                            mu=pop['v1_mu'], sig=pop['v1_sig'], c50=pop['v1_c50'], q=pop['v1_q'], k=pop['v1_k'],
                            taus=pop['v1_taus'], taut=pop['v1_taut'], ws=pop['v1_ws'])
        vis = M.bold_from_drive(drive, pop['v1_hrf_scale'], pop['v1_amp'])
        lg = np.load(p(f'game{i}_logs.npz'))
        rng = np.random.default_rng(7700 + i)
        Y = M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), lg['events'])
        np.savez(out_path, v1_vis=vis, v1_Y=Y)
        log('cst truth', i)


def _one_run(Ec, Es, Tc, Y, Q, mu, sig):
    Dc, Ds, Dt = M.core_dcs(Ec.astype(np.float32), Es.astype(np.float32), Tc.astype(np.float32), mu, sig)
    return dict(Dc=Dc, Ds=Ds, Dt=Dt, Y=Y, Q=Q)


def gameplay_cst_runs(n, oracle, mu, sig):
    runs = []
    for i in range(n):
        fe = np.load(p(f'game{i}_feat_oracle.npz' if oracle else f'game{i}_feat.npz'))
        Y = np.load(p(f'game{i}_Y_cst.npz'))['v1_Y'][:, :NV]
        lg = np.load(p(f'game{i}_logs.npz'))
        Q = fit.nuisance_basis(Y.shape[0], lg['events'], arousal=len(lg['events']) > 0)
        runs.append(_one_run(fe['Ec'][:, :NV], fe['Es'][:, :NV], fe['Tc'][:, :NV], Y, Q, mu, sig))
    return runs


_G = {}


def _fit_cst_voxel(v):
    fz = _G['fz']
    fixed = dict(c50=_G['c50'][v], q=_G['q'][v], k=_G['k'][v])
    return fz.fit_voxel(v, fixed, _G['starts'])


def fit_cst(runs, c50, q, k):
    _G['fz'] = M.CSTFitter(runs)
    _G['starts'] = M.cst_starts(np.random.default_rng(13), n=4)
    _G['c50'], _G['q'], _G['k'] = c50, q, k
    with Pool(4) as pool:
        res = pool.map(_fit_cst_voxel, range(NV), chunksize=10)
    _G.clear()
    return {kk: np.array([r[kk] for r in res]) for kk in ('taus', 'taut', 'ws')}


def score_cst(pop, est):
    from scipy.stats import spearmanr
    out = {}
    for k in ('taus', 'taut', 'ws'):
        t, e = pop[f'v1_{k}'][:NV], est[k]
        out[k] = dict(spearman=float(spearmanr(t, e).correlation), median_abs_err=float(np.median(np.abs(e - t))))
    return out


def run_cst_fits(pop):
    out_path = p('results_cst.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for mins, n in ((20, 4), (40, 8), (60, 12)):
        for cond, oracle in (('oracle', True), ('real', False)):
            name = f'cst_{cond}_{mins}m'
            if name in res:
                continue
            t0 = time.time()
            pp = R6._plugin_params(oracle)
            runs = gameplay_cst_runs(n, oracle, pp['mu'], pp['sig'])
            est = fit_cst(runs, pp['c50'], pp['q'], pp['k'])
            np.savez(p(f'fitcst_{name}'), **est)
            res[name] = dict(minutes=mins, cond=cond, scores=score_cst(pop, est), seconds=time.time() - t0)
            json.dump(res, open(out_path, 'w'), indent=1)
            log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()},
                f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------ Part B: SEQ-SIM controlled design
N_SEQSIM_RUNS = 12   # 60 min total


def seqsim_data(pop):
    Gc_t, Gs_t, _, _ = V2.gauss_sets(pop)
    for i in range(N_SEQSIM_RUNS):
        path = p(f'seqsim{i}_Y.npz')
        if os.path.exists(path):
            continue
        sink = bands.BandSink()
        r = ctl.run_seqsim_experiment(2000 + i, duration=300.0, frame_sink=sink)
        maps = sink.result()
        Ec, Es, Tc = bands.pool(maps, r['gaze'], Gc_t, Gs_t)
        drive = M.cst_drive(Ec, Es, Tc, mu=pop['v1_mu'], sig=pop['v1_sig'], c50=pop['v1_c50'], q=pop['v1_q'],
                            k=pop['v1_k'], taus=pop['v1_taus'], taut=pop['v1_taut'], ws=pop['v1_ws'])
        vis = M.bold_from_drive(drive, pop['v1_hrf_scale'], pop['v1_amp'])
        rng = np.random.default_rng(2500 + i)
        Y = M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), [])
        np.savez(path, v1_vis=vis, v1_Y=Y)
        np.savez(p(f'seqsim{i}_feat'), Ec=Ec.astype(np.float16), Es=Es.astype(np.float16), Tc=Tc.astype(np.float16))
        log('seqsim data', i)


def seqsim_cst_runs(n, mu, sig):
    runs = []
    for i in range(n):
        fe = np.load(p(f'seqsim{i}_feat.npz'))
        Y = np.load(p(f'seqsim{i}_Y.npz'))['v1_Y'][:, :NV]
        Q = fit.nuisance_basis(Y.shape[0])
        runs.append(_one_run(fe['Ec'][:, :NV], fe['Es'][:, :NV], fe['Tc'][:, :NV], Y, Q, mu, sig))
    return runs


def run_seqsim_fits(pop):
    seqsim_data(pop)
    out_path = p('results_cst_seqsim.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    pp = R6._plugin_params(True)     # same plugin tuning as oracle gameplay, for a fair comparison
    for mins, n in ((20, 4), (40, 8), (60, 12)):
        name = f'cst_seqsim_{mins}m'
        if name in res:
            continue
        t0 = time.time()
        runs = seqsim_cst_runs(n, pp['mu'], pp['sig'])
        est = fit_cst(runs, pp['c50'], pp['q'], pp['k'])
        np.savez(p(f'fitcst_{name}'), **est)
        res[name] = dict(minutes=mins, cond='seqsim', scores=score_cst(pop, est), seconds=time.time() - t0)
        json.dump(res, open(out_path, 'w'), indent=1)
        log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()}, f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------ diagnostic
def channel_collinearity_diagnostic(pop):
    """Ground-truth check, not a fit: how correlated are the sustained and transient
    channels (at the population-median taus/taut) under each design? ws trades off against
    the overall response gain whenever the two channels move together over time; gameplay's
    uncontrolled, content-driven transients ride on top of whatever is already sustained-
    visible at that same moment, while SEQ-SIM's sequential-vs-simultaneous manipulation is
    built, by construction, to decouple them."""
    taus_med = float(np.median(pop['v1_taus']))
    taut_med = float(np.median(pop['v1_taut']))

    def corr_median(Ec, Es, Tc):
        Dc, _, Dt = M.core_dcs(Ec.astype(np.float32), Es.astype(np.float32), Tc.astype(np.float32))
        Sc = M.causal_filter(Dc, np.full(Dc.shape[1], taus_med))
        Tt = M.causal_filter(Dt, np.full(Dt.shape[1], taut_med))
        cs = [np.corrcoef(Sc[:, v], Tt[:, v])[0, 1] for v in range(Dc.shape[1])
              if np.std(Sc[:, v]) > 1e-9 and np.std(Tt[:, v]) > 1e-9]
        return float(np.nanmedian(cs))

    fe = np.load(p('game0_feat_oracle.npz'))
    game_corr = corr_median(fe['Ec'], fe['Es'], fe['Tc'])
    seqsim_corr = None
    if os.path.exists(p('seqsim0_feat.npz')):
        fs = np.load(p('seqsim0_feat.npz'))
        seqsim_corr = corr_median(fs['Ec'], fs['Es'], fs['Tc'])
    out = dict(taus_median=taus_med, taut_median=taut_med,
               sustained_transient_corr_gameplay=game_corr,
               sustained_transient_corr_seqsim=seqsim_corr,
               note='median-voxel correlation between the sustained channel (filtered Dc) and '
                    'the transient channel (filtered Dt) over time, at the population-median '
                    'time constants. A high correlation means the two channels are nearly '
                    'collinear over the run -- ws (their relative weight) and the overall '
                    'response gain become hard to tell apart, independent of noise or amount '
                    'of data; a low correlation means the design itself separates them.')
    json.dump(out, open(p('results_cst_diagnostic.json'), 'w'), indent=1)
    log('CST channel collinearity: gameplay', round(game_corr, 3), 'seqsim', seqsim_corr)
    return out


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    V2.calibrate()
    pop = patch_population()
    if stage in ('all', 'cst_data'):
        regen_cst_truth(pop)
    if stage in ('all', 'cst_fit'):
        run_cst_fits(pop)
    if stage in ('all', 'seqsim_data'):
        seqsim_data(pop)
    if stage in ('all', 'seqsim_fit'):
        run_seqsim_fits(pop)
    if stage in ('all', 'diagnostic'):
        channel_collinearity_diagnostic(pop)
    log('round7 stage', stage, 'done')
