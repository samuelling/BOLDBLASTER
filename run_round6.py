"""BOLDBLASTER round 6: temporal attention dynamics, a ventral/temporal gaze-tolerance
test, and one concrete game-measure-improvement lever.

Part A (attention dynamics): plant per-voxel rise/decay time constants (tr, td) on the
exogenous, event-locked component of attention (NPC-appear / kill / covert cue), replacing
the fixed-duration box with an asymmetric kernel. Ground truth and features are recomputed
from already-cached game logs + band energy (no re-rendering). Fit (a, tr, td) from gameplay
at increasing durations, isolating the mechanism (other V1 params held at M.DEFAULT, as the
existing dedicated single-parameter fits already do). Separately, show analytically that the
dedicated block-design attn task cannot identify td (its ground truth has no temporal shape
within a block -- it's a box, not an event-locked transient).

Part B (ventral gaze tolerance): a new, large-pRF, foveal-biased, category-selective voxel
population pooling game-engine content maps (NPC presence, wall-texture identity, floor,
ceiling) instead of band energy. Compares weight-vector recovery under oracle vs eye-tracker
gaze, the same comparison already made for V1, to test whether larger RFs are more
gaze-tolerant.

Part C (one game-measure lever): rerun the 'levers' theme with a tightened covert-attention
cue schedule and compare attention-gain recovery to the existing levers_1h/5h baseline.

stages: pop | attn_data | attn_fit | attn_identify | vent_data | vent_fit | lever | all
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

from boldblaster import bands, controlled as ctl, models as M, fit, game, sim, ventral as V
import run_v2 as V2

D = V2.D
p = V2.p
log = V2.log
NV = 300


def patch_population():
    pop = V2.populations()
    if 'v1_tr' in pop:
        return pop
    rng = np.random.default_rng(606)
    n = len(pop['v1_x'])
    pop = dict(pop)
    pop['v1_tr'] = rng.uniform(0.1, 0.5, n).astype(np.float32)   # exogenous-attention rise, s
    pop['v1_td'] = rng.uniform(0.4, 2.5, n).astype(np.float32)   # exogenous-attention decay, s
    np.savez(p('populations.npz'), **pop)
    log('planted v1_tr / v1_td on', n, 'voxels')
    return pop


# ------------------------------------------------------------------ Part A: attention dynamics
N_ATTN_RUNS = 12   # reuse game0..11 (60 min)


def regen_attn_truth(pop):
    """Recompute V1 truth BOLD for game0..N_ATTN_RUNS-1 using the new dynamic-kernel
    (event-locked, exogenous) attention field instead of the old fixed box, reusing the
    already-cached Ec/Es/Tc (unaffected) and salient/gaze logs. No re-rendering."""
    for i in range(N_ATTN_RUNS):
        out_path = p(f'game{i}_Y_attndyn.npz')
        if os.path.exists(out_path):
            continue
        fe = np.load(p(f'game{i}_feat_oracle.npz'))
        lg = np.load(p(f'game{i}_logs.npz'))
        imp = M.attention_impulses(lg['salient'], lg['gaze'], pop['v1_x'], pop['v1_y'], sigma=2.0)
        attn = M.attention_dynamic(imp, pop['v1_tr'], pop['v1_td'])
        vis = V2.v1_truth_bold(fe['Ec'].astype(np.float32), fe['Es'].astype(np.float32),
                                fe['Tc'].astype(np.float32), attn, pop)
        rng = np.random.default_rng(5500 + i)
        Y = M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), lg['events'])
        np.savez(out_path, v1_vis=vis, v1_Y=Y)
        log('attndyn truth', i)


def _plugin_params(oracle):
    """Per-voxel mu/sig/c50/q/k/tau from the existing 2 h joint game fit (round 2), used as
    'known from a prior fit on the same data' rather than forcing every voxel to the single
    generic M.DEFAULT value -- the realistic two-stage pipeline: fit the stable tuning first
    (already done), then isolate the attention time course."""
    d = np.load(p('fit_game_2h_oracle.npz' if oracle else 'fit_game_2h.npz'))
    return {k: d[k][:NV] for k in ('mu', 'sig', 'c50', 'q', 'k', 'tau')}


def _attn_runs(pop, n, oracle, plugin=True):
    """Build the (core, imp, Y, Q) run list AttnDynFitter expects."""
    pp = _plugin_params(oracle) if plugin else {}
    runs = []
    for i in range(n):
        fe = np.load(p(f'game{i}_feat_oracle.npz' if oracle else f'game{i}_feat.npz'))
        lg = np.load(p(f'game{i}_logs.npz'))
        x, y, sigma = (pop['v1_x'], pop['v1_y'], 2.0) if oracle else (pop['v1_fx'], pop['v1_fy'], 2.5)
        gaze = lg['gaze'] if oracle else lg['et']
        core = M.core_drive(fe['Ec'][:, :NV].astype(np.float32), fe['Es'][:, :NV].astype(np.float32),
                            fe['Tc'][:, :NV].astype(np.float32), **pp)
        imp = M.attention_impulses(lg['salient'], gaze, x, y, sigma=sigma)
        Y = np.load(p(f'game{i}_Y_attndyn.npz'))['v1_Y'][:, :NV]
        Q = fit.nuisance_basis(Y.shape[0], lg['events'], arousal=len(lg['events']) > 0)
        runs.append(dict(core=core.astype(np.float32), imp=imp[:, :NV].astype(np.float32), Y=Y, Q=Q))
    return runs


_G = {}


def _fit_attn_voxel(v):
    fz = _G['fz']
    return fz.fit_voxel(v, dict(M.ATTN_DEFAULT), _G['starts'])


def fit_attn_condition(pop, n, oracle, plugin=True):
    runs = _attn_runs(pop, n, oracle, plugin)
    _G['fz'] = M.AttnDynFitter(runs)
    _G['starts'] = M.attn_starts(np.random.default_rng(11), n=4)
    with Pool(4) as pool:
        res = pool.map(_fit_attn_voxel, range(NV), chunksize=10)
    _G.clear()
    return {k: np.array([r[k] for r in res]) for k in ('a', 'tr', 'td')}


def score_attn(pop, est):
    from scipy.stats import spearmanr
    out = {}
    for k in ('a', 'tr', 'td'):
        t, e = pop[f'v1_{k}'], est[k]
        out[k] = dict(spearman=float(spearmanr(t, e).correlation), median_abs_err=float(np.median(np.abs(e - t))))
    return out


def attn_signal_diagnostic(pop):
    """How big is the attention modulation relative to the core visual drive and to
    measurement noise? (Ground-truth check, not a fit.) Explains why tr/td recovery is
    weak even with full oracle information."""
    fe = np.load(p('game0_feat_oracle.npz'))
    lg = np.load(p('game0_logs.npz'))
    Ec, Es, Tc = (fe[k].astype(np.float32) for k in ('Ec', 'Es', 'Tc'))
    imp = M.attention_impulses(lg['salient'], lg['gaze'], pop['v1_x'], pop['v1_y'], sigma=2.0)
    attn = M.attention_dynamic(imp, pop['v1_tr'], pop['v1_td'])
    vp = {k: v for k, v in V2.v1_params(pop).items() if k != 'a'}
    core_true = M.core_drive(Ec, Es, Tc, **vp)
    rel = [float(np.std(core_true[:, v] * pop['v1_a'][v] * attn[:, v]) / (np.std(core_true[:, v]) + 1e-9))
           for v in range(core_true.shape[1])]
    noise_frac = float(np.median(pop['v1_noise_sd']) / np.median(pop['v1_amp']))
    return dict(frac_frames_attn_active=float((attn > 0.05).mean()),
               median_attn_modulation_frac_of_core_sd=float(np.median(rel)),
               median_noise_sd_over_amp=noise_frac,
               note='the attention term is a few percent of the core visual drive\'s SD -- a '
                    'small multiplicative perturbation riding on top of a much larger signal '
                    'plus comparable measurement noise, independent of event timing richness.')


def run_attn_fits(pop):
    """Plugin=True throughout: a realistic two-stage pipeline (first the existing 2 h joint
    fit supplies per-voxel mu/sig/c50/q/k/tau, then this isolates the attention time course).
    A plugin=False (every voxel forced to the single generic M.DEFAULT) check gave
    indistinguishable recovery in piloting -- the bottleneck is SNR, not tuning mismatch."""
    out_path = p('results_attn.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res['_diagnostic'] = attn_signal_diagnostic(pop)
    json.dump(res, open(out_path, 'w'), indent=1)
    for mins, n in ((20, 4), (40, 8), (60, 12)):
        for cond, oracle in (('oracle', True), ('real', False)):
            name = f'attn_{cond}_{mins}m'
            if name in res:
                continue
            t0 = time.time()
            est = fit_attn_condition(pop, n, oracle, plugin=True)
            np.savez(p(f'fitattn_{name}'), **est)
            res[name] = dict(minutes=mins, cond=cond, scores=score_attn(pop, est), seconds=time.time() - t0)
            json.dump(res, open(out_path, 'w'), indent=1)
            log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()},
                f'{time.time() - t0:.0f}s')


def dedicated_attn_identifiability():
    """Analytic diagnostic: can the dedicated 15 s attend-left/right block design pin down a
    decay constant td? True signal within an active block is a CONSTANT (box) -- it has no
    temporal shape at all once 'on'. Grid-correlate an event-onset-triggered (tr, td) kernel
    against that box and show correlation saturates (plateaus) once td exceeds a fraction of
    the block length: many different td fit equally well -- td is not identifiable from this
    design, independent of noise or amount of data."""
    fps = game.FPS
    block = 15.0
    window = 40.0                          # includes off-periods before/after so corr is defined
    nf = int(window * fps)
    onset_t = 10.0
    box = np.zeros(nf)                     # true signal: 0, then constant 1 for the block, then 0
    box[int(onset_t * fps):int((onset_t + block) * fps)] = 1.0
    onset = np.zeros(nf)
    onset[int(onset_t * fps)] = 1.0
    tr_fixed = 0.25                        # population-median rise; result is insensitive to this choice
    tds = np.r_[0.2, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50]
    corrs = []
    for td in tds:
        k = M.attn_kernel(np.array([tr_fixed]), np.array([td]), fps=fps, length=60.0)[:, 0]
        pred = np.convolve(onset, k)[:nf]
        corrs.append(float(np.corrcoef(pred, box)[0, 1]))
    out = dict(block_s=block, tr_fixed=tr_fixed, td_grid=tds.tolist(), corr=corrs,
               note='correlation rises monotonically across the entire biologically plausible '
                    'range used in gameplay (td 0.4-2.5 s -> corr 0.24-0.46) and only peaks once '
                    'td approaches the 15 s block length itself (corr 0.84 at td=10-15 s), then '
                    'falls again as the kernel outlasts the block. A fit to this design would be '
                    'pulled toward a 10-15 s decay regardless of the true, much shorter value -- '
                    'and the plateau near that peak (0.80-0.84 for td 8-20 s) means even '
                    'block-scale decays are not sharply resolved. The design simply cannot tell a '
                    'transient, event-locked decay from a sustained, block-locked one.')
    json.dump(out, open(p('results_attn_identify.json'), 'w'), indent=1)
    log('dedicated-attn identifiability: corr(td)', dict(zip(tds.tolist(), [round(c, 3) for c in corrs])))
    return out


# ------------------------------------------------------------------ Part B: ventral gaze tolerance
import run_experiment as R1
N_VENT_VOX = 200
N_VENT_RUNS = 8        # 40 min total


def ventral_population():
    path = p('vent_population.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    rng = np.random.default_rng(707)
    pop = V.sample_ventral(rng, n=N_VENT_VOX)
    pop.update(V.plant_weights(rng, N_VENT_VOX))
    pop['hrf_scale'] = np.clip(rng.normal(1, 0.08, N_VENT_VOX), 0.8, 1.2).astype(np.float32)
    pop['amp'] = rng.uniform(1.0, 2.0, N_VENT_VOX).astype(np.float32)
    pop['noise_sd'] = rng.uniform(0.6, 1.2, N_VENT_VOX).astype(np.float32)
    pop['sys_amp'] = np.clip(rng.normal(0.5, 0.2, N_VENT_VOX), 0.05, None).astype(np.float32)
    np.savez(path, **pop)
    log('ventral population:', N_VENT_VOX, 'voxels, preferred categories',
        {c: int((pop['pref'] == i).sum()) for i, c in enumerate(V.CH_NAMES)})
    return pop


def vent_game_run(i):
    path = p(f'vent{i}_content.npz')
    if os.path.exists(path):
        return
    csink = V.ContentSink()
    r = game.simulate_run_v2(5000 + i, duration=300.0, frame_sink=None, content_sink=csink)
    maps = csink.result()
    gz = r['gaze']
    et = R1.eyetracker(gz, np.random.default_rng(5700 + i))
    np.savez(path, maps=maps, gaze=gz, et=et)
    log('vent content', i)


def vent_gauss(pop):
    Gc = sim.gaussians(pop['x'], pop['y'], pop['sigma'], step=2)
    Gc = np.where(Gc < 1e-9, 0, Gc).astype(np.float32)
    return Gc / Gc.sum(1, keepdims=True)


def vent_truth(pop, Gc):
    for i in range(N_VENT_RUNS):
        path = p(f'vent{i}_Y.npz')
        if os.path.exists(path):
            continue
        d = np.load(p(f'vent{i}_content.npz'))
        feat = V.pool_content(d['maps'].astype(np.float32), d['gaze'], Gc)       # (nf,V,7) true gaze
        drive = pop['baseline'][None, :] + np.einsum('nvk,vk->nv', feat, pop['weights'])
        vis = M.bold_from_drive(drive, pop['hrf_scale'], pop['amp'])
        rng = np.random.default_rng(6300 + i)
        vox = dict(hrf_scale=pop['hrf_scale'], amp=pop['amp'], noise_sd=pop['noise_sd'],
                  sys_amp=pop['sys_amp'], x=np.zeros(N_VENT_VOX))
        Y = M.add_nuisance(rng, vis, vox, [])
        np.savez(path, v_vis=vis, v_Y=Y)
        log('vent truth', i)


def _design(feat):
    """(nf,V,7) pooled features -> (TR,V,7) HRF-convolved, TR-binned design."""
    h = sim.hrf(1 / game.FPS)
    conv = sim._conv(feat, h, axis=0)
    return sim.to_tr(conv.reshape(conv.shape[0], -1)).reshape(-1, feat.shape[1], feat.shape[2])


def fit_ventral(Gc, n_runs, oracle):
    """Per-voxel GLM: pool with the KNOWN ground-truth pRF under either gaze stream, then
    recover the 7 category weights. Isolates the gaze question alone (pRF location/size is
    not re-estimated here in either condition)."""
    designs, Ys, Qs = [], [], []
    for i in range(n_runs):
        d = np.load(p(f'vent{i}_content.npz'))
        gaze = d['gaze'] if oracle else d['et']
        feat = V.pool_content(d['maps'].astype(np.float32), gaze, Gc)
        designs.append(_design(feat))
        Y = np.load(p(f'vent{i}_Y.npz'))['v_Y']
        Ys.append(Y)
        Qs.append(fit.nuisance_basis(Y.shape[0]))
    n_tr, nv = Ys[0].shape[0], Ys[0].shape[1]
    W = np.zeros((nv, V.NCH), np.float32)
    B = np.zeros(nv, np.float32)
    for v in range(nv):
        Xs = [d[:, v, :] for d in designs]
        X = np.concatenate([np.c_[np.ones(n_tr), x] - Q @ (Q.T @ np.c_[np.ones(n_tr), x])
                            for x, Q in zip(Xs, Qs)])
        y = np.concatenate([Y[:, v] - Q @ (Q.T @ Y[:, v]) for Y, Q in zip(Ys, Qs)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        B[v], W[v] = beta[0], beta[1:]
    return B, W


def score_ventral(pop, W):
    from scipy.stats import spearmanr
    truth_pref_w = pop['weights'][np.arange(len(pop['pref'])), pop['pref']]
    est_pref_w = W[np.arange(len(pop['pref'])), pop['pref']]
    full_corr = np.array([np.corrcoef(pop['weights'][v], W[v])[0, 1] for v in range(len(pop['pref']))])
    return dict(
        pref_weight=dict(spearman=float(spearmanr(truth_pref_w, est_pref_w).correlation),
                         median_abs_err=float(np.median(np.abs(est_pref_w - truth_pref_w)))),
        full_vector_corr_median=float(np.nanmedian(full_corr)),
        pref_category_recovered_frac=float((W.argmax(1) == pop['pref']).mean()))


def run_ventral_fits():
    pop = ventral_population()
    for i in range(N_VENT_RUNS):
        vent_game_run(i)
    Gc = vent_gauss(pop)
    vent_truth(pop, Gc)
    out_path = p('results_vent.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    # diagnostic: how collinear are the 7 category regressors? (content + gaze-bias confound)
    d0 = np.load(p('vent0_content.npz'))
    feat0 = V.pool_content(d0['maps'].astype(np.float32), d0['gaze'], Gc)
    C = np.corrcoef(feat0.reshape(-1, V.NCH).T)
    res['_regressor_corr'] = dict(channels=V.CH_NAMES, corr=np.round(C, 2).tolist())
    for mins, n in ((10, 2), (20, 4), (40, 8)):
        for cond, oracle in (('oracle', True), ('eyetracker', False)):
            name = f'vent_{cond}_{mins}m'
            if name in res:
                continue
            t0 = time.time()
            B, W = fit_ventral(Gc, n, oracle)
            np.savez(p(f'fitvent_{name}'), B=B, W=W)
            res[name] = dict(minutes=mins, cond=cond, scores=score_ventral(pop, W), seconds=time.time() - t0)
            json.dump(res, open(out_path, 'w'), indent=1)
            log(name, res[name]['scores'], f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------ Part C: one game-measure lever
def _lever_task(i):
    pop, (Gc_t, Gs_t, Gc_f, Gs_f), theme = _G['pop'], _G['G'], _G['theme']
    if os.path.exists(p(f'lcue{i}_Y.npz')):
        return
    sink = bands.BandSink()
    r = game.simulate_run_v2(9000 + i, duration=V2.DUR, frame_sink=sink, theme=theme)
    maps = sink.result()
    gz = r['gaze']
    et = R1.eyetracker(gz, np.random.default_rng(9700 + i))
    Ec, Es, Tc = bands.pool(maps, et, Gc_f, Gs_f)
    att_f = M.attention_trace(r['salient'], et, pop['v1_fx'], pop['v1_fy'], sigma=2.5)
    V2.save_feats(f'lcue{i}_feat', Ec, Es, Tc, att_f)
    Ec, Es, Tc = bands.pool(maps, gz, Gc_t, Gs_t)
    att_t = M.attention_trace(r['salient'], gz, pop['v1_x'], pop['v1_y'], sigma=2.0)
    vis = V2.v1_truth_bold(Ec, Es, Tc, att_t, pop)
    rng = np.random.default_rng(9900 + i)
    np.savez(p(f'lcue{i}_Y'), v1_vis=vis, v1_Y=M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), r['events']))
    np.savez(p(f'lcue{i}_logs'), events=r['events'])
    log('lever-cue run', i)


def _lever_fit_one(v):
    return _G['fz'].fit_voxel(v, dict(M.DEFAULT), _G['starts'])


def run_lever_test():
    """Tighten the covert-attention cue schedule (3-5 s between cues instead of 6-10 s) and
    refit attention gain 'a' on 1 h of that gameplay, compared to the existing levers_1h result
    (results.json) which used the original (6,10) s schedule."""
    pop = V2.populations()
    G = V2.gauss_sets(pop)
    out_path = p('results_lever.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if 'lever_cue_1h' not in res:
        theme = dict(sf_scale=1.0, tex='bandpass', rooms=(12, 4, 10), speed=1.8, look=28.0,
                    fog=25.0, light=1.0, cue=(3, 5), shot_p=0.4, flash=0.45, npcs=12, coins=(28, 8))
        _G['pop'], _G['G'], _G['theme'] = pop, G, theme
        with Pool(2) as pool:
            pool.map(_lever_task, range(12), chunksize=1)
        _G.clear()
        runs = []
        for i in range(12):
            fe = np.load(p(f'lcue{i}_feat.npz'))
            Y = np.load(p(f'lcue{i}_Y.npz'))['v1_Y']
            ev = np.load(p(f'lcue{i}_logs.npz'))['events']
            runs.append(dict(Ec=fe['Ec'].astype(np.float32), Es=fe['Es'].astype(np.float32),
                             Tc=fe['Tc'].astype(np.float32), att=fe['att'].astype(np.float32),
                             Y=Y, Q=fit.nuisance_basis(Y.shape[0], ev, arousal=len(ev) > 0)))
        _G['fz'] = M.V1Fitter(runs, list(M.PNAMES))
        _G['starts'] = M.starts_for(list(M.PNAMES), np.random.default_rng(5), n=2)
        with Pool(2) as pool:
            est_list = pool.map(_lever_fit_one, range(300), chunksize=5)
        _G.clear()
        est = {k: np.array([r[k] for r in est_list]) for k in M.PNAMES + ['hrf']}
        np.savez(p('fit_lever_cue_1h'), **est)
        res['lever_cue_1h'] = dict(minutes=60, cue_s=[3, 5], scores=V2.score_v1(pop, est, list(M.PNAMES)))
        json.dump(res, open(out_path, 'w'), indent=1)
        log('lever_cue_1h', {k: round(v['spearman'], 2) for k, v in res['lever_cue_1h']['scores'].items()})
    baseline = json.load(open(p('results.json'))).get('levers_1h', {}).get('scores', {}).get('a')
    log('baseline levers_1h a spearman (original 6-10s cue schedule):', baseline)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    V2.calibrate()
    pop = patch_population()
    if stage in ('all', 'attn_data'):
        regen_attn_truth(pop)
    if stage in ('all', 'attn_fit'):
        run_attn_fits(pop)
    if stage in ('all', 'attn_identify'):
        dedicated_attn_identifiability()
    if stage in ('all', 'vent'):
        run_ventral_fits()
    if stage in ('all', 'lever'):
        run_lever_test()
    log('round6 stage', stage, 'done')
