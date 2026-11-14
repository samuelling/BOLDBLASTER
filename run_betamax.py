"""Round 8: the same model battery on a real game -- Betamax's beach battle, played forward.

Betamax (samuelling/betamax) is rendered headless by tools/betamax/ (Chrome + Mesa llvmpipe under
Xvfb): FORWARD mode only (time runs in order, kills stay down, no rewind, no taking bodies -- if the
player falls he carries on as the nearest of his side). A scripted bot plays a Cape Cod defender,
restarting at a random defender and tape time whenever the tape ends. Each 10 Hz frame is the
game's own render (VCR filter off) reduced to 192x144 Rec.709 luma on the same 24 x 18 deg display
as the raycaster, plus a per-frame log of every soldier on screen and per-event logs of the bot's
bursts and kills. This script turns those into exactly the per-run files run_v2.py writes for the
raycaster (prefix 'betamax'), then fits the same models.

Stages: data | fit | nonv1 | cst | stats | all
"""
import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

from boldblaster import bands, game, models as M, fit
import run_v2 as V2
import run_experiment as R1
import run_round7 as R7

D, p, log = V2.D, V2.p, V2.log
SRC = os.environ.get('BETAMAX_RUNS', '/home/claude/bbx/runs')
FPS, NF, H, W = game.FPS, V2.NF, game.H, game.W
N_RUNS = 12
NEAR = 60.0          # m: enemies this close count as items (numerosity) and pop out (salience)
PRE = 'betamax'


# ------------------------------------------------------------------ raw harness output -> logs
def load_raw(i):
    recs = [json.loads(l) for l in open(f'{SRC}/{PRE}{i}_log.jsonl')]
    ev = json.load(open(f'{SRC}/{PRE}{i}_events.json'))
    assert len(recs) == NF, (i, len(recs))
    t0 = recs[0]['t']
    fr = lambda t: int(np.clip(round((t - t0) * FPS), 0, NF - 1))
    return recs, ev, t0, fr


def burst_table():
    """All bursts across rendered runs: (log dist, aim error deg, hit). Used to calibrate the
    shooter's expected hit probability, the 'prediction' in the RPE model."""
    rows = []
    for i in range(N_RUNS):
        if not os.path.exists(f'{SRC}/{PRE}{i}_events.json'):
            continue
        for b in json.load(open(f'{SRC}/{PRE}{i}_events.json'))['bursts']:
            rows.append((np.log(max(b[1], 1.0)), b[2], b[6]))
    return np.array(rows, float)


def logistic(X, y, iters=50):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (y - pr)
        Hm = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(Hm, g)
    return w


def p_hit(w, dist, err):
    return float(1 / (1 + np.exp(-(w[0] + w[1] * np.log(max(dist, 1.0)) + w[2] * err))))


def frames(i):
    """(NF, H, W) float32 luma in [0,1] with the raycaster's crosshair drawn in (the game's own
    crosshair is a DOM overlay the canvas capture doesn't include)."""
    L = np.fromfile(f'{SRC}/{PRE}{i}_lum.u16', np.uint16).reshape(NF, H, W).astype(np.float32) / 65535
    c = W // 2
    L[:, H // 2, c - 4:c + 5] = 0.95
    L[:, H // 2 - 4:H // 2 + 5, c] = 0.95
    return L


def build_logs(i, w):
    recs, ev, t0, fr = load_raw(i)
    kills_at = {}
    for t, x, y in ev['kills']:
        kills_at.setdefault(fr(t), (x, y))
    gm = game.GazeModel(np.random.default_rng(5000 + i))
    gaze = np.zeros((NF, 2), np.float32)
    n_items = np.zeros(NF, np.int16)
    salient, last_seen = [], {}
    for f, r in enumerate(recs):
        t = f / FPS
        en = [v for v in r['vis'] if v[5]]
        vis = [(v[0], v[1], v[2], v[3], v[4]) for v in en]
        near = [v for v in en if v[3] and v[2] < NEAR]
        n_items[f] = len(near)
        for v in near:
            if t - last_seen.get(v[4], -1e9) > 1.0:
                salient.append((t, v[0], v[1], 0))
            last_seen[v[4]] = t
        onset = kills_at.get(f)
        if onset is not None:
            salient.append((t, onset[0], onset[1], 1))
        gaze[f] = gm.step(vis, onset)
    events = np.array([(fr(t) / FPS, x, y) for t, x, y in ev['kills']], np.float32).reshape(-1, 3)
    shots = np.array([(fr(b[0]) / FPS, p_hit(w, b[1], b[2]), b[6], b[3], b[4]) for b in ev['bursts']],
                     np.float32).reshape(-1, 5)
    speed = np.array([r['speed'] for r in recs], np.float32)
    yaw = np.gradient(np.unwrap(np.array([r['yaw'] for r in recs]))) * FPS
    # a death (next man up) or a restart is a camera cut, not self-motion: zero both around it
    body = np.array([r['body'] for r in recs])
    tau = np.array([r['tau'] for r in recs])
    cut = np.where((np.diff(body) != 0) | (np.abs(np.diff(tau) - 0.1) > 0.5))[0]
    for c in cut:
        yaw[max(0, c - 1):c + 3] = 0
        speed[max(0, c - 1):c + 3] = 0
    return dict(gaze=gaze, events=events, salient=np.array(salient, np.float32).reshape(-1, 4), shots=shots,
                n_coins=n_items, fwd=(speed * 1.8 / 4.2).astype(np.float32), yaw=yaw.astype(np.float32),
                deaths=np.array(ev['deaths'], np.float32).reshape(-1), cuts=np.array(ev['cuts'], np.float32).reshape(-1, 2),
                tau=np.array([r['tau'] for r in recs], np.float32))


def make_run(i, pop, G, w):
    """Mirror of run_v2.game_run, fed by Betamax instead of the raycaster."""
    if os.path.exists(p(f'{PRE}{i}_Y.npz')):
        return
    Gc_t, Gs_t, Gc_f, Gs_f = G
    lg = build_logs(i, w)
    L = frames(i)
    sink = bands.BandSink()
    for k in range(0, NF, 50):
        sink(L[k:k + 50])
    maps = sink.result()
    del L
    gz = lg['gaze']
    et = R1.eyetracker(gz, np.random.default_rng(700 + i))
    Ec, Es, Tc = bands.pool(maps, gz, Gc_t, Gs_t)
    att_t = M.attention_trace(lg['salient'], gz, pop['v1_x'], pop['v1_y'], sigma=2.0)
    vis = V2.v1_truth_bold(Ec, Es, Tc, att_t, pop)
    V2.save_feats(f'{PRE}{i}_feat_oracle', Ec, Es, Tc, att_t)
    del Ec, Es, Tc
    Ec, Es, Tc = bands.pool(maps, et, Gc_f, Gs_f)
    att_f = M.attention_trace(lg['salient'], et, pop['v1_fx'], pop['v1_fy'], sigma=2.5)
    V2.save_feats(f'{PRE}{i}_feat', Ec, Es, Tc, att_f)
    del Ec, Es, Tc, maps
    ev = lg['events']
    rng = np.random.default_rng(900 + i)
    out = dict(v1_vis=vis, v1_Y=M.add_nuisance(rng, vis, V2.vox(pop, 'v1'), ev))
    nd = M.num_tuning(lg['n_coins'], pop['num_mu'], pop['num_sig'])
    out['num_Y'] = M.add_nuisance(rng, M.bold_from_drive(nd, pop['num_hrf_scale'], pop['num_amp']), V2.vox(pop, 'num'), ev)
    mf = np.c_[lg['fwd'] / 1.8, np.maximum(lg['yaw'], 0), np.maximum(-lg['yaw'], 0)]
    md = mf @ np.stack([pop['mst_wf'], pop['mst_wl'], pop['mst_wr']])
    out['mst_Y'] = M.add_nuisance(rng, M.bold_from_drive(md, pop['mst_hrf_scale'], pop['mst_amp']), V2.vox(pop, 'mst'), ev)
    sh = lg['shots']
    on = M.events_1s(sh[:, 0], np.ones(len(sh)), NF)
    rp = M.events_1s(sh[:, 0], sh[:, 2] - sh[:, 1], NF)
    rd = on[:, None] * pop['rpe_g'][None] + rp[:, None] * pop['rpe_b'][None]
    out['rpe_Y'] = M.add_nuisance(rng, M.bold_from_drive(rd, pop['rpe_hrf_scale'], pop['rpe_amp']), V2.vox(pop, 'rpe'), ev)
    np.savez(p(f'{PRE}{i}_logs'), et=et, **lg)
    np.savez(p(f'{PRE}{i}_Y'), **out)
    log(f'{PRE}{i}: {len(sh)} bursts ({sh[:, 2].mean() if len(sh) else 0:.2f} hit), {len(ev)} kills, '
        f'items/frame {lg["n_coins"].mean():.1f}, deaths {len(lg["deaths"])}')


def data(n):
    pop = V2.populations()
    if os.path.exists(p('betamax_phit.json')):          # calibrated once, then held fixed for every run
        w = np.array(json.load(open(p('betamax_phit.json')))['w'])
    else:
        bt = burst_table()
        w = logistic(bt[:, :2], bt[:, 2])
        json.dump(dict(w=w.tolist(), n=len(bt), hit_rate=float(bt[:, 2].mean())), open(p('betamax_phit.json'), 'w'))
        log('p(hit) model', np.round(w, 3), 'from', len(bt), 'bursts, hit rate', round(float(bt[:, 2].mean()), 2))
    G = V2.gauss_sets(pop)
    for i in range(n):
        if os.path.exists(f'{SRC}/{PRE}{i}_events.json'):
            make_run(i, pop, G, w)


# ------------------------------------------------------------------ V1 joint fits
def fits(n_avail):
    pop = V2.populations()
    out_path = p('results_betamax.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    allp = list(M.PNAMES)
    plan = []
    for mins, n in ((30, 6), (60, 12)):
        if n > n_avail:
            continue
        for oracle in (False, True):
            tag = '_oracle' if oracle else ''
            plan.append((f'bm_{mins}m{tag}', V2.game_specs(range(n), PRE), oracle, mins))
            plan.append((f'rc_{mins}m{tag}', V2.game_specs(range(n), 'game'), oracle, mins))   # raycaster, matched
    for name, specs, oracle, mins in plan:
        if name in res:
            continue
        t0 = time.time()
        est = V2.fit_v1(name, specs(), allp, oracle)
        np.savez(p(f'fit_{name}'), **est)
        res[name] = dict(game='betamax' if name.startswith('bm') else 'raycaster', minutes=mins,
                         cond='oracle' if oracle else 'realistic', scores=V2.score_v1(pop, est, allp),
                         seconds=time.time() - t0)
        json.dump(res, open(out_path, 'w'), indent=1)
        log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()}, f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------ numerosity, self-motion, RPE
def nonv1(n_avail):
    pop = V2.populations()
    out_path = p('results_betamax.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for mins, n in ((30, 6), (60, 12)):
        if n > n_avail:
            continue
        for g, pre in (('bm', PRE), ('rc', 'game')):
            rs = []
            for i in range(n):
                lg, Y = np.load(p(f'{pre}{i}_logs.npz')), np.load(p(f'{pre}{i}_Y.npz'))
                rs.append((lg, Y, fit.nuisance_basis(200, lg['events'], arousal=True)))
            mu, sg = M.fit_numerosity([dict(N=lg['n_coins'], Y=Y['num_Y'], Q=Q) for lg, Y, Q in rs])
            res[f'num_{g}_{mins}m'] = V2.score_num(pop, mu, sg, mins)
            X = [np.stack([M.hrf_regressor(lg['fwd'] / 1.8), M.hrf_regressor(np.maximum(lg['yaw'], 0)),
                           M.hrf_regressor(np.maximum(-lg['yaw'], 0))], 1) for lg, Y, Q in rs]
            B = M.glm(X, [Y['mst_Y'] for _, Y, _ in rs], [Q for _, _, Q in rs])
            res[f'mst_{g}_{mins}m'] = V2.score_mst(pop, B, mins)
            X = [V2.rpe_regs(lg['shots']) for lg, Y, Q in rs]
            B = M.glm(X, [Y['rpe_Y'] for _, Y, _ in rs], [Q for _, _, Q in rs])
            res[f'rpe_{g}_{mins}m'] = V2.score_rpe(pop, B, mins, sum(len(lg['shots']) for lg, _, _ in rs))
            for k in ('num', 'mst', 'rpe'):
                r = res[f'{k}_{g}_{mins}m']
                r['game'] = 'betamax' if g == 'bm' else 'raycaster'
                log(f'{k}_{g}_{mins}m', {kk: round(v['spearman'], 2) for kk, v in r['scores'].items()})
    json.dump(res, open(out_path, 'w'), indent=1)


# ------------------------------------------------------------------ round-7 CST on Betamax
def cst(n_avail):
    pop = R7.patch_population()
    for i in range(n_avail):
        out_path = p(f'{PRE}{i}_Y_cst.npz')
        if os.path.exists(out_path):
            continue
        fe = np.load(p(f'{PRE}{i}_feat_oracle.npz'))
        drive = M.cst_drive(fe['Ec'].astype(np.float32), fe['Es'].astype(np.float32), fe['Tc'].astype(np.float32),
                            mu=pop['v1_mu'], sig=pop['v1_sig'], c50=pop['v1_c50'], q=pop['v1_q'], k=pop['v1_k'],
                            taus=pop['v1_taus'], taut=pop['v1_taut'], ws=pop['v1_ws'])
        vis = M.bold_from_drive(drive, pop['v1_hrf_scale'], pop['v1_amp'])
        lg = np.load(p(f'{PRE}{i}_logs.npz'))
        Y = M.add_nuisance(np.random.default_rng(7700 + i), vis, V2.vox(pop, 'v1'), lg['events'])
        np.savez(out_path, v1_vis=vis, v1_Y=Y)
    out_path = p('results_betamax.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for mins, n in ((30, 6), (60, 12)):
        if n > n_avail:
            continue
        for cond, oracle in (('oracle', True), ('real', False)):
            name = f'cst_bm_{cond}_{mins}m'
            if name in res:
                continue
            t0 = time.time()
            # plug-in tuning from Betamax's own joint fit (the round-7 convention, applied to this game)
            fz = np.load(p(f'fit_bm_{mins}m{"_oracle" if oracle else ""}.npz'))
            runs = []
            for i in range(n):
                fe = np.load(p(f'{PRE}{i}_feat_oracle.npz' if oracle else f'{PRE}{i}_feat.npz'))
                Y = np.load(p(f'{PRE}{i}_Y_cst.npz'))['v1_Y'][:, :R7.NV]
                lg = np.load(p(f'{PRE}{i}_logs.npz'))
                Q = fit.nuisance_basis(Y.shape[0], lg['events'], arousal=len(lg['events']) > 0)
                runs.append(R7._one_run(fe['Ec'][:, :R7.NV], fe['Es'][:, :R7.NV], fe['Tc'][:, :R7.NV], Y, Q,
                                        fz['mu'][:R7.NV], fz['sig'][:R7.NV]))
            est = R7.fit_cst(runs, fz['c50'][:R7.NV], fz['q'][:R7.NV], fz['k'][:R7.NV])
            res[name] = dict(game='betamax', minutes=mins, cond=cond, scores=R7.score_cst(pop, est),
                             seconds=time.time() - t0)
            json.dump(res, open(out_path, 'w'), indent=1)
            log(name, {k: round(v['spearman'], 2) for k, v in res[name]['scores'].items()}, f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------ stimulus statistics, game vs game
def stats(n_avail):
    out = {}
    for g, pre in (('betamax', PRE), ('raycaster', 'game')):
        fe = [np.load(p(f'{pre}{i}_feat_oracle.npz')) for i in range(n_avail)]
        lg = [np.load(p(f'{pre}{i}_logs.npz')) for i in range(n_avail)]
        Ec = np.concatenate([f['Ec'].astype(np.float32) for f in fe])
        Tc = np.concatenate([f['Tc'].astype(np.float32) for f in fe])
        band_mean = Ec.mean((0, 1))                       # (NB,) mean centre energy per SF band
        # per-voxel temporal SD of each band's drive, median over voxels: how much each band varies
        band_var = np.median(Ec.std(0), 0)
        sh = np.concatenate([l['shots'] for l in lg]) if any(len(l['shots']) for l in lg) else np.zeros((0, 5))
        mins = n_avail * 5
        gz = np.concatenate([l['gaze'] for l in lg])
        out[g] = dict(
            band_mean=band_mean.tolist(), band_temporal_sd=band_var.tolist(),
            transient_to_sustained=float(Tc.mean() / Ec.mean()),
            items_mean=float(np.mean(np.concatenate([l['n_coins'] for l in lg]))),
            items_sd=float(np.std(np.concatenate([l['n_coins'] for l in lg]))),
            kills_per_min=float(sum(len(l['events']) for l in lg) / mins),
            bursts_per_min=float(len(sh) / mins), hit_rate=float(sh[:, 2].mean()) if len(sh) else float('nan'),
            salient_per_min=float(sum(len(l['salient']) for l in lg) / mins),
            fwd_mean=float(np.mean(np.concatenate([l['fwd'] for l in lg]))),
            yaw_sd=float(np.std(np.concatenate([l['yaw'] for l in lg]))),
            gaze_sd=[float(gz[:, 0].std()), float(gz[:, 1].std())])
    out['bands_cpd'] = bands.BAND_F.tolist()
    json.dump(out, open(p('results_betamax_stats.json'), 'w'), indent=1)
    for g in ('betamax', 'raycaster'):
        log(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[g].items() if not k.startswith('band')})


def frame_strip(path, n=5):
    """Example frames as the models see them: Betamax (top row) vs the raycaster (bottom row)."""
    from PIL import Image
    L = frames(0)
    lg = np.load(p(f'{PRE}0_logs.npz'))
    busy = np.argsort(-lg['n_coins'][::30])[:n * 3] * 30          # frames with soldiers near, spread out
    pick = np.sort(busy[np.linspace(0, len(busy) - 1, n).astype(int)])
    top = np.concatenate([np.pad(L[f], ((0, 0), (0, 4)), constant_values=1) for f in pick], 1)
    r = game.simulate_run_v2(300, duration=60.0, keep_frames=600)
    F = r['frames']
    bot = np.concatenate([np.pad(F[f], ((0, 0), (0, 4)), constant_values=1) for f in np.linspace(40, 590, n).astype(int)], 1)
    img = np.concatenate([top, np.ones((4, top.shape[1])), bot], 0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)


if __name__ == '__main__':
    V2.calibrate()     # band energy in contrast units, the scale every raycaster run used
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_RUNS
    if stage in ('data', 'all'):
        data(n)
    if stage in ('fit', 'all'):
        fits(n)
    if stage in ('nonv1', 'all'):
        nonv1(n)
    if stage in ('cst', 'all'):
        cst(n)
    if stage in ('stats', 'all'):
        stats(n)
    if stage in ('strip', 'all'):
        frame_strip(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data', 'round8_frames.png'))
