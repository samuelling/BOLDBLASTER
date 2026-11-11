"""Export all BOLDBLASTER results as tidy CSVs for the dashboard (dashboard_data/)."""
import csv
import json
import os
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'dashboard_data')
os.makedirs(OUT, exist_ok=True)
D1, D2, DL = (os.path.join(ROOT, d) for d in ('data', 'data_v2', 'data_long'))
PARAMS = ['mu', 'sig', 'c50_eff', 'q', 'k', 'tau', 'a']
PLAB = dict(mu='pSFT peak', sig='pSFT bandwidth', c50_eff='pCRF effective c50', q='pCRF exponent',
            k='Surround suppression k', tau='Transient weight', a='Attention gain')
DED = dict(mu='psft', sig='psft', c50_eff='crf', q='crf', k='size', tau='temporal', a='attn')
V1_DESIGNS = [  # (round, key, label, minutes)
    (2, 'dedicated', 'Short dedicated protocol (10 min)', 10), (2, 'battery_joint', 'V1 battery, joint fit', 50),
    (2, 'game_1h', 'Game 1 h', 60), (2, 'game_2h', 'Game 2 h', 120),
    (2, 'game_2h_oracle', 'Game 2 h, oracle gaze + pRF', 120),
    (3, 'levers_1h', 'Game + levers 1 h', 60), (3, 'levers_2h', 'Game + levers 2 h', 120),
    (3, 'levers_2h_oracle', 'Game + levers 2 h, oracle', 120)]
LONG_DESIGNS = [(4, 'dedicated_10min', 'Short dedicated protocol (10 min), same 120 voxels', 10),
                (4, 'long_2h', 'Varied levels 2 h', 120), (4, 'long_5h', 'Varied levels 5 h', 300),
                (4, 'long_10h', 'Varied levels 10 h', 600), (4, 'long_20h', 'Varied levels 20 h', 1200),
                (4, 'single_4h', 'One level type 4 h', 240), (4, 'varied_4h', 'Ten level types 4 h', 240)]


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.4g}' if isinstance(x, (float, np.floating)) else x for x in r])
    print(name, len(rows), 'rows')


def vals(est, p):
    if p == 'c50_eff':
        return est['c50'] / (1 + est['k']) ** (1 / est['q'])
    return est[p]


def boot_ci(t, e, n=600, seed=0):
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        b = rng.integers(0, len(t), len(t))
        bs.append(spearmanr(t[b], e[b]).correlation)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    pop = dict(np.load(os.path.join(D2, 'populations.npz')))
    truth_all = {p: vals({k: pop[f'v1_{k}'] for k in ('c50', 'k', 'q', 'mu', 'sig', 'tau', 'a')}, p) for p in PARAMS}
    ecc, pa = pop['v1_ecc'], np.degrees(np.arctan2(pop['v1_y'], pop['v1_x'])) % 360
    rec, vox = [], []
    # ---- rounds 2-3, V1 (300 voxels)
    for rnd, key, lab, mins in V1_DESIGNS:
        for p in PARAMS:
            f = os.path.join(D2, f'fit_{"dedicated_" + DED[p] if key == "dedicated" else key}.npz')
            est = dict(np.load(f))
            e, t = vals(est, p), truth_all[p]
            r = spearmanr(t, e).correlation
            lo, hi = boot_ci(t, e)
            mae = np.median(np.abs(np.log2(e / t))) if p == 'c50_eff' else np.median(np.abs(e - t))
            rec.append([rnd, key, lab, 'V1', p, PLAB[p], mins, len(t), r, lo, hi, mae])
            for v in range(len(t)):
                vox.append([rnd, key, p, v, ecc[v], pa[v], t[v], e[v]])
    # ---- round 4 (120 voxels)
    S = json.load(open(os.path.join(DL, 'summary.json')))
    n4 = 120
    for rnd, key, lab, mins in LONG_DESIGNS:
        for p in PARAMS:
            x = S[key][p]
            rec.append([rnd, key, lab, 'V1', p, PLAB[p], mins, n4, x['r'], x['lo'], x['hi'], ''])
            if key == 'dedicated_10min':
                continue
            est = dict(np.load(os.path.join(DL, f'fit_{key}.npz')))
            e = vals(est, p)
            for v in range(n4):
                vox.append([rnd, key, p, v, ecc[v], pa[v], truth_all[p][v], e[v]])
    # ---- round 2 non-V1 populations
    r2 = json.load(open(os.path.join(D2, 'results.json')))
    NON = [('num', 'Numerosity', [('mu', 'Numerosity preference'), ('sig', 'Numerosity tuning width')]),
           ('mst', 'Self-motion', [('rotation_pref', 'Rotation preference'), ('expansion_frac', 'Expansion share')]),
           ('rpe', 'Striatal RPE', [('rpe_beta', 'RPE weight')])]
    for pre, popname, prms in NON:
        for suf, lab in (('dedicated', 'Short dedicated protocol (10 min)'), ('game_1h', 'Game 1 h'), ('game_2h', 'Game 2 h')):
            res = r2[f'{pre}_{suf}']
            for p, plab in prms:
                sc = res['scores'][p]
                rec.append([2, 'dedicated' if suf == 'dedicated' else suf, lab, popname, f'{pre}_{p}', plab,
                            res['minutes'], {'num': 150, 'mst': 150, 'rpe': 100}[pre], sc['spearman'], '', '',
                            sc['median_abs_err']])
    # ---- round 5: realistic dedicated protocols (first 120 voxels, from export_fair.py -> fair.csv)
    import csv as _csv
    R5 = dict(ded_30m=('dedicated_30m', 'Dedicated experiment (30 min)'), ded_60m=('dedicated_60m', 'Dedicated experiment (60 min)'),
              battery_60m=('battery_5h', 'Full dedicated battery, joint (5 h)'))
    POPN = dict(num='Numerosity', mst='Self-motion', rpe='Striatal RPE')
    fair = list(_csv.DictReader(open(os.path.join(OUT, 'fair.csv'))))
    for r in fair:
        if r['design'] not in R5:
            continue
        key, lab = R5[r['design']]
        popname = 'V1' if r['parameter'] in PARAMS else POPN[r['parameter'].split('_')[0]]
        rec.append([5, key, lab, popname, r['parameter'], r['parameter_label'], int(float(r['minutes'])),
                    int(float(r['n_voxels'])), float(r['spearman']), r['ci_lo'], r['ci_hi'], ''])
    for src, key in (('ded', 'dedicated_60m'), ('battery', 'battery_5h')):
        for p in PARAMS:
            f = os.path.join(D2, f'fitfair_ded_{DED[p]}_60m.npz' if src == 'ded' else 'fitfair_battery_60m.npz')
            e = vals(dict(np.load(f)), p)[:120]
            for v in range(120):
                vox.append([5, key, p, v, ecc[v], pa[v], truth_all[p][v], e[v]])
    write('recovery.csv', ['round', 'design', 'design_label', 'population', 'parameter', 'parameter_label', 'minutes',
                           'n_voxels', 'spearman', 'ci_lo', 'ci_hi', 'median_abs_err'], rec)
    write('v1_voxels.csv', ['round', 'design', 'parameter', 'voxel', 'ecc', 'polar_angle', 'planted', 'fitted'], vox)

    # ---- round 1 pRF
    r1 = json.load(open(os.path.join(D1, 'results.json')))
    t1 = dict(np.load(os.path.join(D1, 'truth.npz')))
    LAB1 = dict(bars='Bars (fixation)', game_truegaze='Game, perfect gaze', game_eyetracker='Game, eye tracker',
                game_nogaze='Game, gaze ignored', game_et_noarousal='Game, ET, no arousal regressor',
                **{'bars+game': 'Bars + game (ET)'}, game_et_1run='Game ET 5 min', game_et_2run='Game ET 10 min',
                game_et_4run='Game ET 20 min', bars_1run='Bars 5 min', bars_2run='Bars 10 min',
                game_et_2h='Game ET 2 h')
    rows, prfv = [], []
    for k, v in r1.items():
        b = v['by_ecc']
        rows.append([k, LAB1.get(k, k), v['minutes_train'], v['pos_err_median'], b['0-2']['pos_err'], b['2-5']['pos_err'],
                     b['5-9']['pos_err'], b['9-12.1']['pos_err'], v['size_logratio_median'], v['frac_ceiling_game'],
                     v['frac_ceiling_bar']])
        f = np.load(os.path.join(D1, f'fit_{k}.npz'))
        prm = f['params']
        pe = np.hypot(prm[:, 0] - t1['x'], prm[:, 1] - t1['y'])
        for i in range(len(pe)):
            prfv.append([k, i, t1['ecc'][i], t1['x'][i], t1['y'][i], t1['sigma'][i], prm[i, 0], prm[i, 1], prm[i, 2],
                         pe[i], f['r2_game'][i], f['r2_bar'][i]])
    write('prf_round1.csv', ['design', 'design_label', 'minutes', 'pos_err_median', 'pos_err_ecc_0_2',
                             'pos_err_ecc_2_5', 'pos_err_ecc_5_9', 'pos_err_ecc_9_12', 'size_bias_log2',
                             'r2_frac_ceiling_game', 'r2_frac_ceiling_bar'], rows)
    write('prf_voxels.csv', ['design', 'voxel', 'ecc', 'true_x', 'true_y', 'true_sigma', 'fit_x', 'fit_y', 'fit_sigma',
                             'pos_err', 'r2_game', 'r2_bar'], prfv)

    # ---- signal vs noise (round 1)
    vb = np.median([t1[f'bar{i}_vis'].std(0) for i in range(4)], 0)
    vg = np.median([t1[f'game{i}_vis'].std(0) for i in range(8)], 0)
    sy = np.median([t1[f'game{i}_sys'].std(0) for i in range(8)], 0)
    write('signal_components.csv', ['voxel', 'ecc', 'visual_sd_bars', 'visual_sd_game', 'systemic_sd_game',
                                    'noise_sd', 'amplitude', 'systemic_weight'],
          [[i, t1['ecc'][i], vb[i], vg[i], sy[i], t1['noise_sd'][i], t1['amp'][i], t1['sys_amp'][i]]
           for i in range(len(vb))])

    # ---- noise model
    nm = [
        ['Measurement noise', 'All runs', 'AR(1), rho = 0.3', 'SD per voxel ~ U(0.6, 1.2) %', 'No', 'Not modeled (least squares, no prewhitening)'],
        ['Slow drift', 'All runs', 'Legendre order 1-3', 'Coefficients ~ N(0, 0.5) % per voxel per run', 'No', 'Legendre 0-3 regressors per run'],
        ['Event-locked arousal', 'Game hits/explosions; reward-task wins', 'Gamma response peaking at 8 s per event', 'Per-voxel weight ~ N(0.5, 0.2), min 0.05', 'Shape shared, scaled per voxel', 'Regressor from logged events (7 s kernel, mismatched) + derivative'],
        ['Slow global fluctuation', 'All runs', 'Random walk smoothed by 20 s boxcar, SD 0.3', 'Same per-voxel weight as arousal', 'Yes', 'Not modeled (partly absorbed by drift terms)'],
        ['HRF timing', 'All voxels', 'Double-gamma, per-voxel time scaling', 'Scale ~ N(1, 0.08), clipped 0.8-1.2', 'No', 'Round 1 canonical; rounds 2-4 per-voxel scale fitted'],
        ['Session effects', 'Round 4 sessions', 'Gain ~ N(1, 0.1); noise SD x U(0.85, 1.15)', 'Per 2 h session', 'Per session', 'One gain per session fitted'],
        ['Eye-tracker error', 'Game runs', 'Offset N(0, 0.5 deg)/run, drift N(0, 0.3 deg), gain N(0, 4 %), 0.3 deg sample noise, blinks ~1.2 %', 'Per run', '-', 'Not corrected'],
        ['Not simulated', '-', 'Head motion; spatially correlated noise; cardiac/respiratory; voxel-size scaling', '-', '-', 'Testbed is optimistic here'],
    ]
    write('noise_model.csv', ['component', 'applies_to', 'form', 'parameters', 'shared_across_voxels', 'handled_by_fit'], nm)

    # ---- diagnostics
    dec = json.load(open(os.path.join(D1, 'decompose_check.json')))
    tra = json.load(open(os.path.join(D1, 'transient_check.json')))
    dg = [['Round 1: what limits game pRF fits (120 voxels, 30 min, true gaze)', k, 'median position error (deg)', v['pos_err']]
          for k, v in dec.items()]
    dg += [['Round 1: game/bar visual modulation by V1 channel', k, 'median ratio', v['game_over_bar_modulation']]
           for k, v in tra.items()]
    write('diagnostics.csv', ['check', 'condition', 'metric', 'value'], dg)


if __name__ == '__main__':
    main()
