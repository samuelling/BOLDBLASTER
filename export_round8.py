"""Tidy CSVs for the dashboard's round-8 section (Betamax vs the raycaster)."""
import os
import csv
import json

import numpy as np

import run_v2 as V2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data')
PLAB = dict(c50_eff='Effective c50', mu='SF peak (pSFT)', sig='SF bandwidth', c50='c50 (raw)', q='CRF exponent',
            k='Surround suppression', tau='Transient weight', a='Attention gain')
OTHER = [('num', 'mu', 'Numerosity', 'Preferred numerosity'), ('num', 'sig', 'Numerosity', 'Numerosity tuning width'),
         ('mst', 'rotation_pref', 'Self-motion', 'Rotation direction preference'),
         ('mst', 'expansion_frac', 'Self-motion', 'Expansion (forward) weight'),
         ('rpe', 'rpe_beta', 'Striatal RPE', 'Prediction-error beta')]
CST = [('taus', 'Sustained time constant'), ('taut', 'Transient time constant'), ('ws', 'Sustained-channel weight')]


def main():
    res = json.load(open(V2.p('results_betamax.json')))
    rows = []
    for name, r in res.items():
        if not (name.startswith('bm_') or name.startswith('rc_')):
            continue
        for par, sc in r['scores'].items():
            rows.append(dict(game=r['game'], game_label='Betamax' if r['game'] == 'betamax' else 'Raycaster',
                             condition=r['cond'], minutes=r['minutes'], parameter=par, parameter_label=PLAB.get(par, par),
                             spearman=round(sc['spearman'], 4), median_abs_err=round(sc['median_abs_err'], 4)))
    with open(os.path.join(OUT, 'round8_v1.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    orows = []
    for mins in (30, 60):
        for g, gl in (('bm', 'Betamax'), ('rc', 'Raycaster')):
            for model, meas, ml, lab in OTHER:
                k = f'{model}_{g}_{mins}m'
                if k in res:
                    orows.append(dict(model=ml, measure=meas, measure_label=lab, game_label=gl, condition='realistic',
                                      minutes=mins, spearman=round(res[k]['scores'][meas]['spearman'], 4)))
    # CST: Betamax from this round, raycaster from round 7 at the nearest matched duration
    r7 = json.load(open(V2.p('results_cst.json'))) if os.path.exists(V2.p('results_cst.json')) else {}
    for mins in (30, 60):
        for cond, cl in (('oracle', 'oracle'), ('real', 'realistic')):
            k = f'cst_bm_{cond}_{mins}m'
            if k in res:
                for meas, lab in CST:
                    orows.append(dict(model='Spatiotemporal (CST)', measure=meas, measure_label=lab, game_label='Betamax',
                                      condition=cl, minutes=mins, spearman=round(res[k]['scores'][meas]['spearman'], 4)))
            k7 = f'cst_{cond}_{mins}m'
            if k7 not in r7 and mins == 30:     # round 7 ran 20/40/60: average 20 and 40 for a 30-min stand-in
                a, b = r7.get(f'cst_{cond}_20m'), r7.get(f'cst_{cond}_40m')
                if a and b:
                    for meas, lab in CST:
                        v = (a['scores'][meas]['spearman'] + b['scores'][meas]['spearman']) / 2
                        orows.append(dict(model='Spatiotemporal (CST)', measure=meas, measure_label=lab,
                                          game_label='Raycaster (mean of 20 & 40 min)', condition=cl, minutes=mins,
                                          spearman=round(v, 4)))
            elif k7 in r7:
                for meas, lab in CST:
                    orows.append(dict(model='Spatiotemporal (CST)', measure=meas, measure_label=lab, game_label='Raycaster',
                                      condition=cl, minutes=mins, spearman=round(r7[k7]['scores'][meas]['spearman'], 4)))
    with open(os.path.join(OUT, 'round8_other.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(orows[0]))
        w.writeheader()
        w.writerows(orows)
    st = json.load(open(V2.p('results_betamax_stats.json')))
    brows = []
    for g, gl in (('betamax', 'Betamax'), ('raycaster', 'Raycaster')):
        e = np.array(st[g]['band_mean'])
        for f_, v, sd in zip(st['bands_cpd'], e / e.max(), np.array(st[g]['band_temporal_sd']) / e.max()):
            brows.append(dict(game_label=gl, band_cpd=round(f_, 3), energy_rel=round(float(v), 4), temporal_sd_rel=round(float(sd), 4)))
    with open(os.path.join(OUT, 'round8_bands.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0]))
        w.writeheader()
        w.writerows(brows)
    S = [('kills_per_min', 'Kills per minute', 1), ('bursts_per_min', 'Shots (bursts) per minute', 1),
         ('hit_rate', 'Shot hit rate', 2), ('salient_per_min', 'Salient onsets per minute', 1),
         ('items_mean', 'Items on screen (mean)', 1), ('items_sd', 'Items on screen (SD)', 1),
         ('fwd_mean', 'Forward speed (raycaster units)', 2), ('yaw_sd', 'Turn rate SD (rad/s)', 2),
         ('transient_to_sustained', 'Transient / sustained energy', 2)]
    srows = [dict(measure=k, label=l, betamax=round(st['betamax'][k], d), raycaster=round(st['raycaster'][k], d)) for k, l, d in S]
    srows.append(dict(measure='gaze_sd', label='Gaze spread, x / y (deg SD)',
                      betamax=' / '.join(f'{v:.1f}' for v in st['betamax']['gaze_sd']),
                      raycaster=' / '.join(f'{v:.1f}' for v in st['raycaster']['gaze_sd'])))
    with open(os.path.join(OUT, 'round8_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    print(len(rows), len(orows), len(brows), len(srows))


if __name__ == '__main__':
    main()
