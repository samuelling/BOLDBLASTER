"""Tidy CSVs for the dashboard's round-8 tab: three real games against the raycaster."""
import os
import csv
import json
import math

import numpy as np

import run_v2 as V2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data')
GAMES = [('rc', 'raycaster', 'Raycaster'), ('bm', 'betamax', 'Betamax'), ('bn', 'brickninja', 'Brick Ninja'),
         ('as', 'sanctuary', 'Animal Sanctuary'), ('bt', 'betamaxt', 'Betamax + tunnels'), ('bu', 'betamaxu', 'Tuned tunnels')]
TAG = {t: (g, l) for t, g, l in GAMES}
PLAB = dict(c50_eff='Effective c50', mu='SF peak (pSFT)', sig='SF bandwidth', c50='c50 (raw)', q='CRF exponent',
            k='Surround suppression', tau='Transient weight', a='Attention gain')
OTHER = [('num', 'mu', 'Numerosity', 'Preferred numerosity'), ('num', 'sig', 'Numerosity', 'Numerosity tuning width'),
         ('mst', 'rotation_pref', 'Self-motion', 'Rotation direction preference'),
         ('mst', 'expansion_frac', 'Self-motion', 'Expansion (forward) weight'),
         ('rpe', 'rpe_beta', 'Striatal RPE', 'Prediction-error beta')]
CST = [('taus', 'Sustained time constant'), ('taut', 'Transient time constant'), ('ws', 'Sustained-channel weight')]


def num(v):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def write(name, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    res = json.load(open(V2.p('results_round8.json')))
    rows = []
    for name, r in res.items():
        tag = name.split('_')[0]
        if tag not in TAG or name.startswith(('num_', 'mst_', 'rpe_', 'cst_')):
            continue
        g, gl = TAG[tag]
        for par, sc in r['scores'].items():
            rows.append(dict(game=g, game_label=gl, condition=r['cond'], minutes=r['minutes'], parameter=par,
                             parameter_label=PLAB.get(par, par), spearman=num(sc['spearman']),
                             median_abs_err=num(sc['median_abs_err'])))
    write('round8_v1.csv', rows)

    orows = []
    for mins in (30, 60):
        for t, g, gl in GAMES:
            for model, meas, ml, lab in OTHER:
                k = f'{model}_{t}_{mins}m'
                if k in res:
                    orows.append(dict(model=ml, measure=meas, measure_label=lab, game=g, game_label=gl, condition='realistic',
                                      minutes=mins, spearman=num(res[k]['scores'][meas]['spearman'])))
    # CST: the three games from this round; the raycaster from round 7 (60 min, and 20/40 min averaged for 30)
    r7 = json.load(open(V2.p('results_cst.json'))) if os.path.exists(V2.p('results_cst.json')) else {}
    for mins in (30, 60):
        for cond, cl in (('oracle', 'oracle'), ('real', 'realistic')):
            for t, g, gl in GAMES[1:]:
                k = f'cst_{t}_{cond}_{mins}m'
                if k in res:
                    for meas, lab in CST:
                        orows.append(dict(model='Spatiotemporal (CST)', measure=meas, measure_label=lab, game=g, game_label=gl,
                                          condition=cl, minutes=mins, spearman=num(res[k]['scores'][meas]['spearman'])))
            k7 = f'cst_{cond}_{mins}m'
            for meas, lab in CST:
                if k7 in r7:
                    v, gl = r7[k7]['scores'][meas]['spearman'], 'Raycaster'
                elif mins == 30 and f'cst_{cond}_20m' in r7 and f'cst_{cond}_40m' in r7:
                    v = (r7[f'cst_{cond}_20m']['scores'][meas]['spearman'] + r7[f'cst_{cond}_40m']['scores'][meas]['spearman']) / 2
                    gl = 'Raycaster (mean of 20 & 40 min)'
                else:
                    continue
                orows.append(dict(model='Spatiotemporal (CST)', measure=meas, measure_label=lab, game='raycaster', game_label=gl,
                                  condition=cl, minutes=mins, spearman=num(v)))
    write('round8_other.csv', orows)

    st = json.load(open(V2.p('results_round8_stats.json')))
    brows = []
    for t, g, gl in GAMES:
        if g not in st:
            continue
        e = np.array(st[g]['band_mean'])
        for f_, v in zip(st['bands_cpd'], e / e.max()):
            brows.append(dict(game=g, game_label=gl, band_cpd=round(f_, 3), energy_rel=round(float(v), 4)))
    write('round8_bands.csv', brows)
    S = [('kills_per_min', 'Kills / befriends per minute', 1), ('bursts_per_min', 'Shots / tries per minute', 1),
         ('hit_rate', 'Hit rate', 2), ('salient_per_min', 'Salient onsets per minute', 1),
         ('items_mean', 'Items on screen (mean)', 1), ('items_sd', 'Items on screen (SD)', 1),
         ('fwd_mean', 'Forward speed (raycaster units)', 2), ('yaw_sd', 'Turn / scroll rate SD (rad/s)', 2),
         ('transient_to_sustained', 'Transient / sustained energy', 2)]
    srows = []
    for k, l, d in S:
        row = dict(measure=k, label=l)
        for t, g, gl in GAMES:
            row[g] = round(st[g][k], d) if g in st else ''
        srows.append(row)
    row = dict(measure='gaze_sd', label='Gaze spread, x / y (deg SD)')
    for t, g, gl in GAMES:
        row[g] = ' / '.join(f'{v:.1f}' for v in st[g]['gaze_sd']) if g in st else ''
    srows.append(row)
    write('round8_stats.csv', srows)
    print(len(rows), len(orows), len(brows), len(srows))


if __name__ == '__main__':
    main()
