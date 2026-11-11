"""Round 5 (fair dedicated baseline) -> dashboard_data/fair.csv, budget.csv; results/fair.json.

Everything V1 here is scored on the same first 120 voxels used in round 4, so dedicated,
battery and gameplay rows are directly comparable. Head-to-head differences use a paired
bootstrap over voxels (same resampled voxels for both designs).
"""
import csv
import json
import os
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
D2, DL, OUT = (os.path.join(ROOT, d) for d in ('data_v2', 'data_long', 'dashboard_data'))
NV = 120
PARAMS = ['mu', 'sig', 'c50_eff', 'q', 'k', 'tau', 'a']
PLAB = dict(mu='pSFT peak', sig='pSFT bandwidth', c50_eff='pCRF effective c50', q='pCRF exponent',
            k='Surround suppression k', tau='Transient weight', a='Attention gain')
DED = dict(mu='psft', sig='psft', c50_eff='crf', q='crf', k='size', tau='temporal', a='attn')
NB = 1000


def vals(est, p):
    if p == 'c50_eff':
        return est['c50'] / (1 + est['k']) ** (1 / est['q'])
    return est[p]


def load(path):
    return {k: v[:NV] for k, v in np.load(path).items()}


def truth():
    pop = dict(np.load(os.path.join(D2, 'populations.npz')))
    t = {k: pop[f'v1_{k}'][:NV] for k in ('c50', 'k', 'q', 'mu', 'sig', 'tau', 'a')}
    return {p: vals(t, p) for p in PARAMS}


BOOT = np.random.default_rng(0).integers(0, NV, (NB, NV))


def r_ci(t, e):
    bs = [spearmanr(t[b], e[b]).correlation for b in BOOT]
    return spearmanr(t, e).correlation, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def diff_ci(t, eg, ed):
    d = [spearmanr(t[b], eg[b]).correlation - spearmanr(t[b], ed[b]).correlation for b in BOOT]
    return (spearmanr(t, eg).correlation - spearmanr(t, ed).correlation,
            np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))


def fits():
    """(group, design key, label, minutes, loader(param) -> estimates)"""
    F = []
    for m in (10, 30, 60):
        F.append(('dedicated', f'ded_{m}m', f'Dedicated, one experiment ({m} min)', m,
                  lambda p, m=m: load(os.path.join(D2, f'fitfair_ded_{DED[p]}_{m}m.npz'))))
    for m in (10, 30, 60):
        F.append(('battery', f'battery_{m}m', f'Dedicated battery, joint ({5 * m} min)', 5 * m,
                  lambda p, m=m: load(os.path.join(D2, f'fitfair_battery_{m}m.npz'))))
    for key, m in (('levers_1h', 60), ('levers_2h', 120)):
        F.append(('levers', key, f'Game + levers ({m // 60} h)', m,
                  lambda p, key=key: load(os.path.join(D2, f'fit_{key}.npz'))))
    for key, m in (('long_2h', 120), ('long_5h', 300), ('long_10h', 600), ('long_20h', 1200)):
        F.append(('long', key, f'Varied levels ({m // 60} h)', m,
                  lambda p, key=key: load(os.path.join(DL, f'fit_{key}.npz'))))
    return F


def nonv1_rows():
    r2 = json.load(open(os.path.join(D2, 'results.json')))
    rf = json.load(open(os.path.join(D2, 'results_fair_other.json')))
    P = [('num', 'mu', 'num_mu', 'Numerosity preference'), ('num', 'sig', 'num_sig', 'Numerosity tuning width'),
         ('mst', 'rotation_pref', 'mst_rotation_pref', 'Rotation preference'),
         ('mst', 'expansion_frac', 'mst_expansion_frac', 'Expansion share'),
         ('rpe', 'rpe_beta', 'rpe_rpe_beta', 'RPE weight')]
    NVX = dict(num=150, mst=150, rpe=100)
    rows = []
    for pop, k, pid, lab in P:
        for m in (10, 30, 60):
            rows.append(['dedicated', f'ded_{m}m', f'Dedicated, one experiment ({m} min)', pid, lab, m, NVX[pop],
                         rf[f'{pop}_ded_{m}m']['scores'][k]['spearman'], '', ''])
        for g, m in (('game_1h', 60), ('game_2h', 120)):
            rows.append(['game', g, f'Game ({m // 60} h)', pid, lab, m, NVX[pop], r2[f'{pop}_{g}']['scores'][k]['spearman'], '', ''])
    return rows


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.4g}' if isinstance(x, (float, np.floating)) else x for x in r])
    print(name, len(rows), 'rows')


def verdict(lo, hi):
    return 'Gameplay' if lo > 0 else ('Dedicated' if hi < 0 else 'Tie')


def main():
    T = truth()
    F = fits()
    est = {}
    rows = []
    for grp, key, lab, mins, ld in F:
        for p in PARAMS:
            e = vals(ld(p), p)
            est[(key, p)] = e
            r, lo, hi = r_ci(T[p], e)
            rows.append([grp, key, lab, p, PLAB[p], mins, NV, r, lo, hi])
    rows += nonv1_rows()
    write('fair.csv', ['group', 'design', 'design_label', 'parameter', 'parameter_label', 'minutes', 'n_voxels',
                       'spearman', 'ci_lo', 'ci_hi'], rows)

    # head-to-head: matched time per experiment (1 h) and matched total budget (5 h)
    H = [('1 h', 'ded_60m', 'Dedicated experiment, 1 h', 'levers_1h', 'Game + levers, 1 h'),
         ('5 h', 'battery_60m', 'Full dedicated battery, 5 h', 'long_5h', 'Varied-level gameplay, 5 h')]
    brow, summ = [], {}
    for budget, dk, dl, gk, gl in H:
        for p in PARAMS:
            rd = spearmanr(T[p], est[(dk, p)]).correlation
            rg = spearmanr(T[p], est[(gk, p)]).correlation
            d, lo, hi = diff_ci(T[p], est[(gk, p)], est[(dk, p)])
            v = verdict(lo, hi)
            brow.append([budget, p, PLAB[p], dk, dl, rd, gk, gl, rg, d, lo, hi, v])
            summ.setdefault(budget, {})[p] = dict(dedicated=rd, game=rg, diff=d, lo=lo, hi=hi, verdict=v)
    write('budget.csv', ['budget', 'parameter', 'parameter_label', 'ded_design', 'ded_label', 'ded_r', 'game_design',
                         'game_label', 'game_r', 'diff', 'diff_lo', 'diff_hi', 'verdict'], brow)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    json.dump(dict(fair={f'{r[1]}|{r[3]}': dict(minutes=r[5], r=r[7]) for r in rows}, head_to_head=summ),
              open(os.path.join(ROOT, 'results', 'fair.json'), 'w'), indent=1, default=float)
    for b, s in summ.items():
        print(b, {p: (round(x['dedicated'], 2), round(x['game'], 2), x['verdict']) for p, x in s.items()})


if __name__ == '__main__':
    main()
