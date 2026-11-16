"""Round 10 diagnostics: planted vs fitted values, one dot per voxel or simulated player, so the
kind of error (scatter, bias, shrinkage, estimates stuck at the edge of the grid) shows beside
the correlation. Stages: compute (reruns the cognitive fits and keeps every player's values) | plot."""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PKLS = {'betamaxt': 'data_v2/diag_round10_cog.pkl', 'betamaxu': 'data_v2/diag_round10u_cog.pkl'}
PKL = PKLS['betamaxt']
OUT = 'dashboard_data'
GREY, BLUE, ORANGE = '#8a8890', '#2a78d6', '#eb6834'
# Betamax: open grey; first tunnels: blue; tuned tunnels: orange
STYLE = {'betamax': (GREY, 'Betamax: ', False), 'betamaxt': (BLUE, 'Tunnels: ', True), 'betamaxu': (ORANGE, 'Tuned: ', True)}


def compute():
    """Run once per tunnel version (R10_GAME=betamaxt or betamaxu)."""
    import run_round10 as R10
    R9 = R10.R9
    G = R10.G
    out = {}
    for g in ('betamax', G):
        R9.cog_game(g, 12)
        out[g] = R9.SAVE[(g, 12)]
    runs = R10.session(12)
    rng = np.random.default_rng(1000 + 12)
    R10.pathint(runs, 12, rng)
    R10.mapupd(runs, 12, rng)
    rows = [(k, r['fr'](f[0]) / R10.FPS, 0.5 + 0.5 * f[6] * (1 if f[3] else -1), 1.0)
            for k, r in enumerate(runs) for f in r['ev']['figures']]
    save = R9.opportunities
    R9.opportunities = lambda g, n: np.array(rows, float)
    R9.cog_game(G, 12)
    R9.opportunities = save
    out['ddm_id'] = R9.SAVE[(G, 12)]
    out['pathint'] = R10.SAVE[('pathint', 12)]
    out['mapupd'] = R10.SAVE[('mapupd', 12)]
    pickle.dump(out, open(PKLS[G], 'wb'))


def summary(t, e):
    k = np.isfinite(t) & np.isfinite(e)
    t, e = t[k], e[k]
    return spearmanr(t, e).correlation, np.median(e - t), np.polyfit(t, e, 1)[0]


def panel(a, series, title, lim=None, unit=''):
    """series: (planted, fitted, colour, label, filled). Grid-fitted estimates take few values, so a
    little vertical jitter (1.5% of the range) keeps stacked dots visible."""
    allv = np.concatenate([np.r_[s[0], s[1]] for s in series])
    lo, hi = lim or (np.nanmin(allv), np.nanmax(allv))
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    rng = np.random.default_rng(0)
    lines = []
    for t, e, c, lab, filled in series:
        t, e = np.asarray(t, float), np.asarray(e, float)
        j = rng.normal(0, (hi - lo) * 0.015, len(e)) if len(np.unique(np.round(e, 4))) < 20 else 0
        a.scatter(t, e + j, s=16, facecolor=c if filled else 'none', edgecolor=c if not filled else 'white',
                  linewidth=0.9 if not filled else 0.4, alpha=0.85, zorder=3)
        r, b, sl = summary(t, e)
        ic = np.polyfit(t, e, 1)[1]
        xs = np.array([np.nanmin(t), np.nanmax(t)])
        a.plot(xs, sl * xs + ic, color=c, lw=1.8, zorder=4)
        lines.append(f'{lab}r {r:.2f} · bias {b:+.2g}{unit} · slope {sl:.2f}')
    a.plot([lo, hi], [lo, hi], '--', color='#b4b2aa', lw=1, zorder=1)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_title(title + '\n' + '\n'.join(lines), fontsize=7.5, loc='left')
    a.tick_params(labelsize=7)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)


def plot():
    D2 = {g: pickle.load(open(f, 'rb')) for g, f in PKLS.items() if os.path.exists(f)}
    d = D2['betamaxt']
    conds = [('betamax', d['betamax'])] + [(g, D2[g][g]) for g in ('betamaxt', 'betamaxu') if g in D2]
    ser = lambda tk, ek: [(c[0][tk], c[1][ek]) + STYLE[g] for g, c in conds]
    # ---- shared models, Betamax (grey, open) vs with tunnels (blue, filled)
    rows = [('v', 'v_beh', 'v_bold', 'Drift scale v (firefight)'), ('a', 'a_beh', 'a_bold', 'Boundary a (firefight)'),
            ('alpha', 'alpha_beh', 'alpha_bold', 'Learning rate α')]
    fig, ax = plt.subplots(2, 4, figsize=(11, 6.4), dpi=110)
    for j, (tk, kb, kd, nm) in enumerate(rows):
        for i, (k, src) in enumerate(((kb, 'behaviour'), (kd, 'BOLD'))):
            panel(ax[i, j], ser(tk, k), f'{nm}, from {src}')
    panel(ax[0, 3], ser('beta', 'beta_beh'), 'Choice consistency β, from behaviour')
    panel(ax[1, 3], ser('f', 'f_bold'), 'Place forgetting rate (per min), from BOLD')
    for a in ax[1]:
        a.set_xlabel('planted', fontsize=8)
    for a in ax[:, 0]:
        a.set_ylabel('fitted', fontsize=8)
    fig.suptitle('40 simulated players: planted vs fitted. Open grey = Betamax, blue = first tunnels, orange = tuned tunnels. Dashed = perfect recovery; lines = fitted trend',
                 fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f'{OUT}/fig_diag_cog.png')
    plt.close()
    # ---- models only the tunnels can test
    tun = [(g, D2[g]) for g in ('betamaxt', 'betamaxu') if g in D2]
    fig, ax = plt.subplots(2, 4, figsize=(11, 6.6), dpi=110)
    P = [(ax[0, 0], 'pathint', 'sh', 'sh_beh', 'Path integration: heading noise,\nfrom where players think home is'),
         (ax[1, 0], 'pathint', 'sh', 'sh_bold', 'Path integration: heading noise,\nfrom entorhinal BOLD'),
         (ax[0, 1], 'pathint', 'gd', 'gd_beh', 'Path integration: distance gain,\nfrom behaviour'),
         (ax[1, 1], 'pathint', 'la', 'la_beh', 'Path integration: leak (per m),\nfrom behaviour'),
         (ax[0, 2], 'mapupd', 'alpha', 'alpha_beh', 'Map updating: update rate,\nfrom junction choices'),
         (ax[1, 2], 'mapupd', 'alpha', 'alpha_bold', 'Map updating: update rate,\nfrom hippocampal BOLD'),
         (ax[0, 3], 'ddm_id', 'v', 'v_beh', 'Friend or foe: drift scale,\nfrom behaviour'),
         (ax[1, 3], 'ddm_id', 'v', 'v_bold', 'Friend or foe: drift scale,\nfrom accumulator BOLD')]
    for a, mod, tk, ek, nm in P:
        panel(a, [(np.asarray(dd[mod][0][tk], float), np.asarray(dd[mod][1][ek], float)) + STYLE[g] for g, dd in tun], nm)
    for a in ax[1]:
        a.set_xlabel('planted', fontsize=8)
    for a in ax[:, 0]:
        a.set_ylabel('fitted', fontsize=8)
    fig.suptitle('Models only the tunnels can test: 40 simulated players, planted vs fitted. Blue = first tunnels, orange = tuned. Dashed = perfect recovery; lines = fitted trend', fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f'{OUT}/fig_diag_new.png')
    plt.close()


def plot_v1():
    """300 V1 voxels: SF peak and effective c50, Betamax vs tunnels, eye tracker vs true gaze."""
    import run_v2 as V2
    pop = V2.populations()

    def get(tag):
        e = dict(np.load(f'data_v2/fit_{tag}.npz'))
        e['c50_eff'] = np.log2(e['c50'] / (1 + e['k']) ** (1 / e['q']))
        t = {k: pop[f'v1_{k}'] for k in ('mu', 'c50', 'q', 'k')}
        t['c50_eff'] = np.log2(t['c50'] / (1 + t['k']) ** (1 / t['q']))
        return t, e
    conds = [('bm_60m', 'Betamax, eye tracker'), ('bt_60m', 'First tunnels, eye tracker'), ('bu_60m', 'Tuned tunnels, eye tracker'),
             ('bu_60m_oracle', 'Tuned tunnels, true gaze')]
    conds = [c for c in conds if os.path.exists(f'data_v2/fit_{c[0]}.npz')]
    fig, ax = plt.subplots(2, 4, figsize=(11, 6.2), dpi=110)
    for j, (tag, lab) in enumerate(conds):
        t, e = get(tag)
        for i, (k, nm, lim) in enumerate((('mu', 'SF peak (log2 c/deg)', (-2.5, 2.0)), ('c50_eff', 'effective c50 (log2)', (-6, 1)))):
            a = ax[i, j]
            col = GREY if tag.startswith('bm') else BLUE if tag.startswith('bt') else ORANGE
            panel(a, [(t[k], np.clip(e[k], *lim), col, '', True)], f'{lab}: {nm}', lim=lim)
    for a in ax[1]:
        a.set_xlabel('planted', fontsize=8)
    for a in ax[:, 0]:
        a.set_ylabel('fitted', fontsize=8)
    fig.suptitle('300 V1 voxels, 60 min: planted vs fitted. Dashed = perfect recovery; line = fitted trend. Estimates beyond the plot are drawn at its edge',
                 fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f'{OUT}/fig_diag_v1.png')
    plt.close()


def table():
    """One row per panel: correlation, median bias, slope of fitted on planted, fitted/planted SD,
    and the share of estimates at the edge of the fit's range."""
    import csv
    rows = []

    def add(key, model, measure, source, game, t, e, lo=None, hi=None):
        t, e = np.asarray(t, float), np.asarray(e, float)
        k = np.isfinite(t) & np.isfinite(e)
        t, e = t[k], e[k]
        r, b, sl = summary(t, e)
        lo = e.min() if lo is None else lo
        hi = e.max() if hi is None else hi
        tol = (hi - lo) * 0.005
        edge = np.mean((e <= lo + tol) | (e >= hi - tol))
        rows.append(dict(key=key, model=model, measure=measure, source=source, game=game, n=len(t), spearman=round(r, 3),
                         bias=round(b, 4), slope=round(sl, 3), sd_ratio=round(e.std() / t.std(), 2), edge_frac=round(edge, 2)))
    G = (('alpha', 'alpha_beh', 'behaviour', 0.03, 0.65), ('alpha', 'alpha_bold', 'bold', 0.03, 0.65), ('v', 'v_beh', 'behaviour', None, None),
         ('v', 'v_bold', 'bold', 0.8, 4.2), ('a', 'a_beh', 'behaviour', None, None), ('a', 'a_bold', 'bold', 0.7, 2.1),
         ('beta', 'beta_beh', 'behaviour', 1, 14), ('e', 'e_bold', 'bold', 0.15, 0.85), ('f', 'f_bold', 'bold', 0.01, 0.8))
    D2 = {g: pickle.load(open(f, 'rb')) for g, f in PKLS.items() if os.path.exists(f)}
    shared = [('betamax', D2['betamaxt']['betamax'])] + [(g, D2[g][g]) for g in D2]
    for g, (T, E) in shared:
        for tk, ek, src, lo, hi in G:
            mod = {'v': 'ddm', 'a': 'ddm', 'alpha': 'rl', 'beta': 'rl', 'e': 'novelty', 'f': 'novelty'}[tk]
            add(f'{mod}_{tk}_{src}_{g}', mod, tk, src, g, T[tk], E[ek], lo, hi)
    for g, d in D2.items():
      T, E = d['ddm_id']
      for tk, ek, src, lo, hi in G[2:6]:
        add(f'ddm_id_{tk}_{src}_{g}', 'ddm_id', tk, src, g, T[tk], E[ek], lo, hi)
      T, E = d['pathint']
      for tk, ek, src, lo, hi in (('sh', 'sh_beh', 'behaviour', 0.02, 0.16), ('sh', 'sh_bold', 'bold', 0.02, 0.16),
                                  ('gd', 'gd_beh', 'behaviour', 0.7, 1.3), ('la', 'la_beh', 'behaviour', 0, 0.03)):
        add(f'pathint_{tk}_{src}_{g}', 'pathint', tk, src, g, T[tk], E[ek], lo, hi)
      T, E = d['mapupd']
      for tk, ek, src, lo, hi in (('alpha', 'alpha_beh', 'behaviour', 0.05, 0.95), ('alpha', 'alpha_bold', 'bold', 0.05, 0.95),
                                  ('tau', 'tau_beh', 'behaviour', 0.3, 4.0)):
        add(f'mapupd_{tk}_{src}_{g}', 'mapupd', tk, src, g, T[tk], E[ek], lo, hi)
        if ek == 'alpha_beh':
            rows[-1]['max_fit'] = round(float(np.max(E[ek])), 2)
    import run_v2 as V2
    pop = V2.populations()
    for tag, g, gz in (('bm_60m', 'betamax', 'eyetracker'), ('bt_60m', 'betamaxt', 'eyetracker'), ('bu_60m', 'betamaxu', 'eyetracker'),
                       ('bm_60m_oracle', 'betamax', 'true'), ('bt_60m_oracle', 'betamaxt', 'true'), ('bu_60m_oracle', 'betamaxu', 'true')):
        if not os.path.exists(f'data_v2/fit_{tag}.npz'):
            continue
        e = dict(np.load(f'data_v2/fit_{tag}.npz'))
        for k, lo, hi in (('mu', -2.5, 2.0), ('sig', 0.3, 3.0)):
            add(f'v1_{k}_{gz}_{g}', 'v1', k, gz, g, pop[f'v1_{k}'], e[k], lo, hi)
        te = np.log2(pop['v1_c50'] / (1 + pop['v1_k']) ** (1 / pop['v1_q']))
        add(f'v1_c50eff_{gz}_{g}', 'v1', 'c50_eff', gz, g, te, np.log2(e['c50'] / (1 + e['k']) ** (1 / e['q'])))
    F = ['key', 'model', 'measure', 'source', 'game', 'n', 'spearman', 'bias', 'slope', 'sd_ratio', 'edge_frac', 'max_fit']
    with open(f'{OUT}/diag_round10.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, F); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if st in ('compute', 'all'):
        compute()
    plot()
    plot_v1()
    table()
