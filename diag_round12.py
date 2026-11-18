"""Round 12 diagnostics: planted against fitted values for every simulated player of the Godot Halcyon, one panel
per parameter, from behaviour and from BOLD, with the bias, slope and share of estimates at the edge
of the searched range written in; plus a tidy CSV of those numbers."""
import csv
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = 'dashboard_data'
BLUE, ORANGE = '#2a78d6', '#eb6834'
cog = pickle.load(open('data_v2/diag_round12_cog.pkl', 'rb'))
new = pickle.load(open('data_v2/diag_round12_new.pkl', 'rb'))
T, E = cog[('halcyon_gd', 12)]
PANELS = [  # (label, planted, behaviour estimate, BOLD estimate)
    ('Call drift scale v', T['v'], E['v_beh'], E['v_bold']),
    ('Call boundary a', T['a'], E['a_beh'], E['a_bold']),
    ('Value learning rate', T['alpha'], E['alpha_beh'], E['alpha_bold']),
    ('Place forgetting rate', T['f'], None, E['f_bold']),
]
for key, lab, tk, bk, ok in (('bandit', 'Stairwell learning rate', 'alpha', 'alpha_beh', 'alpha_bold'),
                             ('bandit', 'Stairwell risk weight', 'lam', 'lam_beh', None),
                             ('mapupd', 'Map update rate', 'alpha', 'alpha_beh', 'alpha_bold'),
                             ('pathint', 'Path-integration heading noise', 'sh', 'sh_beh', 'sh_bold')):
    if (key, 12) in new:
        t, e = new[(key, 12)]
        PANELS.append((lab, t[tk], e[bk], e[ok] if ok else None))


def stats(t, e):
    t, e = np.asarray(t, float), np.asarray(e, float)
    k = np.isfinite(t) & np.isfinite(e)
    t, e = t[k], e[k]
    lo, hi = e.min(), e.max()
    edge = np.mean((np.isclose(e, lo) | np.isclose(e, hi))) if len(np.unique(e)) > 2 else 1.0
    return dict(r=spearmanr(t, e).correlation, bias=float(np.median(e - t)), slope=float(np.polyfit(t, e, 1)[0]), edge=float(edge))


rows = []
fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
for ax, (lab, t, eb, eo) in zip(axs.flat, PANELS):
    t = np.asarray(t, float)
    lo, hi = np.nanmin(t), np.nanmax(t); pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#bbb', lw=1, zorder=0)
    txt = []
    for est, col, name in ((eb, BLUE, 'behaviour'), (eo, ORANGE, 'BOLD')):
        if est is None:
            continue
        est = np.asarray(est, float)
        ax.scatter(t, est, s=18, color=col, alpha=0.75, edgecolor='none', label=name)
        s = stats(t, est)
        txt.append(f"{name}: r {s['r']:.2f}, slope {s['slope']:.2f}, edge {100 * s['edge']:.0f}%")
        rows.append(dict(parameter=lab, source=name, spearman=round(s['r'], 3), bias=round(s['bias'], 4), slope=round(s['slope'], 3), edge_frac=round(s['edge'], 3)))
    ax.set_title(lab, fontsize=11)
    ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel('planted', fontsize=9); ax.set_ylabel('fitted', fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in list(axs.flat)[len(PANELS):]:
    ax.axis('off')
axs.flat[0].legend(fontsize=8, frameon=False, loc='lower right')
fig.suptitle('Halcyon (Godot), 60 minutes: planted vs fitted for 40 simulated players', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_diag_halcyon_gd.png', dpi=110)
with open(f'{OUT}/diag_round12.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows')
