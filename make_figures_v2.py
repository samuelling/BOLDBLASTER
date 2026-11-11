"""v2 figures: recovery matrix across models and designs, plus true-vs-fit scatters."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.abspath(__file__))
D, F = os.path.join(ROOT, 'data_v2'), os.path.join(ROOT, 'figures')
os.makedirs(F, exist_ok=True)
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlesize': 10, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
SEQ = LinearSegmentedColormap.from_list('blue', ['#f3f6fb', '#c6dbf5', '#7fb0ea', '#2a78d6', '#103f7a'])

ROWS = [  # (label, population, result key prefix/dedicated key, param)
    ('pSFT peak (μ)', 'v1', 'dedicated_psft', 'mu'),
    ('pSFT bandwidth (σ)', 'v1', 'dedicated_psft', 'sig'),
    ('pCRF effective c50', 'v1', 'dedicated_crf', 'c50_eff'),
    ('pCRF exponent (q)', 'v1', 'dedicated_crf', 'q'),
    ('Surround suppression (k)', 'v1', 'dedicated_size', 'k'),
    ('Transient weight (τ)', 'v1', 'dedicated_temporal', 'tau'),
    ('Attention gain (a)', 'v1', 'dedicated_attn', 'a'),
    ('Numerosity preference', 'num', 'num_dedicated', 'mu'),
    ('Numerosity tuning width', 'num', 'num_dedicated', 'sig'),
    ('Self-motion: rotation preference', 'mst', 'mst_dedicated', 'rotation_pref'),
    ('Self-motion: expansion share', 'mst', 'mst_dedicated', 'expansion_frac'),
    ('Striatal RPE β', 'rpe', 'rpe_dedicated', 'rpe_beta'),
]
COLS = ['Dedicated\n(10 min each)', 'V1 battery,\njoint (50 min)', 'Game 1 h', 'Game 2 h', 'Game 2 h\noracle gaze+pRF']


def cell(res, pop, ded_key, param, col):
    if col == 0:
        key = ded_key
    elif col == 1:
        key = 'battery_joint' if pop == 'v1' else None
    else:
        tag = ['game_1h', 'game_2h', 'game_2h_oracle'][col - 2]
        key = tag if pop == 'v1' else (f'{pop}_{tag}' if tag != 'game_2h_oracle' else None)
    if key is None or key not in res or param not in res[key]['scores']:
        return np.nan
    return res[key]['scores'][param]['spearman']


def fig_matrix(res):
    M = np.array([[cell(res, pop, dk, prm, c) for c in range(len(COLS))] for _, pop, dk, prm in ROWS])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.imshow(np.nan_to_num(M, nan=0), cmap=SEQ, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = '—' if np.isnan(v) else f'{v:.2f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color='white' if (not np.isnan(v) and v > 0.62) else INK)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.xaxis.tick_top()
    for y in (6.5, 8.5, 10.5):
        ax.axhline(y, color='white', lw=3)
    ax.set_title('Parameter recovery: Spearman r between planted and fitted values\n', loc='left')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig6_recovery_matrix.png'), dpi=140)
    plt.close()
    return M


def fig_scatter(pop):
    pairs = [('mu', 'pSFT peak μ (log2 cpd)'), ('c50', 'pCRF c50'), ('k', 'surround k'),
             ('tau', 'transient τ'), ('a', 'attention gain a')]
    conds = [('dedicated', '#2a78d6'), ('game_2h', '#eb6834')]
    fig, ax = plt.subplots(1, len(pairs), figsize=(15, 3.3))
    for a_, (k, lab) in zip(ax, pairs):
        t = pop[f'v1_{k}']
        for cname, col in conds:
            f = os.path.join(D, f'fit_{"dedicated_" + DED[k] if cname == "dedicated" else cname}.npz')
            if not os.path.exists(f):
                continue
            e = np.load(f)[k]
            a_.scatter(t, e, s=7, alpha=0.55, color=col, lw=0,
                       label='dedicated 10 min' if cname == 'dedicated' else 'game 2 h')
        lo, hi = np.percentile(t, [0, 100])
        a_.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls='--')
        if k == 'c50':
            a_.set_xscale('log')
            a_.set_yscale('log')
            a_.set_xticks([0.1, 0.2, 0.4])
            a_.set_xticklabels(['0.1', '0.2', '0.4'])
            a_.set_yticks([0.01, 0.1, 1])
            a_.set_yticklabels(['0.01', '0.1', '1'])
            a_.minorticks_off()
        a_.set_title(lab)
        a_.set_xlabel('planted')
    ax[0].set_ylabel('fitted')
    leg = ax[0].legend(frameon=False, fontsize=8, markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig7_v1_scatter.png'), dpi=130)
    plt.close()


DED = dict(mu='psft', sig='psft', c50='crf', q='crf', k='size', tau='temporal', a='attn')

if __name__ == '__main__':
    res = json.load(open(os.path.join(D, 'results.json')))
    pop = dict(np.load(os.path.join(D, 'populations.npz')))
    M = fig_matrix(res)
    fig_scatter(pop)
    print(np.round(M, 2))
