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


def fig_levers(res):
    """Round 3: v1 parameters, dedicated vs game v2 vs game v3 (design levers)."""
    params = [('mu', 'pSFT peak'), ('sig', 'pSFT bandwidth'), ('c50_eff', 'pCRF eff. c50'),
              ('q', 'pCRF exponent'), ('k', 'Surround k'), ('tau', 'Transient τ'), ('a', 'Attention gain')]
    series = [('Dedicated 10 min', None, '#2a78d6'), ('Game 2 h', 'game_2h', '#9c9a92'),
              ('Game + levers 2 h', 'levers_2h', '#eb6834'),
              ('Game + levers 2 h, oracle', 'levers_2h_oracle', '#1baf7a')]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    x = np.arange(len(params))
    w = 0.2
    for j, (lab, key, col) in enumerate(series):
        vals = []
        for prm, _ in params:
            k = key or 'dedicated_' + DED['c50' if prm == 'c50_eff' else prm]
            vals.append(res[k]['scores'][prm]['spearman'] if k in res else np.nan)
        b = ax.bar(x + (j - 1.5) * w, vals, width=w - 0.02, color=col, label=lab)
        for xi, v in zip(x + (j - 1.5) * w, vals):
            if not np.isnan(v):
                ax.text(xi, max(v, 0) + 0.015, f'{v:.2f}', ha='center', fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in params])
    ax.set_ylabel('Spearman r (planted vs fitted)')
    ax.set_ylim(0, 1)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.grid(axis='y', color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    ax.set_title('V1 parameter recovery with design levers (covert-attention cues + SF/contrast-decorrelated textures)', loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig8_levers.png'), dpi=140)
    plt.close()


if __name__ == '__main__':
    fig_levers(json.load(open(os.path.join(D, 'results.json'))))


def fig_long(S):
    """S: data_long/summary.json (Spearman r with bootstrap 95% CIs, same 120 voxels)."""
    params = [('mu', 'pSFT peak'), ('sig', 'pSFT bandwidth'), ('c50_eff', 'pCRF eff. c50'),
              ('q', 'pCRF exponent'), ('k', 'Surround k'), ('tau', 'Transient τ'), ('a', 'Attention gain')]
    names = ['long_2h', 'long_5h', 'long_10h', 'long_20h']
    hrs = [2, 5, 10, 20]
    fig, axs = plt.subplots(1, len(params) + 1, figsize=(16, 3.4),
                            gridspec_kw=dict(width_ratios=[1] * len(params) + [1.5]))
    for a_, (prm, lab) in zip(axs, params):
        y = [S[n][prm]['r'] for n in names]
        lo = [S[n][prm]['lo'] for n in names]
        hi = [S[n][prm]['hi'] for n in names]
        a_.fill_between(hrs, lo, hi, color='#eb6834', alpha=0.18, lw=0)
        a_.plot(hrs, y, '-o', color='#eb6834', lw=2, ms=4)
        dd = S['dedicated_10min'][prm]
        a_.axhspan(dd['lo'], dd['hi'], color='#2a78d6', alpha=0.12, lw=0)
        a_.axhline(dd['r'], color='#2a78d6', lw=1.5, ls='--')
        a_.set_xscale('log')
        a_.set_xticks(hrs)
        a_.set_xticklabels([str(h) for h in hrs])
        a_.minorticks_off()
        a_.set_ylim(-0.1, 1)
        a_.set_title(lab, fontsize=9)
        a_.set_xlabel('hours')
        a_.grid(color='#e4e3df', lw=0.5)
        a_.set_axisbelow(True)
    axs[0].set_ylabel('Spearman r (95% CI)')
    axs[0].text(2.1, 0.93, 'game (orange)\ndedicated 10 min\n(blue dashed)', fontsize=7, color=INK2, va='top')
    a_ = axs[-1]
    D_ = S['varied_minus_single_4h']
    y = np.arange(len(params))
    for yi, (prm, lab) in zip(y, params):
        v = D_[prm]
        a_.plot([v['lo'], v['hi']], [yi, yi], color='#eb6834', lw=2)
        a_.plot(v['diff'], yi, 'o', color='#eb6834', ms=6)
    a_.axvline(0, color=INK2, lw=0.8)
    a_.set_yticks(y)
    a_.set_yticklabels([l for _, l in params], fontsize=8)
    a_.invert_yaxis()
    a_.set_xlim(-0.35, 0.35)
    a_.set_xlabel('Δr, varied − one level (4 h)')
    a_.set_title('Variety at equal time', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig9_long.png'), dpi=140)
    plt.close()


if __name__ == '__main__':
    LP = os.path.join(ROOT, 'data_long', 'summary.json')
    if os.path.exists(LP):
        fig_long(json.load(open(LP)))
