"""Figures + sandbox clip for the BOLDBLASTER report."""
import json
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from boldblaster import stimuli as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(F, exist_ok=True)

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C = dict(bars='#2a78d6', game_truegaze='#1baf7a', game_eyetracker='#eb6834',
         game_nogaze='#e87ba4', game_et_noarousal='#eda100', **{'bars+game': '#6250d6'})
LABEL = {'bars': 'Bars (fixation)', 'game_truegaze': 'Game, perfect gaze',
         'game_eyetracker': 'Game, in-bore eye tracker', 'game_nogaze': 'Game, gaze ignored',
         'game_et_noarousal': 'Game, ET, no arousal regressor', 'bars+game': 'Bars + game (ET)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def load(name):
    return dict(np.load(os.path.join(D, name)))


def fig_sandbox():
    fr = np.load(os.path.join(D, 'game0_frames.npy')).astype(np.float32) / 255
    ce = np.load(os.path.join(D, 'game0_ce.npy'), mmap_mode='r')
    gz = np.load(os.path.join(D, 'game0_gaze.npy'))
    bfr = np.load(os.path.join(D, 'bar0_frames.npy')).astype(np.float32) / 255
    bce = np.load(os.path.join(D, 'bar0_ce.npy'), mmap_mode='r')
    idx = [95, 260, 455]
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.4))
    for k, i in enumerate(idx):
        ax[0, k].imshow(fr[i], cmap='gray', vmin=0, vmax=1, extent=[-12, 12, -9, 9])
        ax[0, k].plot(*gz[i], marker='+', color='#e34948', ms=14, mew=2)
        ax[0, k].set_title(f'Game frame t={i / 10:.1f}s (+ = gaze)')
        ret = st.retinal_stack(np.asarray(ce[i:i + 1], np.float32), gz[i:i + 1])[0]
        ax[1, k].imshow(ret, cmap='magma', extent=[-18, 18, -18, 18], vmin=0, vmax=0.09)
        ax[1, k].set_title('Retinal contrast energy')
        ax[1, k].set_xlabel('deg (retinal)')
    ax[0, 3].imshow(bfr[200], cmap='gray', vmin=0, vmax=1, extent=[-12, 12, -9, 9])
    ax[0, 3].set_title('Bar retinotopy frame')
    meanCE = np.asarray(ce, np.float32).mean(0)
    ax[1, 3].imshow(meanCE, cmap='magma', extent=[-15, 15, -12, 12])
    ax[1, 3].set_title('Mean game CE (screen coords)')
    for a in ax.flat:
        a.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig1_sandbox.png'), dpi=130)
    plt.close()


def fig_signal(truth):
    vis_sd = {k: np.median([truth[f'{k}{i}_vis'].std(0) for i in range(n)], 0)
              for k, n in (('bar', 4), ('game', 8))}
    sys_sd = np.median([truth[f'game{i}_sys'].std(0) for i in range(8)], 0)
    noise_sd = truth['noise_sd']
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    v = int(np.argsort(np.abs(truth['ecc'] - 4))[0])
    t = np.arange(200) * 1.5
    Y, V, S = truth['game6_Y'][:, v], truth['game6_vis'][:, v], truth['game6_sys'][:, v]
    ev = np.load(os.path.join(D, 'game6_events.npy'))
    ax.plot(t, Y - Y.mean(), color='#bdbcb6', lw=1, label='simulated BOLD')
    ax.plot(t, V - V.mean(), color=C['bars'], lw=2, label='visual (pRF) component')
    ax.plot(t, S - S.mean(), color=C['game_eyetracker'], lw=2, label='systemic arousal component')
    for e in ev[:, 0]:
        ax.axvline(e, color=INK2, lw=0.6, alpha=0.4)
    ax.set_xlabel('time (s); grey ticks = tag/explosion events')
    ax.set_ylabel('% signal change')
    ax.set_title(f'One V1 voxel during gameplay (ecc {truth["ecc"][v]:.1f}°)')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax2 = fig.add_subplot(gs[1])
    vals = [np.median(vis_sd['bar']), np.median(vis_sd['game']), np.median(sys_sd), np.median(noise_sd)]
    names = ['visual\n(bars)', 'visual\n(game)', 'systemic\n(game)', 'noise']
    bars = ax2.bar(names, vals, color=[C['bars'], C['game_eyetracker'], '#9c9a92', '#cfcdc6'], width=0.6)
    for b, val in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', fontsize=8, color=INK)
    ax2.set_ylabel('median SD across time (%)')
    ax2.set_title('Signal components')
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(truth['ecc'], vis_sd['game'] / vis_sd['bar'], s=8, color=C['game_eyetracker'], alpha=0.6, lw=0)
    ax3.set_xlabel('true eccentricity (deg)')
    ax3.set_ylabel('game / bar visual modulation')
    ax3.set_title('Modulation ratio by eccentricity')
    ax3.axvline(9, color=INK2, lw=0.8, ls='--')
    ax3.text(9.2, ax3.get_ylim()[1] * 0.9, 'bar aperture edge', fontsize=7, color=INK2)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig2_signal.png'), dpi=130)
    plt.close()


def fig_recovery(truth, conds):
    fig, ax = plt.subplots(2, len(conds), figsize=(3.3 * len(conds), 6.4), sharey='row')
    ecc = truth['ecc']
    bins = np.linspace(0, 12, 9)
    for k, c in enumerate(conds):
        r = load(f'fit_{c}.npz')
        prm = r['params']
        pe = np.hypot(prm[:, 0] - truth['x'], prm[:, 1] - truth['y'])
        a = ax[0, k]
        a.scatter(ecc, pe, s=6, color=C[c], alpha=0.45, lw=0)
        med = [np.median(pe[(ecc >= lo) & (ecc < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
        a.plot((bins[:-1] + bins[1:]) / 2, med, color=INK, lw=2)
        a.set_yscale('log')
        a.set_ylim(0.03, 30)
        a.set_title(LABEL[c], color=INK)
        a.set_xlabel('true eccentricity (deg)')
        if k == 0:
            a.set_ylabel('position error (deg), line = median')
        a.axvline(9, color=INK2, lw=0.6, ls='--')
        a = ax[1, k]
        sel = np.random.default_rng(0).choice(len(ecc), 160, replace=False)
        for i in sel:
            a.plot([truth['x'][i], prm[i, 0]], [truth['y'][i], prm[i, 1]], color=C[c], lw=0.8, alpha=0.8)
        a.scatter(truth['x'][sel], truth['y'][sel], s=5, color=INK, zorder=3)
        circ = plt.Circle((0, 0), 9, fill=False, ls='--', color=INK2, lw=0.6)
        a.add_patch(circ)
        a.set_xlim(-14, 14)
        a.set_ylim(-14, 14)
        a.set_aspect('equal')
        a.set_xlabel('x (deg)')
        if k == 0:
            a.set_ylabel('y (deg)\n(dot = true centre, line → fit)')
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig3_recovery.png'), dpi=130)
    plt.close()


def fig_summary(res, conds):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    y = np.arange(len(conds))[::-1]
    names = [LABEL[c] for c in conds]
    m1 = [res[c]['pos_err_median'] for c in conds]
    m2 = [res[c]['frac_ceiling_game'] for c in conds]
    m3 = [res[c]['frac_ceiling_bar'] for c in conds]
    for a, vals, title, fmt in ((ax[0], m1, 'Median pRF position error (deg)', '{:.2f}'),
                                (ax[1], m2, 'Held-out GAME runs: R² / ceiling', '{:.2f}'),
                                (ax[2], m3, 'Held-out BAR run: R² / ceiling', '{:.2f}')):
        a.barh(y, vals, color=[C[c] for c in conds], height=0.6)
        for yi, v in zip(y, vals):
            a.text(v, yi, ' ' + fmt.format(v), va='center', fontsize=8, color=INK)
        a.set_yticks(y)
        a.set_yticklabels(names if a is ax[0] else [''] * len(y))
        a.set_title(title)
        a.set_xlim(0, max(vals) * 1.25 if a is ax[0] else 1.15)
        a.grid(axis='x', color=GRID, lw=0.6)
        a.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig4_summary.png'), dpi=130)
    plt.close()


def fig_quantity(res):
    series = {'Bars (fixation)': (['bars_1run', 'bars_2run', 'bars'], C['bars']),
              'Game, in-bore eye tracker': (['game_et_1run', 'game_et_2run', 'game_et_4run', 'game_eyetracker',
                                             'game_et_2h'], C['game_eyetracker'])}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for lab, (names, col) in series.items():
        names = [n for n in names if n in res]
        mins = [res[n]['minutes_train'] for n in names]
        ax[0].plot(mins, [res[n]['pos_err_median'] for n in names], '-o', color=col, lw=2, ms=6, label=lab)
        ax[1].plot(mins, [res[n]['by_ecc']['9-12.1']['pos_err'] for n in names], '-o', color=col, lw=2, ms=6)
        ax[2].plot(mins, [res[n]['frac_ceiling_bar'] for n in names], '-o', color=col, lw=2, ms=6)
    ax[0].set_title('Median position error, all voxels')
    ax[1].set_title('Median position error, ecc 9–12°')
    ax[2].set_title('Held-out bar run: R² / ceiling')
    for a in ax:
        a.set_xscale('log')
        a.set_xticks([5, 10, 15, 30, 60, 120])
        a.set_xticklabels(['5', '10', '15', '30', '60', '120'])
        a.set_xlabel('training data (minutes, log scale)')
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel('deg')
    for a, ticks in ((ax[0], [0.4, 0.6, 1, 2, 4, 8]), (ax[1], [0.5, 1, 2, 4])):
        a.set_yscale('log')
        a.set_yticks(ticks)
        a.set_yticklabels([f'{t:g}' for t in ticks])
        a.minorticks_off()
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(F, 'fig5_quantity.png'), dpi=130)
    plt.close()


def clip(seconds=30):
    fr = np.load(os.path.join(D, 'game0_frames.npy'))[: seconds * 10]
    ce = np.load(os.path.join(D, 'game0_ce.npy'), mmap_mode='r')
    gz = np.load(os.path.join(D, 'game0_gaze.npy'))
    cmap = plt.get_cmap('magma')
    out = os.path.join(F, 'sandbox_clip.mp4')
    h, w = 432, 576 + 432
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{w}x{h}', '-r', '10', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                             '-pix_fmt', 'yuv420p', '-crf', '23', out], stdin=subprocess.PIPE)
    for i in range(len(fr)):
        left = np.repeat(np.repeat(fr[i], 3, 0), 3, 1)
        left = np.stack([left] * 3, -1).copy()
        gx = int(288 + gz[i, 0] * 24)
        gy = int(216 - gz[i, 1] * 24)
        left[max(0, gy - 1):gy + 2, max(0, gx - 12):gx + 13] = (227, 73, 72)
        left[max(0, gy - 12):gy + 13, max(0, gx - 1):gx + 2] = (227, 73, 72)
        ret = st.retinal_stack(np.asarray(ce[i:i + 1], np.float32), gz[i:i + 1])[0][18:126, 18:126]
        ret = (cmap(np.clip(ret / 0.09, 0, 1))[..., :3] * 255).astype(np.uint8)
        ret = np.repeat(np.repeat(ret, 4, 0), 4, 1)
        proc.stdin.write(np.concatenate([left, ret], 1).tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == '__main__':
    truth = load('truth.npz')
    res = json.load(open(os.path.join(D, 'results.json')))
    main = [c for c in ('bars', 'game_truegaze', 'game_eyetracker', 'game_nogaze',
                        'game_et_noarousal', 'bars+game') if c in res]
    fig_sandbox()
    fig_signal(truth)
    fig_recovery(truth, [c for c in ('bars', 'game_truegaze', 'game_eyetracker', 'game_nogaze') if c in res])
    fig_summary(res, main)
    fig_quantity(res)
    clip()
    print('figures done')
