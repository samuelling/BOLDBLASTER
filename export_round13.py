"""Round 13 tidy CSVs: (1) pacing of calls, from logic-only hours of the Godot Halcyon (three separate
hours per setting, so every number has a spread); (2) one hour against three; (3) what moved the V1
spatial-frequency fit: the same playthroughs rendered normally, without the arms and pistol, and
without fog and bloom; (4) contrast by spatial-frequency band in the frames."""
import os, csv, json
import numpy as np

D, O, S = '/home/claude/BOLDBLASTER/data_v2', 'dashboard_data', '/home/claude/bbx/sweep'
CFG = [('t0', 'One group at a time'), ('t3', 'About 6 s (default)'), ('t2', 'Dense')]
MEAS = [('cog', ('ddm', 'a', 'bold'), 'Call boundary a', 'BOLD'), ('cog', ('ddm', 'v', 'bold'), 'Call drift v', 'BOLD'),
        ('cog', ('rl', 'alpha', 'bold'), 'Value learning rate', 'BOLD'), ('new', ('bandit', 'alpha', 'bold'), 'Stairwell learning rate', 'BOLD'),
        ('new', ('pathint', 'sh', 'bold'), 'Path-integration heading noise', 'BOLD'), ('new', ('mapupd', 'alpha', 'bold'), 'Map update rate', 'BOLD'),
        ('cog', ('novelty', 'f', 'bold'), 'Place forgetting rate', 'BOLD'), ('nav', ('boundary', 'd'), 'Distance to walls', 'BOLD'),
        ('nav', ('grid_move', 'phi'), 'Grid orientation', 'BOLD'),
        ('cog', ('ddm', 'a', 'behaviour'), 'Call boundary a', 'behaviour'), ('cog', ('ddm', 'v', 'behaviour'), 'Call drift v', 'behaviour'),
        ('new', ('bandit', 'alpha', 'behaviour'), 'Stairwell learning rate', 'behaviour'), ('new', ('pathint', 'sh', 'behaviour'), 'Path-integration heading noise', 'behaviour')]


def get(d, ks):
    for k in ks:
        d = d[k]
    return float(d)


def res(name, kind, mins=60):
    return json.load(open(f'{D}/results_sweep_{name}_{kind}.json'))[f'halcyon_{name}_{mins}']


def gaps(name, n=12):
    calls, g = 0, []
    for i in range(n):
        t = [c[0] for c in json.load(open(f'{S}/{name}/halcyon_{name}{i}_events.json'))['calls']]
        calls += len(t); g += list(np.diff(t))
    return calls * 12 / n, float(np.median(g)), float(np.percentile(g, 10))


pm = lambda v: f'{np.mean(v):.2f} ± {np.std(v):.2f}'
rows = []
st = {c: np.array([gaps(f'{c}r{j}') for j in range(3)]) for c, _ in CFG}
rows.append(['Calls per hour', ''] + [f'{st[c][:, 0].mean():.0f}' for c, _ in CFG])
rows.append(['Median gap between calls (s)', ''] + [f'{st[c][:, 1].mean():.1f}' for c, _ in CFG])
rows.append(['Shortest tenth of gaps, under (s)', ''] + [f'{st[c][:, 2].mean():.1f}' for c, _ in CFG])
for kind, ks, label, src in MEAS:
    rows.append([label, src] + [pm([get(res(f'{c}r{j}', kind), ks) for j in range(3)]) for c, _ in CFG])
with open(f'{O}/round13_pacing.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['measure', 'from'] + [c for c, _ in CFG]); w.writerows(rows)

rows = []
for kind, ks, label, src in MEAS:
    if kind == 'nav':
        continue
    rows.append([label, src] + [f'{get(res(n, kind, m), ks):.2f}' for n in ('h0', 'h2') for m in (60, 180)])
with open(f'{O}/round13_hours.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['measure', 'from', 'spaced_1h', 'spaced_3h', 'dense_1h', 'dense_3h']); w.writerows(rows)

# vision: the same six playthroughs under three renders, 30 minutes each, beside rounds 11 and 12 at 30 minutes
r8 = json.load(open(f'{D}/results_round8.json'))
VL = dict(c50_eff='Effective c50', mu='Spatial-frequency peak', sig='Spatial-frequency bandwidth', q='Surround exponent',
          k='Surround strength', tau='Temporal constant', a='Attention gain')
cols = [('_va_20m', 'normal'), ('_vb_20m', 'no_arms'), ('_vc_20m', 'no_haze'), ('hg_30m', 'godot_r12_30m'), ('hx_30m', 'browser_r11_30m')]   # the three renders: 20 minutes each, the same four playthroughs
if all(k in r8 for k, _ in cols):
    with open(f'{O}/round13_vision.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['parameter'] + [c for _, c in cols])
        for p_, lab in VL.items():
            w.writerow([lab] + [f"{r8[k]['scores'][p_]['spearman']:.2f}" for k, _ in cols])
    print('vision written')
print(open(f'{O}/round13_pacing.csv').read()); print(open(f'{O}/round13_hours.csv').read())


# contrast by spatial-frequency band in the frames (run 0 of each; 8 pixels per degree)
def bands(path):
    a = np.fromfile(path, '<u2').reshape(-1, 144, 192).astype(np.float32) / 65535
    H, W, ppd = 144, 192, 8.0
    fr = np.hypot(np.fft.fftfreq(W)[None, :] * ppd, np.fft.fftfreq(H)[:, None] * ppd)
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    E = [0.125, 0.25, 0.5, 1, 2, 4]; out = []
    for f in a[::5]:
        m = f.mean(); F = np.abs(np.fft.fft2((f - m) / max(m, 1e-3) * win)) ** 2
        out.append([np.sqrt(F[(fr >= lo) & (fr < hi)].sum()) / (H * W) * 4 for lo, hi in zip(E[:-1], E[1:])])
    b = np.array(out); c = np.corrcoef(np.log(b + 1e-6).T)
    return b.mean(0), c[0, 4]


BL = ['0.125-0.25', '0.25-0.5', '0.5-1', '1-2', '2-4']
src = [('browser_r11', '/home/claude/bbx/runs_hx/halcyon0_lum.u16'), ('godot_r12', '/home/claude/bbx/runs_gd/halcyon0_lum.u16'),
       ('normal', '/home/claude/bbx/runs_v/halcyon_va0_lum.u16'), ('no_arms', '/home/claude/bbx/runs_v/halcyon_vb0_lum.u16'), ('no_haze', '/home/claude/bbx/runs_v/halcyon_vc0_lum.u16')]
if all(os.path.exists(q) for _, q in src):
    B = {k: bands(q) for k, q in src}
    with open(f'{O}/round13_bands.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['band', 'unit'] + [k for k, _ in src])
        for n, lab in enumerate(BL):
            w.writerow([f'Contrast at {lab} cycles per degree', ''] + [f'{B[k][0][n]:.3f}' for k, _ in src])
        w.writerow(['Lowest and highest band rise and fall together', 'r'] + [f'{B[k][1]:.2f}' for k, _ in src])
    print(open(f'{O}/round13_bands.csv').read())
