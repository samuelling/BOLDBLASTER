"""Which nuisance hurts game-based pRF fits: measurement noise or systemic arousal?
Grid-only fits on 120 voxels, 6 game runs, true gaze."""
import json, numpy as np
import run_experiment as R
from boldblaster import fit

t = dict(np.load(R.p('truth.npz')))
sel = np.random.default_rng(1).choice(500, 120, replace=False)
idx = range(6)
S = [np.load(R.p(f'S_game{i}_true.npy')) for i in idx]
ev = [np.load(R.p(f'game{i}_events.npy')) for i in idx]
vis = [t[f'game{i}_vis'][:, sel] for i in idx]
sysm = [t[f'game{i}_sys'][:, sel] for i in idx]
full = [t[f'game{i}_Y'][:, sel] for i in idx]
noise = [f - v - s for f, v, s in zip(full, vis, sysm)]
cases = {
    'visual only': (vis, False),
    'visual + systemic (arousal regressor)': ([v + s for v, s in zip(vis, sysm)], True),
    'visual + systemic (no regressor)': ([v + s for v, s in zip(vis, sysm)], False),
    'visual + noise': ([v + n for v, n in zip(vis, noise)], False),
    'full (arousal regressor)': (full, True),
}
out = {}
for name, (Y, ar) in cases.items():
    Q = [fit.nuisance_basis(200, e, ar) for e in ev]
    prm, r = fit.grid_fit(S, Y, Q)
    pe = np.hypot(prm[:, 0] - t['x'][sel], prm[:, 1] - t['y'][sel])
    se = np.log2((prm[:, 2] / np.sqrt(prm[:, 3])) / (t['sigma'][sel] / np.sqrt(t['n'][sel])))
    out[name] = dict(pos_err=float(np.median(pe)), size_log2=float(np.median(se)))
    print(name, out[name], flush=True)
json.dump(out, open(R.p('decompose_check.json'), 'w'), indent=1)
