"""Sensitivity check: how much does adding a transient (temporal-derivative) V1
channel change gameplay-driven modulation relative to bars?
Drive_sust = G . CE_ret(t);  Drive_trans = G . |CE_ret(t) - CE_ret(t-1)|  (10 Hz, retinal)
Saccades shift the whole retinal image, so they show up in the transient channel."""
import json, numpy as np
import run_experiment as R
from boldblaster import sim, stimuli as st

t = dict(np.load(R.p('truth.npz')))
sel = np.random.default_rng(3).choice(500, 200, replace=False)
vox = {k: t[k][sel] for k in ('x', 'y', 'sigma', 'n', 'hrf_scale', 'amp')}
G = sim.gaussians(vox['x'], vox['y'], vox['sigma'], step=1)


def drives(kind, i):
    ce, gz, _ = R.load_run(kind, i)
    sus, tra, prev = [], [], None
    for s in range(0, ce.shape[0], 500):
        stk = st.retinal_stack(np.asarray(ce[s:s + 500], np.float32), gz[s:s + 500])
        flat = stk.reshape(stk.shape[0], -1)
        d = np.abs(np.diff(np.concatenate([flat[:1] if prev is None else prev, flat]), axis=0))
        prev = flat[-1:]
        sus.append(flat @ G.T)
        tra.append(d @ G.T)
    return np.concatenate(sus), np.concatenate(tra)


def mod_sd(d):
    v = {'n': vox['n'], 'hrf_scale': vox['hrf_scale'], 'amp': np.ones(len(sel))}
    b = sim.visual_bold(d, v, np.ones(len(sel)))
    return b.std(0)


out = {}
bs, bt = drives('bar', 0)
gs_, gt = drives('game', 0)
for name, (b, g) in {'sustained': (bs, gs_), 'transient': (bt, gt)}.items():
    rb, rg = mod_sd(b), mod_sd(g)
    out[name] = dict(game_over_bar_modulation=float(np.median(rg / rb)))
# 50/50 mixture with each channel normalised to its bar-run response
mix_b = bs / np.percentile(bs, 95, 0) + bt / np.percentile(bt, 95, 0)
mix_g = gs_ / np.percentile(bs, 95, 0) + gt / np.percentile(bt, 95, 0)
out['mixture_50_50'] = dict(game_over_bar_modulation=float(np.median(mod_sd(mix_g) / mod_sd(mix_b))))
print(json.dumps(out, indent=1))
json.dump(out, open(R.p('transient_check.json'), 'w'), indent=1)
