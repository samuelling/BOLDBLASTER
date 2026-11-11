"""Round-4 summary with bootstrap CIs on the same 120-voxel subset for every design."""
import json
import os
import numpy as np
from scipy.stats import spearmanr
import run_v2 as V2
import run_long as L

PARAMS = ['mu', 'sig', 'c50_eff', 'q', 'k', 'tau', 'a']
DED = dict(mu='psft', sig='psft', c50_eff='crf', q='crf', k='size', tau='temporal', a='attn')
NB = 2000


def values(est, prm, n):
    if prm == 'c50_eff':
        return est['c50'][:n] / (1 + est['k'][:n]) ** (1 / est['q'][:n])
    return est[prm][:n]


def main():
    pop = V2.populations()
    n = L.NV
    truth = {p: values({k: pop[f'v1_{k}'] for k in ('c50', 'k', 'q', 'mu', 'sig', 'tau', 'a')}, p, n) for p in PARAMS}
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n, n) for _ in range(NB)]
    est = {}
    for name in ('long_2h', 'single_4h', 'varied_4h', 'long_5h', 'long_10h', 'long_20h'):
        f = L.p(f'fit_{name}.npz')
        if os.path.exists(f):
            est[name] = dict(np.load(f))
    for prm in PARAMS:
        est.setdefault(f'ded_{prm}', {})
    out = {}

    def ci(t, e):
        r = spearmanr(t, e).correlation
        bs = [spearmanr(t[b], e[b]).correlation for b in boots[:500]]
        return dict(r=float(r), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

    for name, e in est.items():
        if name.startswith('ded_'):
            continue
        out[name] = {p: ci(truth[p], values(e, p, n)) for p in PARAMS}
    ded = {}
    for p in PARAMS:
        e = dict(np.load(os.path.join(V2.D, f'fit_dedicated_{DED[p]}.npz')))
        if p == 'c50_eff':
            e['k'] = np.full(len(e['c50']), 0.5)
        ded[p] = ci(truth[p], values(e, p, n))
    out['dedicated_10min'] = ded
    if 'varied_4h' in est and 'single_4h' in est:
        diff = {}
        for p in PARAMS:
            t, a, b = truth[p], values(est['varied_4h'], p, n), values(est['single_4h'], p, n)
            d = [spearmanr(t[bb], a[bb]).correlation - spearmanr(t[bb], b[bb]).correlation for bb in boots[:500]]
            diff[p] = dict(diff=float(spearmanr(t, a).correlation - spearmanr(t, b).correlation),
                           lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
        out['varied_minus_single_4h'] = diff
    json.dump(out, open(L.p('summary.json'), 'w'), indent=1)
    for k, v in out.items():
        print(k, {p: (round(x.get('r', x.get('diff')), 2), round(x['lo'], 2), round(x['hi'], 2)) for p, x in v.items()})


if __name__ == '__main__':
    main()
