"""CSS pRF fitting on retinal contrast-energy stimuli.

Model: BOLD = g * [ (G(x,y,sigma) . CE)^n  conv  canonical HRF ]  + nuisance
Nuisance per run: Legendre 0-3 (+ optional arousal regressors from logged game
events: canonical arousal kernel and its derivative). Nuisance is projected out of
both data and predictions; fits maximize correlation (gain solved analytically).

Search: coarse polar grid (28 ecc x 36 angle x 16 sizes x 4 exponents = 64,512
candidates), then batched local refinement (3 rounds of 3^4 perturbations/voxel).
"""
import numpy as np
from .sim import gaussians, hrf, arousal_kernel, _conv, TR

DT = 0.5
BPT = 3                     # bins per TR
H_FIT = hrf(DT)


def nuisance_basis(n_tr, events=None, arousal=False):
    tt = np.linspace(-1, 1, n_tr)
    cols = [np.polynomial.legendre.Legendre.basis(k)(tt) for k in range(4)]
    if arousal and events is not None and len(events):
        nb = n_tr * BPT
        imp = np.zeros(nb)
        for t, *_ in events:
            imp[min(int(t / DT), nb - 1)] += 1
        k = arousal_kernel(DT, peak=7.0)            # deliberately not the true 8 s kernel
        r = _conv(imp, k)
        d = np.gradient(r)
        for z in (r, d):
            cols.append(z[:n_tr * BPT].reshape(n_tr, BPT).mean(1))
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q.astype(np.float32)


def resid(Y, Q):          # Y: (T, V)
    return Y - Q @ (Q.T @ Y)


def _bold_from_drive(d, n_tr, Q):
    """d: (C, bins) neural drive -> residualized BOLD (C, n_tr)."""
    b = _conv(d, H_FIT, axis=1)[:, :n_tr * BPT].reshape(d.shape[0], n_tr, BPT).mean(2)
    return (b - (b @ Q) @ Q.T).astype(np.float32)


def predict(x, y, sig, n, S_runs, Q_runs, chunk=1500):
    x, y, sig, n = map(np.atleast_1d, (x, y, sig, n))
    C = len(x)
    ntrs = [Q.shape[0] for Q in Q_runs]
    out = np.empty((C, sum(ntrs)), np.float32)
    for c0 in range(0, C, chunk):
        sl = slice(c0, min(C, c0 + chunk))
        G = gaussians(x[sl], y[sl], sig[sl], step=2)
        cols = []
        for S, Q, nt in zip(S_runs, Q_runs, ntrs):
            lin = np.maximum(G @ S.T, 0)
            cols.append(_bold_from_drive(lin ** n[sl, None], nt, Q))
        out[sl] = np.concatenate(cols, 1)
    return out


def _prep_data(Y_runs, Q_runs):
    Yres = np.concatenate([resid(Y, Q) for Y, Q in zip(Y_runs, Q_runs)]).astype(np.float32)
    nrm = np.linalg.norm(Yres, axis=0)
    return Yres, Yres / nrm


def grid_fit(S_runs, Y_runs, Q_runs, verbose=False):
    ecc = np.geomspace(0.3, 16, 28)
    ang = np.deg2rad(np.arange(0, 360, 10))
    E, A = np.meshgrid(ecc, ang)
    xs, ys = (E * np.cos(A)).ravel(), (E * np.sin(A)).ravel()
    sizes = np.geomspace(0.15, 8, 16)
    exps = (0.5, 0.65, 0.8, 1.0)
    _, Yn = _prep_data(Y_runs, Q_runs)
    V = Yn.shape[1]
    best_r = np.full(V, -np.inf, np.float32)
    best = np.zeros((V, 4), np.float32)
    ntrs = [Q.shape[0] for Q in Q_runs]
    for sz in sizes:
        G = gaussians(xs, ys, np.full(len(xs), sz), step=2)
        lins = [np.maximum(G @ S.T, 0) for S in S_runs]
        for e in exps:
            P = np.concatenate([_bold_from_drive(l ** e, nt, Q)
                                for l, nt, Q in zip(lins, ntrs, Q_runs)], 1)
            nr = np.linalg.norm(P, axis=1, keepdims=True)
            P = P / np.maximum(nr, 1e-12)
            R = P @ Yn
            idx = R.argmax(0)
            r = R[idx, np.arange(V)]
            u = r > best_r
            best_r[u] = r[u]
            best[u] = np.c_[xs[idx[u]], ys[idx[u]], np.full(u.sum(), sz), np.full(u.sum(), e)]
        if verbose:
            print(f'  size {sz:.2f} done, median r {np.median(best_r):.3f}', flush=True)
    return best, best_r


def refine(params, S_runs, Y_runs, Q_runs, rounds=3, verbose=False):
    _, Yn = _prep_data(Y_runs, Q_runs)
    V = Yn.shape[1]
    p = params.copy()
    steps = np.array(np.meshgrid(*[[-1, 0, 1]] * 4, indexing='ij')).reshape(4, -1).T  # (81,4)
    K = len(steps)
    cur_r = None
    for it in range(rounds):
        f = 0.5 ** it
        ecc = np.hypot(p[:, 0], p[:, 1])
        dpos = np.maximum(0.08, 0.09 * ecc) * f
        x = p[:, None, 0] + steps[None, :, 0] * dpos[:, None]
        y = p[:, None, 1] + steps[None, :, 1] * dpos[:, None]
        s = np.clip(p[:, None, 2] * np.exp(steps[None, :, 2] * 0.15 * f), 0.05, 12)
        n = np.clip(p[:, None, 3] + steps[None, :, 3] * 0.08 * f, 0.25, 1.3)
        P = predict(x.ravel(), y.ravel(), s.ravel(), n.ravel(), S_runs, Q_runs)
        P = P.reshape(V, K, -1)
        P /= np.maximum(np.linalg.norm(P, axis=2, keepdims=True), 1e-12)
        R = np.einsum('vkt,tv->vk', P, Yn)
        k = R.argmax(1)
        p = np.c_[x[np.arange(V), k], y[np.arange(V), k], s[np.arange(V), k], n[np.arange(V), k]]
        cur_r = R[np.arange(V), k]
        if verbose:
            print(f'  refine round {it}: median r {np.median(cur_r):.3f}', flush=True)
    return p.astype(np.float32), cur_r


def gains(params, S_runs, Y_runs, Q_runs):
    Yres, _ = _prep_data(Y_runs, Q_runs)
    P = predict(*params.T, S_runs, Q_runs)
    return np.maximum((P * Yres.T).sum(1) / np.maximum((P * P).sum(1), 1e-12), 0)


def cv_r2(params, g, S_runs, Y_runs, Q_runs):
    Yres, _ = _prep_data(Y_runs, Q_runs)
    P = predict(*params.T, S_runs, Q_runs)
    err = Yres - (g[:, None] * P).T
    return 1 - (err ** 2).sum(0) / (Yres ** 2).sum(0)


def ceiling_r2(Yvis_runs, Y_runs, Q_runs):
    Yres, _ = _prep_data(Y_runs, Q_runs)
    Vres = np.concatenate([resid(Yv, Q) for Yv, Q in zip(Yvis_runs, Q_runs)])
    return 1 - ((Yres - Vres) ** 2).sum(0) / (Yres ** 2).sum(0)
