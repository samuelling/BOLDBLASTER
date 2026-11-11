"""v2 ground-truth models and their fitters.

V1 'rich' voxel (per voxel: pRF from v1 truth, plus 7 planted parameters):
    w_b      = log-Gaussian SF tuning over bands (mu = log2 peak cpd, sig = octaves)   [pSFT]
    Dc,Ds,Dt = SF-weighted centre / surround (3x sigma) / transient (|dE/dt|) energy
    R = (Dc^q + tau*Dt^q) / (c50^q + Dc^q + k*Ds^q) * (1 + a*Att)
          c50, q  [pCRF]   k [normalization / surround suppression]
          tau     [sustained vs transient]   a [attention gain]
Numerosity voxel: log-Gaussian tuning to number (mu_N, sig_N), silent for N = 0.
Self-motion (MST-like) voxel: wf*forward + wl*rotate-left + wr*rotate-right.
Striatal voxel: g*outcome-onset + b*RPE, RPE = outcome - p(hit).
All BOLD: amplitude * (drive conv HRF) + systemic arousal + AR(1) noise + drift.
"""
import numpy as np
from scipy.optimize import least_squares
from .bands import BAND_F, NB
from .sim import hrf, _conv, to_tr, nuisance_bold
from .game import FPS

LF = np.log2(BAND_F)
PNAMES = ['mu', 'sig', 'c50', 'q', 'k', 'tau', 'a']
DEFAULT = dict(mu=0.0, sig=1.1, c50=0.2, q=2.0, k=0.5, tau=0.4, a=0.0)
BOUNDS = dict(mu=(-2.5, 2.0), sig=(0.3, 3.0), c50=(0.01, 3.0), q=(0.8, 4.0), k=(0.0, 4.0),
              tau=(0.0, 3.0), a=(-0.5, 1.5))


def sample_v1(rng, n):
    ecc = rng.uniform(0, 12, n)  # placeholder, overwritten by caller with pRF ecc
    return dict(
        sig=rng.uniform(0.8, 1.4, n), c50=np.exp(rng.normal(np.log(0.2), 0.4, n)),
        q=rng.uniform(1.5, 2.5, n), k=rng.uniform(0, 1.5, n), tau=rng.uniform(0, 1.0, n),
        a=rng.uniform(0, 0.4, n))


def plant_mu(rng, ecc):
    return np.log2(2.0 / (1 + ecc / 5)) + rng.normal(0, 0.3, len(ecc))


def sf_weights(mu, sig):
    w = np.exp(-(LF[None, :] - np.asarray(mu)[:, None]) ** 2 / (2 * np.asarray(sig)[:, None] ** 2))
    return w / w.sum(1, keepdims=True)


def v1_drive(Ec, Es, Tc, att, P):
    """Ec,Es,Tc: (n,V,NB); att (n,V); P: dict of (V,) arrays -> (n,V)."""
    w = sf_weights(P['mu'], P['sig'])
    Dc = np.einsum('nvb,vb->nv', Ec, w)
    Ds = np.einsum('nvb,vb->nv', Es, w)
    Dt = np.einsum('nvb,vb->nv', Tc, w)
    q = P['q'][None]
    num = Dc ** q + P['tau'][None] * Dt ** q
    den = P['c50'][None] ** q + Dc ** q + P['k'][None] * Ds ** q
    return num / den * (1 + P['a'][None] * att)


def attention_trace(salient, gaze, x, y, sigma=2.0, dur=1.5):
    """(nf,V) attention field value at each pRF centre from salient onsets."""
    nf = len(gaze)
    att = np.zeros((nf, len(x)), np.float32)
    for t, ex, ey, _ in salient:
        f0, f1 = int(t * FPS), min(nf, int((t + dur) * FPS))
        if f1 <= f0:
            continue
        rx = ex - gaze[f0:f1, 0]
        ry = ey - gaze[f0:f1, 1]
        d2 = (x[None] - rx[:, None]) ** 2 + (y[None] - ry[:, None]) ** 2
        att[f0:f1] += np.exp(-d2 / (2 * sigma ** 2))
    return np.minimum(att, 1.0)


def attention_blocks(side, x):
    """Controlled attention: side (nf,) in {-1,0,1}; pRF x (V,) -> (nf,V)."""
    return (np.abs(side)[:, None] / (1 + np.exp(-side[:, None] * x[None] / 0.5))).astype(np.float32)


def bold_from_drive(drive10, vox_hrf, amp):
    """drive at 10 Hz (n,V) -> (TR,V) % with per-voxel HRF scaling."""
    out = np.empty_like(drive10)
    for v in range(drive10.shape[1]):
        out[:, v] = _conv(drive10[:, v], hrf(1 / FPS, vox_hrf[v]))
    return to_tr(out) * amp[None]


def add_nuisance(rng, vis, vox, events, duration=300.0):
    sysm, noise = nuisance_bold(rng, vox, vis.shape[0], events, duration)
    return (vis + sysm + noise).astype(np.float32)


def bin2hz(x):
    n = x.shape[0] // 5
    return x[:n * 5].reshape(n, 5, *x.shape[1:]).mean(1)


# ----------------------------------------------------------------- fitting machinery
H2 = hrf(0.5)


def _to_bold(d2hz, n_tr, Q, h=H2):
    b = np.convolve(d2hz, h)[:n_tr * 3].reshape(n_tr, 3).mean(1)
    return b - Q @ (Q.T @ b)


class V1Fitter:
    """Per-voxel nonlinear least squares with the gain solved analytically."""

    def __init__(self, runs, free):
        # runs: list of dict(Ec,Es,Tc (T,V,NB) at 10 Hz, att (T,V), Y (TR,V), Q)
        self.runs, self.free = runs, list(free) + ['hrf']     # HRF time-scale always fitted
        self.edges = np.cumsum([0] + [r['Ec'].shape[0] for r in runs])

    def fit_voxel(self, v, fixed, starts):
        runs = self.runs
        Ec = np.concatenate([r['Ec'][:, v] for r in runs])
        Es = np.concatenate([r['Es'][:, v] for r in runs])
        Tc = np.concatenate([r['Tc'][:, v] for r in runs])
        att = np.concatenate([r['att'][:, v] for r in runs])
        y = np.concatenate([r['Y'][:, v] - r['Q'] @ (r['Q'].T @ r['Y'][:, v]) for r in runs])
        bnd = dict(BOUNDS, hrf=(0.75, 1.3))
        lo = np.array([bnd[k][0] for k in self.free])
        hi = np.array([bnd[k][1] for k in self.free])
        edges = self.edges

        def pred(P):
            w = np.exp(-(LF - P['mu']) ** 2 / (2 * P['sig'] ** 2))
            w /= w.sum()
            Dc, Ds, Dt = Ec @ w, Es @ w, Tc @ w
            q = P['q']
            Dcq = Dc ** q
            d = (Dcq + P['tau'] * Dt ** q) / (P['c50'] ** q + Dcq + P['k'] * Ds ** q) * (1 + P['a'] * att)
            h = hrf(0.5, P['hrf'])
            out = []
            for i, r in enumerate(runs):
                seg = d[edges[i]:edges[i + 1]]
                seg = seg[:len(seg) // 5 * 5].reshape(-1, 5).mean(1)      # 10 Hz -> 2 Hz after the nonlinearity
                out.append(_to_bold(seg, r['Y'].shape[0], r['Q'], h))
            return np.concatenate(out)

        def res(theta):
            P = dict(fixed)
            P.update(zip(self.free, theta))
            p = pred(P)
            g = max(p @ y / max(p @ p, 1e-12), 0)
            return y - g * p

        best = None
        for s in starts:
            s = dict(s, hrf=1.0)
            x0 = np.clip([s[k] for k in self.free], lo + 1e-6, hi - 1e-6)
            try:
                r = least_squares(res, x0, bounds=(lo, hi), max_nfev=60, x_scale='jac')
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        out = dict(fixed)
        if best is not None:
            out.update(zip(self.free, best.x))
        return out


def starts_for(free, rng, n=3):
    base = dict(DEFAULT)
    out = [dict(base)]
    for _ in range(n - 1):
        s = dict(base)
        s.update(mu=rng.uniform(-1.5, 1.2), sig=rng.uniform(0.7, 1.6), c50=np.exp(rng.uniform(-2.5, -0.7)),
                 q=rng.uniform(1.2, 3.0), k=rng.uniform(0, 1.5), tau=rng.uniform(0, 1.2), a=rng.uniform(0, 0.4))
        out.append(s)
    return out


# ----------------------------------------------------------------- numerosity
def num_tuning(N, mu, sig):
    N = np.asarray(N, float)
    with np.errstate(divide='ignore'):
        r = np.exp(-(np.log(np.maximum(N, 1e-9))[:, None] - np.log(mu)[None]) ** 2 / (2 * sig[None] ** 2))
    r[N < 1] = 0
    return r


def fit_numerosity(runs):
    """runs: list of dict(N (frames,), Y (TR,V), Q). Grid search -> (mu, sig)."""
    mus = np.geomspace(0.8, 12, 48)
    sigs = np.geomspace(0.12, 1.6, 24)
    M, S = np.meshgrid(mus, sigs)
    M, S = M.ravel(), S.ravel()
    P, Y = [], []
    for r in runs:
        d = bin2hz(num_tuning(r['N'], M, S))           # (T, C)
        n_tr = r['Y'].shape[0]
        b = np.stack([_to_bold(d[:, c], n_tr, r['Q']) for c in range(d.shape[1])], 1)
        P.append(b)
        Y.append(r['Y'] - r['Q'] @ (r['Q'].T @ r['Y']))
    P, Y = np.concatenate(P), np.concatenate(Y)
    Pn = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    Yn = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    R = Pn.T @ Yn
    i = R.argmax(0)
    return M[i], S[i]


# ----------------------------------------------------------------- linear GLM helper
def glm(regs_runs, Y_runs, Q_runs):
    """regs_runs: list of (TR,K). Returns betas (K,V) after nuisance projection."""
    X = np.concatenate([X - Q @ (Q.T @ X) for X, Q in zip(regs_runs, Q_runs)])
    Y = np.concatenate([Y - Q @ (Q.T @ Y) for Y, Q in zip(Y_runs, Q_runs)])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def hrf_regressor(x10hz):
    """10 Hz regressor -> TR-sampled HRF-convolved regressor."""
    return to_tr(_conv(np.asarray(x10hz, float), hrf(1 / FPS))[:, None])[:, 0]


def impulses(times, values, nf):
    x = np.zeros(nf)
    for t, v in zip(times, values):
        i = int(t * FPS)
        if i < nf:
            x[i] += v
    return x


def events_1s(times, values, nf, dur=1.0):
    """1 s neural events (outcome responses), 10 Hz."""
    x = np.zeros(nf)
    k = int(dur * FPS)
    for t, v in zip(times, values):
        i = int(t * FPS)
        x[i:min(nf, i + k)] += v
    return x
