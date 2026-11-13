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
    for t, ex, ey, kind in salient:
        d = 2.0 if kind == 2 else dur          # covert-attention episodes last 2 s
        f0, f1 = int(t * FPS), min(nf, int((t + d) * FPS))
        if f1 <= f0:
            continue
        rx = ex - gaze[f0:f1, 0]
        ry = ey - gaze[f0:f1, 1]
        d2 = (x[None] - rx[:, None]) ** 2 + (y[None] - ry[:, None]) ** 2
        att[f0:f1] += np.exp(-d2 / (2 * sigma ** 2))
    return np.minimum(att, 1.0)


def attention_impulses(salient, gaze, x, y, sigma=2.0):
    """(nf,V) impulse train: spatial Gaussian weight at the single onset frame of each
    salient event (NPC-appear / kill / covert cue), summed when events coincide.
    Feeds attention_dynamic; replaces the fixed-duration box of attention_trace with
    a per-voxel temporal kernel applied downstream."""
    nf = len(gaze)
    imp = np.zeros((nf, len(x)), np.float32)
    for t, ex, ey, kind in salient:
        f0 = int(round(t * FPS))
        if f0 >= nf:
            continue
        rx = ex - gaze[f0, 0]
        ry = ey - gaze[f0, 1]
        d2 = (x - rx) ** 2 + (y - ry) ** 2
        imp[f0] += np.exp(-d2 / (2 * sigma ** 2))
    return imp


def attn_kernel(tr, td, fps=FPS, length=8.0):
    """Per-voxel asymmetric rise(tr)/decay(td) kernel, peak-normalized to 1. tr, td: (V,)."""
    nk = int(length * fps)
    t = (np.arange(nk) / fps)[:, None]
    tr, td = np.maximum(np.asarray(tr, float), 1e-3), np.maximum(np.asarray(td, float), 1e-3)
    K = (1 - np.exp(-t / tr[None, :])) * np.exp(-t / td[None, :])
    return (K / np.maximum(K.max(0, keepdims=True), 1e-9)).astype(np.float32)


def attention_dynamic(imp, tr, td, fps=FPS, length=8.0):
    """imp (nf,V) event impulses -> (nf,V) attention-gain field via convolution with each
    voxel's own rise/decay kernel (exogenous, event-locked attention). Vectorized over
    voxels and time via an explicit tap loop (kernel is short: length*fps samples)."""
    K = attn_kernel(tr, td, fps, length)          # (nk, V)
    nk = K.shape[0]
    out = np.zeros_like(imp)
    for l in range(nk):
        if l == 0:
            out += imp * K[0]
        else:
            out[l:] += imp[:-l] * K[l]
    return np.minimum(out, 1.0)


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
        grp = np.concatenate([np.full(r['Y'].shape[0], hash(r.get('group', 0)) % 100003) for r in runs])
        gids = [grp == u for u in np.unique(grp)]          # one response gain per scan session
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
            out = y.copy()
            for gm in gids:
                pg = p[gm]
                out[gm] -= max(pg @ y[gm] / max(pg @ pg, 1e-12), 0) * pg
            return out

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


# ----------------------------------------------------------------- attention dynamics
# Isolates recovery of the attention-gain time course (a, rise tr, decay td) with every
# other V1 mechanism held at M.DEFAULT (not ground truth) -- the same "one model at a
# time, generic elsewhere" convention as the dedicated single-parameter fits above.
ATTN_DEFAULT = dict(a=0.2, tr=0.25, td=1.0)
ATTN_BOUNDS = dict(a=(-0.5, 1.5), tr=(0.02, 1.5), td=(0.1, 5.0))


def core_drive(Ec, Es, Tc, mu=None, sig=None, c50=None, q=None, k=None, tau=None):
    """(nf,V,NB) features -> (nf,V) attention-free drive. Params default to the single,
    generic M.DEFAULT value (same scalar for every voxel -- the 'nothing else known' case),
    but may instead be given as per-voxel (V,) arrays, e.g. plugged in from a prior joint
    fit on the same data, to isolate attention-dynamics recovery without also forcing a
    generic-tuning mismatch onto every voxel."""
    V = Ec.shape[1]
    d = DEFAULT
    mu = np.full(V, d['mu']) if mu is None else np.asarray(mu)
    sig = np.full(V, d['sig']) if sig is None else np.asarray(sig)
    c50 = np.full(V, d['c50']) if c50 is None else np.asarray(c50)
    q = np.full(V, d['q']) if q is None else np.asarray(q)
    k = np.full(V, d['k']) if k is None else np.asarray(k)
    tau = np.full(V, d['tau']) if tau is None else np.asarray(tau)
    w = sf_weights(mu, sig)                        # (V, NB)
    Dc = np.einsum('nvb,vb->nv', Ec, w)
    Ds = np.einsum('nvb,vb->nv', Es, w)
    Dt = np.einsum('nvb,vb->nv', Tc, w)
    num = Dc ** q[None] + tau[None] * Dt ** q[None]
    den = c50[None] ** q[None] + Dc ** q[None] + k[None] * Ds ** q[None]
    return (num / den).astype(np.float32)


class AttnDynFitter:
    """Per-voxel fit of (a, tr, td) against a precomputed attention-free 'core' drive and
    a precomputed event-impulse train (both at 10 Hz); the dynamic attention-gain field is
    recomputed from (tr, td) on every optimizer step."""

    def __init__(self, runs, free=('a', 'tr', 'td')):
        # runs: dict(core (nf,V), imp (nf,V), Y (TR,V), Q)
        self.runs, self.free = runs, list(free)
        self.edges = np.cumsum([0] + [r['imp'].shape[0] for r in runs])

    def fit_voxel(self, v, fixed, starts):
        runs = self.runs
        core = np.concatenate([r['core'][:, v] for r in runs])
        imp = np.concatenate([r['imp'][:, v] for r in runs])
        y = np.concatenate([r['Y'][:, v] - r['Q'] @ (r['Q'].T @ r['Y'][:, v]) for r in runs])
        lo = np.array([ATTN_BOUNDS[k][0] for k in self.free])
        hi = np.array([ATTN_BOUNDS[k][1] for k in self.free])
        edges = self.edges
        h = hrf(0.5, 1.0)

        def pred(P):
            attn = attention_dynamic(imp[:, None], np.array([P['tr']]), np.array([P['td']]))[:, 0]
            d = core * (1 + P['a'] * attn)
            out = []
            for i, r in enumerate(runs):
                seg = d[edges[i]:edges[i + 1]]
                seg = seg[:len(seg) // 5 * 5].reshape(-1, 5).mean(1)
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
            x0 = np.clip([s[k] for k in self.free], lo + 1e-6, hi - 1e-6)
            try:
                r = least_squares(res, x0, bounds=(lo, hi), max_nfev=50, x_scale='jac')
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        out = dict(fixed)
        if best is not None:
            out.update(zip(self.free, best.x))
        return out


def attn_starts(rng, n=4):
    base = dict(ATTN_DEFAULT)
    out = [dict(base)]
    for _ in range(n - 1):
        out.append(dict(a=rng.uniform(0.0, 0.6), tr=rng.uniform(0.05, 0.8), td=rng.uniform(0.2, 3.0)))
    return out


# ----------------------------------------------------------------- spatiotemporal (CST) pRF
# Compressive-spatiotemporal extension (after Kim/Kupers/Grill-Spector): the single
# instantaneous tau*Dt^q term in v1_drive/core_drive is replaced by two parallel channels
# with their OWN memory -- a sustained channel (causal low-pass of the centre drive Dc,
# time constant taus) and a transient channel (causal low-pass of the frame-difference
# drive Dt, time constant taut) -- each compressed by the existing exponent q and mixed by
# a per-voxel weight ws. Isolates (taus, taut, ws) with every other V1 mechanism held at a
# plugin or generic value and attention switched off entirely, the same "one model at a
# time, generic elsewhere" convention as the attention-dynamics block above.
CST_DEFAULT = dict(taus=0.12, taut=0.08, ws=0.6)
CST_BOUNDS = dict(taus=(0.02, 0.6), taut=(0.02, 0.6), ws=(0.0, 1.0))


def causal_filter(x, tau, fps=FPS, length=3.0):
    """x: (nf,V) signal; tau: (V,) per-voxel causal single-exponential low-pass time
    constant (s), unit-area (DC gain 1). Same explicit-tap-loop style as attention_dynamic."""
    nk = max(2, int(length * fps))
    t = (np.arange(nk) / fps)[:, None]
    tau = np.maximum(np.asarray(tau, float), 1e-3)
    K = np.exp(-t / tau[None, :])
    K = (K / np.maximum(K.sum(0, keepdims=True), 1e-9)).astype(np.float32)
    out = np.zeros_like(x)
    for l in range(nk):
        if l == 0:
            out += x * K[0]
        else:
            out[l:] += x[:-l] * K[l]
    return out


def core_dcs(Ec, Es, Tc, mu=None, sig=None):
    """(nf,V,NB) features -> SF-weighted (Dc, Ds, Dt), each (nf,V). Params default to the
    single generic M.DEFAULT value or may be given as per-voxel (V,) arrays (e.g. a plugin
    from a prior joint fit), exactly like core_drive's mu/sig."""
    V = Ec.shape[1]
    d = DEFAULT
    mu = np.full(V, d['mu']) if mu is None else np.asarray(mu)
    sig = np.full(V, d['sig']) if sig is None else np.asarray(sig)
    w = sf_weights(mu, sig)
    Dc = np.einsum('nvb,vb->nv', Ec, w)
    Ds = np.einsum('nvb,vb->nv', Es, w)
    Dt = np.einsum('nvb,vb->nv', Tc, w)
    return Dc.astype(np.float32), Ds.astype(np.float32), Dt.astype(np.float32)


def cst_combine(Dc, Ds, Dt, c50, q, k, taus, taut, ws):
    """(Dc,Ds,Dt) (...,V) + per-voxel CST/CRF params (V,) -> attention-free drive (...,V).
    Two-channel compressive-spatiotemporal combination, replacing the single instantaneous
    tau*Dt^q term: a sustained channel (low-pass Dc, taus) and a transient channel
    (low-pass Dt, taut), each compressed by q and mixed by ws, normalized as before."""
    Sc = causal_filter(Dc, taus)
    Tt = causal_filter(Dt, taut)
    num = ws[None] * Sc ** q[None] + (1 - ws[None]) * Tt ** q[None]
    den = c50[None] ** q[None] + Dc ** q[None] + k[None] * Ds ** q[None]
    return (num / den).astype(np.float32)


def cst_drive(Ec, Es, Tc, mu=None, sig=None, c50=None, q=None, k=None, taus=None, taut=None, ws=None):
    """(nf,V,NB) features -> (nf,V) attention-free CST drive, ground-truth or plugin
    params as for core_drive/core_dcs."""
    V = Ec.shape[1]
    d, cd = DEFAULT, CST_DEFAULT
    c50 = np.full(V, d['c50']) if c50 is None else np.asarray(c50)
    q = np.full(V, d['q']) if q is None else np.asarray(q)
    k = np.full(V, d['k']) if k is None else np.asarray(k)
    taus = np.full(V, cd['taus']) if taus is None else np.asarray(taus)
    taut = np.full(V, cd['taut']) if taut is None else np.asarray(taut)
    ws = np.full(V, cd['ws']) if ws is None else np.asarray(ws)
    Dc, Ds, Dt = core_dcs(Ec, Es, Tc, mu, sig)
    return cst_combine(Dc, Ds, Dt, c50, q, k, taus, taut, ws)


class CSTFitter:
    """Per-voxel fit of (taus, taut, ws) against precomputed (Dc, Ds, Dt) SF-weighted
    energy streams (10 Hz); c50/q/k come from `fixed` (plugin or M.DEFAULT), attention is
    switched off. The channel filtering + mixing is recomputed from (taus, taut, ws) on
    every optimizer step, exactly as attention_dynamic is recomputed in AttnDynFitter."""

    def __init__(self, runs, free=('taus', 'taut', 'ws')):
        # runs: dict(Dc, Ds, Dt (nf,V), Y (TR,V), Q)
        self.runs, self.free = runs, list(free)
        self.edges = np.cumsum([0] + [r['Dc'].shape[0] for r in runs])

    def fit_voxel(self, v, fixed, starts):
        runs = self.runs
        Dc = np.concatenate([r['Dc'][:, v] for r in runs])
        Ds = np.concatenate([r['Ds'][:, v] for r in runs])
        Dt = np.concatenate([r['Dt'][:, v] for r in runs])
        y = np.concatenate([r['Y'][:, v] - r['Q'] @ (r['Q'].T @ r['Y'][:, v]) for r in runs])
        lo = np.array([CST_BOUNDS[k][0] for k in self.free])
        hi = np.array([CST_BOUNDS[k][1] for k in self.free])
        edges = self.edges
        h = hrf(0.5, 1.0)
        c50v, qv, kv = np.array([fixed['c50']]), np.array([fixed['q']]), np.array([fixed['k']])

        def pred(P):
            taus, taut = np.array([P['taus']]), np.array([P['taut']])
            ws = np.array([P['ws']])
            d = cst_combine(Dc[:, None], Ds[:, None], Dt[:, None], c50v, qv, kv, taus, taut, ws)[:, 0]
            out = []
            for i, r in enumerate(runs):
                seg = d[edges[i]:edges[i + 1]]
                seg = seg[:len(seg) // 5 * 5].reshape(-1, 5).mean(1)
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
            x0 = np.clip([s[k] for k in self.free], lo + 1e-6, hi - 1e-6)
            try:
                r = least_squares(res, x0, bounds=(lo, hi), max_nfev=50, x_scale='jac')
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        out = dict(fixed)
        if best is not None:
            out.update(zip(self.free, best.x))
        return out


def cst_starts(rng, n=4):
    base = dict(CST_DEFAULT)
    out = [dict(base)]
    for _ in range(n - 1):
        out.append(dict(taus=rng.uniform(0.03, 0.5), taut=rng.uniform(0.03, 0.5), ws=rng.uniform(0.1, 0.9)))
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
