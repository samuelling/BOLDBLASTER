"""Ground-truth V1 voxel population and BOLD generation.

Neural drive (10 Hz): d_v(t) = (sum_xy G_v(x,y) * CE_retina(x,y,t)) ** n_v   (CSS pRF)
BOLD: amp_v * (d_v conv HRF_v)  + systemic arousal (game only) + global slow
arousal + AR(1) noise + low-order drift. Units: % signal change.

Truth uses a 0.25 deg retinal grid, 10 Hz, and per-voxel HRF time-scaling; the
fitter uses 0.5 deg, 2 Hz, and the canonical HRF (avoids the inverse crime).
"""
import numpy as np
from scipy.stats import gamma as gamma_dist
from .stimuli import retinal_stack, retinal_coords
from .game import FPS

TR = 1.5


def sample_voxels(rng, n=500, ecc_max=12.0):
    # uniform in cortical distance (M(E) ~ 1/(E+0.75)) -> foveal oversampling, like V1
    u = rng.uniform(np.log(0.5 + 0.75), np.log(ecc_max + 0.75), n)
    ecc = np.exp(u) - 0.75
    pa = rng.uniform(0, 2 * np.pi, n)
    sigma = (0.25 + 0.13 * ecc) * np.exp(rng.normal(0, 0.15, n))
    return dict(
        x=ecc * np.cos(pa), y=ecc * np.sin(pa), ecc=ecc, pa=pa, sigma=sigma,
        n=rng.uniform(0.5, 0.9, n),                 # CSS exponent
        hrf_scale=np.clip(rng.normal(1.0, 0.08, n), 0.8, 1.2),
        amp=rng.uniform(1.5, 3.0, n),               # % at reference drive
        noise_sd=rng.uniform(0.6, 1.2, n),
        sys_amp=np.clip(rng.normal(0.5, 0.2, n), 0.05, None),
    )


def gaussians(x0, y0, sigma, step=1):
    X, Y = retinal_coords(step)
    G = np.exp(-((X[None] - np.asarray(x0)[:, None]) ** 2 + (Y[None] - np.asarray(y0)[:, None]) ** 2)
               / (2 * np.asarray(sigma)[:, None] ** 2))
    return (G / G.sum(1, keepdims=True)).astype(np.float32)


def hrf(dt, scale=1.0, length=32.0):
    t = np.arange(0, length, dt) / scale
    h = gamma_dist.pdf(t, 6) - gamma_dist.pdf(t, 16) / 6
    return h / h.sum()


def arousal_kernel(dt, peak=8.0, length=40.0):
    t = np.arange(0, length, dt)
    k = gamma_dist.pdf(t, peak + 1, scale=1.0)
    return k / k.max()


def _conv(x, h, axis=-1):
    n = x.shape[axis] + len(h)
    nfft = 1 << int(np.ceil(np.log2(n)))
    X = np.fft.rfft(x, nfft, axis=axis)
    Hh = np.fft.rfft(h, nfft)
    shape = [1] * x.ndim
    shape[axis] = -1
    y = np.fft.irfft(X * Hh.reshape(shape), nfft, axis=axis)
    return np.take(y, np.arange(x.shape[axis]), axis=axis)


def linear_drive(ce_run, gaze, G, chunk=500):
    """(frames, voxels) linear pRF drive at 10 Hz."""
    out = []
    for s in range(0, ce_run.shape[0], chunk):
        st = retinal_stack(ce_run[s:s + chunk].astype(np.float32), gaze[s:s + chunk])
        out.append(st.reshape(st.shape[0], -1) @ G.T)
    return np.concatenate(out)


def to_tr(x10hz):
    k = int(TR * FPS)
    n = x10hz.shape[0] // k
    return x10hz[:n * k].reshape(n, k, -1).mean(1)


def visual_bold(lin, vox, ref):
    """lin: (frames, vox) linear drive -> visual BOLD (TR, vox), %."""
    d = np.maximum(lin, 0) ** vox['n'][None, :]
    d = d / ref[None, :]
    out = np.empty_like(d)
    for v in range(d.shape[1]):
        out[:, v] = _conv(d[:, v], hrf(1 / FPS, vox['hrf_scale'][v]))
    return to_tr(out) * vox['amp'][None, :]


def nuisance_bold(rng, vox, n_tr, events, duration):
    """Systemic (event-locked arousal + slow global) and noise components, (TR, vox)."""
    nv = len(vox['x'])
    dt = 1 / FPS
    nf = int(duration * FPS)
    imp = np.zeros(nf)
    for t, *_ in events:
        imp[int(t * FPS)] += 1
    ev = _conv(imp, arousal_kernel(dt))
    slow = np.cumsum(rng.normal(0, 1, nf))
    slow = _conv(slow - slow.mean(), np.ones(int(20 * FPS)) / (20 * FPS))
    slow = 0.3 * slow / (slow.std() + 1e-9)
    sysm = to_tr((ev + slow)[:, None])[:, 0]
    systemic = sysm[:, None] * vox['sys_amp'][None, :]
    # AR(1) noise
    e = rng.normal(0, 1, (n_tr, nv))
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, n_tr):
        ar[i] = 0.3 * ar[i - 1] + np.sqrt(1 - 0.09) * e[i]
    noise = ar * vox['noise_sd'][None, :]
    tt = np.linspace(-1, 1, n_tr)
    leg = np.stack([np.polynomial.legendre.Legendre.basis(k)(tt) for k in (1, 2, 3)], 1)
    drift = leg @ rng.normal(0, 0.5, (3, nv))
    return systemic, noise + drift
