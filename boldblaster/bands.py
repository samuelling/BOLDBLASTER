"""Multi-band front end (v2): 8 isotropic log-Gabor SF bands, half-octave spaced
(0.25-2.83 cpd; the sandbox screen is 8 px/deg so Nyquist is 4 cpd), local energy
pooled at a scale-dependent width, stored at 0.5 deg on the screen padded by 4 deg.

Pooling into voxels: per frame, maps are shifted into retinal coordinates by gaze
(0.5 deg steps) and pooled with centre Gaussians (pRF), surround Gaussians
(3 x sigma, the normalization pool), and centre Gaussians applied to |dE/dt|
(transient channel, computed in retinal coordinates so saccades count)."""
import numpy as np
import scipy.fft as sfft
from .game import W, H, PX_PER_DEG

PAD = 32
PW, PH = W + 2 * PAD, H + 2 * PAD            # 256 x 208 px
BLOCK = 4                                    # px per 0.5 deg cell
MW, MH = PW // BLOCK, PH // BLOCK            # 64 x 52 cells
BAND_F = 0.25 * 2 ** (np.arange(8) / 2)      # cycles/deg
NB = len(BAND_F)
CANVAS = 128                                 # 0.5 deg cells, screen coords +/-32 deg
RET = 72                                     # retinal window +/-18 deg at 0.5 deg
E_SCALE = 1.0                                # set by calibrate_scale()


def _kernels(bw_oct=0.35):
    fy = np.fft.fftfreq(PH)[:, None]
    fx = np.fft.rfftfreq(PW)[None, :]
    fpx = np.hypot(fx, fy)                    # cycles/px
    fdeg = fpx * PX_PER_DEG
    fdeg[0, 0] = 1e-6
    filt, pool = [], []
    for f in BAND_F:
        filt.append(np.exp(-(np.log2(fdeg / f)) ** 2 / (2 * bw_oct ** 2)).astype(np.complex64))
        s = PX_PER_DEG * 0.5 / f              # pooling sigma (px)
        pool.append(np.exp(-2 * np.pi ** 2 * s ** 2 * fpx ** 2).astype(np.complex64))
    return filt, pool


_FILT, _POOL = _kernels()


def band_energy(frames):
    """(n,H,W) luminance -> (n, NB, MH, MW) float32 band energy (contrast units)."""
    n = frames.shape[0]
    c = np.zeros((n, PH, PW), np.float32)
    c[:, PAD:PAD + H, PAD:PAD + W] = (frames - 0.5) / 0.5
    C = sfft.rfft2(c)                         # float32 / complex64 throughout
    out = np.empty((n, NB, MH, MW), np.float32)
    for b in range(NB):
        x = sfft.irfft2(C * _FILT[b], s=(PH, PW))
        e = sfft.irfft2(sfft.rfft2(x * x) * _POOL[b], s=(PH, PW))
        e = np.sqrt(np.maximum(e, 0))
        out[:, b] = e.reshape(n, MH, BLOCK, MW, BLOCK).mean((2, 4))
    return out * E_SCALE


class BandSink:
    def __init__(self):
        self.chunks = []

    def __call__(self, frames):
        self.chunks.append(band_energy(frames).astype(np.float16))

    def result(self):
        return np.concatenate(self.chunks)


def retinal(maps, gaze):
    """maps (n,K,MH,MW), gaze (n,2) deg -> (n,K,RET*RET) retinal maps at 0.5 deg."""
    n, K = maps.shape[:2]
    cv = np.zeros((n, K, CANVAS, CANVAS), np.float32)
    r0, c0 = CANVAS // 2 - MH // 2, CANVAS // 2 - MW // 2
    cv[:, :, r0:r0 + MH, c0:c0 + MW] = maps
    out = np.empty((n, K, RET, RET), np.float32)
    gx = np.round(gaze[:, 0] / 0.5).astype(int)
    gy = np.round(gaze[:, 1] / 0.5).astype(int)
    base = CANVAS // 2 - RET // 2
    for i in range(n):
        out[i] = cv[i, :, base - gy[i]:base - gy[i] + RET, base + gx[i]:base + gx[i] + RET]
    return out.reshape(n, K, -1)


def pool(maps, gaze, Gc, Gs, chunk=200):
    """Returns Ec, Es, Tc each (n, V, NB) float32 at the frame rate."""
    n = maps.shape[0]
    V = Gc.shape[0]
    Ec = np.empty((n, V, NB), np.float32)
    Es = np.empty_like(Ec)
    Tc = np.empty_like(Ec)
    prev = None
    for s in range(0, n, chunk):
        r = retinal(np.asarray(maps[s:s + chunk], np.float32), gaze[s:s + chunk])
        m = r.shape[0]
        d = np.abs(np.diff(np.concatenate([r[:1] if prev is None else prev, r]), axis=0))
        prev = r[-1:]
        flat = r.reshape(m * NB, -1)
        Ec[s:s + m] = (flat @ Gc.T).reshape(m, NB, V).transpose(0, 2, 1)
        Es[s:s + m] = (flat @ Gs.T).reshape(m, NB, V).transpose(0, 2, 1)
        Tc[s:s + m] = (d.reshape(m * NB, -1) @ Gc.T).reshape(m, NB, V).transpose(0, 2, 1)
    return Ec, Es, Tc
