"""Stimulus front end: bar retinotopy, contrast-energy (CE) images, and
gaze-contingent reconstruction of the retinal CE stream.

CE: multi-scale DoG band-pass of contrast c=(L-0.5)/0.5, squared, locally pooled,
square-rooted, averaged over 3 scales (peaks ~1.4, 0.7, 0.35 cpd). Computed on the
screen padded by 3 deg of mean-gray surround, then stored at 0.25 deg resolution.
Shifting a CE map by gaze is exact because filtering commutes with translation.
"""
import numpy as np
from .game import W, H, PX_PER_DEG, FPS

PAD = 24                      # px = 3 deg
PW, PH = W + 2 * PAD, H + 2 * PAD          # 240 x 192 px
CE_RES = 0.25                 # deg per CE cell
CE_W, CE_H = PW // 2, PH // 2               # 120 x 96 cells: x in [-15,15], y in [-12,12]
CANVAS = 256                  # canvas cells (0.25 deg): screen coords [-32, 32]
RET_HALF = 18.0               # retinal window half-width (deg)


def _kernels():
    fy = np.fft.fftfreq(PH)[:, None]
    fx = np.fft.rfftfreq(PW)[None, :]
    f2 = fx ** 2 + fy ** 2
    g = lambda s: np.exp(-2 * np.pi ** 2 * s ** 2 * f2)   # FT of unit-area Gaussian
    dogs, pools = [], []
    for sc in (1.0, 2.0, 4.0):
        dogs.append((g(sc) - g(1.6 * sc)).astype(np.complex64))
        pools.append(g(2.0 * sc).astype(np.complex64))
    return dogs, pools


_DOGS, _POOLS = _kernels()


def contrast_energy(frames):
    """frames: (n,H,W) luminance in [0,1] -> (n, CE_H, CE_W) float16."""
    n = frames.shape[0]
    c = np.zeros((n, PH, PW), np.float32)
    c[:, PAD:PAD + H, PAD:PAD + W] = (frames - 0.5) / 0.5
    C = np.fft.rfft2(c)
    ce = np.zeros((n, PH, PW), np.float32)
    for d, p in zip(_DOGS, _POOLS):
        b = np.fft.irfft2(C * d, s=(PH, PW))
        e = np.fft.irfft2(np.fft.rfft2(b * b) * p, s=(PH, PW))
        ce += np.sqrt(np.maximum(e, 0))
    ce /= len(_DOGS)
    ce = ce.reshape(n, CE_H, 2, CE_W, 2).mean((2, 4))
    return ce.astype(np.float16)


class CESink:
    """Collects CE for a run from frame chunks."""
    def __init__(self):
        self.chunks = []

    def __call__(self, frames):
        self.chunks.append(contrast_energy(frames))

    def result(self):
        return np.concatenate(self.chunks)


# ----------------------------------------------------------------- bars
BAR_DIRS = [0, 135, 270, 45, 180, 315, 90, 225]
TR = 1.5


def bar_schedule(n_tr=200, sweep_trs=18, blank=8):
    sched = [None] * blank
    for k, d in enumerate(BAR_DIRS):
        sched += [(d, i) for i in range(sweep_trs)]
        if k in (1, 3, 5):
            sched += [None] * blank
    sched += [None] * (n_tr - len(sched))
    return sched[:n_tr]


def simulate_bar_run(seed, duration=300.0, frame_sink=None, radius=9.0, width=2.25,
                     check=0.75, keep_frames=0):
    rng = np.random.default_rng(seed)
    nf = int(duration * FPS)
    fpt = int(TR * FPS)
    sched = bar_schedule(int(duration / TR))
    X = ((np.arange(W) + 0.5 - W / 2) / PX_PER_DEG)[None, :]
    Y = ((H / 2 - np.arange(H) - 0.5) / PX_PER_DEG)[:, None]
    R = np.hypot(X, Y)
    fix = R < 0.15
    # fixation jitter + rare microsaccades (true gaze; analysis assumes (0,0))
    gaze = np.zeros((nf, 2), np.float32)
    g = np.zeros(2)
    for f in range(nf):
        g = 0.9 * g + rng.normal(0, 0.06, 2)
        if rng.random() < 0.02:
            g = g + rng.normal(0, 0.3, 2)
        gaze[f] = np.clip(g, -0.8, 0.8)
    kept, buf = [], []
    span = radius + width / 2
    for f in range(nf):
        s = sched[f // fpt]
        img = np.full((H, W), 0.5, np.float32)
        if s is not None:
            th = np.deg2rad(s[0])
            pos = -span + 2 * span * (s[1] + 0.5) / 18
            u = X * np.cos(th) + Y * np.sin(th)
            v = -X * np.sin(th) + Y * np.cos(th)
            mask = (np.abs(u - pos) < width / 2) & (R < radius)
            par = ((np.floor(u / check) + np.floor(v / check)) % 2) * 2 - 1
            par = par * (1 if f % 2 == 0 else -1)
            img = np.where(mask, 0.5 + 0.5 * par, 0.5).astype(np.float32)
        img[fix] = 0.25
        if f < keep_frames:
            kept.append(img.copy())
        buf.append(img)
        if len(buf) == 50 or f == nf - 1:
            if frame_sink is not None:
                frame_sink(np.stack(buf))
            buf = []
    return dict(gaze=gaze, events=np.zeros((0, 3)), frames=np.array(kept))


# ----------------------------------------------------------------- retinal reconstruction
def _canvas(ce_frames):
    n = ce_frames.shape[0]
    cv = np.zeros((n, CANVAS, CANVAS), np.float32)
    r0, c0 = CANVAS // 2 - CE_H // 2, CANVAS // 2 - CE_W // 2
    cv[:, r0:r0 + CE_H, c0:c0 + CE_W] = ce_frames
    return cv


def retinal_stack(ce_frames, gaze, step=1):
    """Retinal CE (n, R, R) at 0.25 deg over +/-RET_HALF deg, given gaze (n,2) in
    screen deg. Retinal point r sees screen point s = r + g. step=2 -> 0.5 deg."""
    n = ce_frames.shape[0]
    R = int(2 * RET_HALF / CE_RES)                       # 144
    cv = _canvas(ce_frames)
    out = np.empty((n, R, R), np.float32)
    gx = np.round(gaze[:, 0] / CE_RES).astype(int)
    gy = np.round(gaze[:, 1] / CE_RES).astype(int)
    base = CANVAS // 2 - R // 2
    for i in range(n):
        r0 = base - gy[i]
        c0 = base + gx[i]
        out[i] = cv[i, r0:r0 + R, c0:c0 + R]
    if step == 2:
        out = out.reshape(n, R // 2, 2, R // 2, 2).mean((2, 4))
    return out


def retinal_coords(step=1):
    R = int(2 * RET_HALF / CE_RES) // step
    d = CE_RES * step
    x = -RET_HALF + d / 2 + d * np.arange(R)
    y = RET_HALF - d / 2 - d * np.arange(R)
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()


def fit_stimulus(ce_run, gaze_used, bin_frames=5, chunk=500):
    """Retinal CE at 0.5 deg, averaged into 0.5 s bins -> (n_bins, 72*72) float32."""
    outs = []
    for s in range(0, ce_run.shape[0], chunk):
        st = retinal_stack(ce_run[s:s + chunk].astype(np.float32), gaze_used[s:s + chunk], step=2)
        n = st.shape[0] // bin_frames
        outs.append(st[:n * bin_frames].reshape(n, bin_frames, -1).mean(1))
    return np.concatenate(outs)
