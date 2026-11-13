"""Dedicated ('one model at a time') experiments, simulated in the same voxels.

V1 experiments (rendered through the same front end), 300 s runs, fixation:
  psft      isotropic band-pass noise at the 8 band centres + blank (3 s on / 1.5 s off, 5 Hz refresh)
  crf       1/f noise at 7 RMS contrasts + blank
  size      1/f noise discs (r = 1,2,4,8 deg, full) and annuli (inner 2,4,8 deg) + blank
  temporal  1/f noise refreshed at 0 (static), 1, 2.5, 5 Hz + blank
  attn      continuous 5 Hz 1/f noise; attention blocks: neutral / attend-left / attend-right (15 s)
Non-visual designs (no rendering needed; the simulated populations read the design):
  numerosity  dot displays N = 1..7 + blank, 3 s on / 1.5 s off
  flow        12 s optic-flow blocks: expand, rotate-left, rotate-right, static
  reward      cue (p = .2/.5/.8) -> outcome 3 s later, ~8 s trials
"""
import numpy as np
from .game import W, H, PX_PER_DEG, FPS

RMS_FULL = 0.35
X = ((np.arange(W) + 0.5 - W / 2) / PX_PER_DEG)[None, :]
Y = ((H / 2 - np.arange(H) - 0.5) / PX_PER_DEG)[:, None]
R = np.hypot(X, Y)
FIX = R < 0.15
_fy = np.fft.fftfreq(H)[:, None] * PX_PER_DEG
_fx = np.fft.fftfreq(W)[None, :] * PX_PER_DEG
_FR = np.hypot(_fx, _fy)
_FR[0, 0] = 1e-6


def noise(rng, kind, f0=None, rms=RMS_FULL):
    n = rng.standard_normal((H, W))
    if kind == 'pink':
        filt = 1 / _FR
        filt[_FR > 3.5] = 0
    else:
        filt = np.exp(-(np.log2(_FR / f0)) ** 2 / (2 * 0.35 ** 2))
    filt[0, 0] = 0
    img = np.real(np.fft.ifft2(np.fft.fft2(n) * filt))
    img = img / img.std() * rms
    return np.clip(0.5 + 0.5 * img, 0, 1).astype(np.float32)


def fixation_gaze(rng, nf):
    g = np.zeros(2)
    out = np.zeros((nf, 2), np.float32)
    for f in range(nf):
        g = 0.9 * g + rng.normal(0, 0.06, 2)
        if rng.random() < 0.02:
            g = g + rng.normal(0, 0.3, 2)
        out[f] = np.clip(g, -0.8, 0.8)
    return out


def trial_schedule(rng, n_cond, duration, on=3.0, off=1.5):
    """Per-frame condition index (-1 = gap). Conditions cycle through random permutations."""
    nf = int(duration * FPS)
    cond = np.full(nf, -1, int)
    t, order = 0.0, []
    while t + on <= duration:
        if not order:
            order = list(rng.permutation(n_cond))
        c = order.pop()
        cond[int(t * FPS):int((t + on) * FPS)] = c
        t += on + off
    return cond


def run_v1_experiment(kind, seed, duration=300.0, frame_sink=None):
    rng = np.random.default_rng(seed)
    nf = int(duration * FPS)
    gaze = fixation_gaze(rng, nf)
    design = dict(kind=kind)
    attn = np.zeros(nf, np.float32)
    if kind == 'psft':
        from .bands import BAND_F
        cond = trial_schedule(rng, len(BAND_F) + 1, duration)
    elif kind == 'crf':
        levels = [0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0]
        cond = trial_schedule(rng, len(levels) + 1, duration)
    elif kind == 'size':
        apert = [R < 1, R < 2, R < 4, R < 8, np.ones_like(R, bool), R >= 2, R >= 4, R >= 8]
        cond = trial_schedule(rng, len(apert) + 1, duration)
    elif kind == 'temporal':
        rates = [0, 1.0, 2.5, 5.0]
        cond = trial_schedule(rng, len(rates) + 1, duration)
    elif kind == 'attn':
        cond = np.zeros(nf, int)
        blk = int(15 * FPS)
        seq = [0] + [1, 2] * 9 + [0]
        for b, s in enumerate(seq):
            attn[b * blk:(b + 1) * blk] = {0: 0, 1: -1, 2: 1}[s]
    else:
        raise ValueError(kind)
    design['cond'] = cond
    design['attn_side'] = attn
    buf, cur, last_c, last_refresh = [], None, None, -1e9
    for f in range(nf):
        c = cond[f]
        t = f / FPS
        img = np.full((H, W), 0.5, np.float32)
        if kind == 'attn':
            if cur is None or f % 2 == 0:
                cur = noise(rng, 'pink', rms=0.5 * RMS_FULL)
            img = cur.copy()
        elif c >= 0:
            if kind == 'psft' and c < len(BAND_F):
                if c != last_c or f % 2 == 0:
                    cur = noise(rng, 'band', BAND_F[c])
                img = cur.copy()
            elif kind == 'crf' and c < 7:
                if c != last_c or f % 2 == 0:
                    cur = noise(rng, 'pink', rms=levels[c] * RMS_FULL)
                img = cur.copy()
            elif kind == 'size' and c < 8:
                if c != last_c or f % 2 == 0:
                    cur = noise(rng, 'pink', rms=0.5 * RMS_FULL)
                img = np.where(apert[c], cur, 0.5).astype(np.float32)
            elif kind == 'temporal' and c < 4:
                rate = rates[c]
                if c != last_c or (rate > 0 and t - last_refresh >= 1 / rate - 1e-6):
                    cur = noise(rng, 'pink', rms=0.5 * RMS_FULL)
                    last_refresh = t
                img = cur.copy()
        last_c = c
        img[FIX] = 0.25
        buf.append(img)
        if len(buf) == 50 or f == nf - 1:
            if frame_sink is not None:
                frame_sink(np.stack(buf))
            buf = []
    return dict(gaze=gaze, design=design, events=np.zeros((0, 3)))


def numerosity_design(seed, duration=300.0):
    rng = np.random.default_rng(seed)
    cond = trial_schedule(rng, 8, duration)        # 0..6 -> N=1..7, 7 = blank
    return np.where((cond >= 0) & (cond < 7), cond + 1, 0).astype(np.int16)


def flow_design(seed, duration=300.0):
    rng = np.random.default_rng(seed)
    nf = int(duration * FPS)
    fwd, yaw = np.zeros(nf, np.float32), np.zeros(nf, np.float32)
    blk = int(12 * FPS)
    order = []
    for b in range(nf // blk):
        if not order:
            order = list(rng.permutation(4))
        c = order.pop()
        sl = slice(b * blk, (b + 1) * blk)
        if c == 0:
            fwd[sl] = 1.8
        elif c == 1:
            yaw[sl] = 1.0
        elif c == 2:
            yaw[sl] = -1.0
    return fwd, yaw


# ----------------------------------------------------------------- SEQ-SIM (CST)
# Mirrors Kim/Kupers' real-data paradigm that dissociates sustained from transient
# temporal summation: two brief full-field contrast probes (A, B; independent noise
# samples) are each shown for a fixed `probe` duration, either SIMULTANEOUSLY (one
# combined onset/offset -- A and B's presentations coincide exactly, so a pooling
# population sees a single transient) or SEQUENTIALLY (A, then a gap of `soa`, then B --
# two separate onsets/offsets) -- varying only WHEN the two probes occur, not how much
# contrast energy appears or where. Full-field (unlike a spatially localized probe pair)
# so every pRF, whatever its eccentricity or polar angle, is driven the same way; a
# population's sustained channel integrates both probes into roughly the same total
# response regardless of order, while its transient channel fires once per onset/offset
# it actually sees, so SEQ-SIM is the controlled, fair counterpart to gameplay's
# uncontrolled transient timing.
SEQSIM_PROBE = 0.35
SEQSIM_SOAS = (0.0, 0.15, 0.3, 0.6)
SEQSIM_OFF = 1.2


def seqsim_trials(rng, duration, probe=SEQSIM_PROBE, soas=SEQSIM_SOAS, off=SEQSIM_OFF):
    """List of (t0, mode, soa): mode 0 = SEQ, 1 = SIM. Conditions (mode x soa, 2*len(soas)
    of them) cycle through random permutations, as trial_schedule does for fixed-length
    blocks -- here each condition's own total trial length differs (SIM = probe; SEQ =
    2*probe + soa), so the schedule is built directly rather than reusing trial_schedule."""
    conds = [(m, s) for m in (0, 1) for s in soas]
    trial_len = [probe if m == 1 else 2 * probe + s for m, s in conds]
    t, order, trials = 0.0, [], []
    while True:
        if not order:
            order = list(rng.permutation(len(conds)))
        c = order.pop()
        tl = trial_len[c]
        if t + tl > duration - off:
            break
        m, s = conds[c]
        trials.append((t, m, s))
        t += tl + off
    return trials


def run_seqsim_experiment(seed, duration=300.0, frame_sink=None, probe=SEQSIM_PROBE,
                          soas=SEQSIM_SOAS, off=SEQSIM_OFF):
    rng = np.random.default_rng(seed)
    nf = int(duration * FPS)
    gaze = fixation_gaze(rng, nf)
    trials = seqsim_trials(rng, duration, probe, soas, off)
    on1, on2 = np.zeros(nf, bool), np.zeros(nf, bool)      # on1: first probe; on2: second
    cond = np.full(nf, -1, int)
    for t0, m, s in trials:
        f0 = int(round(t0 * FPS))
        f1 = int(round((t0 + probe) * FPS))
        ci = soas.index(s) + (len(soas) if m == 1 else 0)
        on1[f0:f1] = True
        if m == 1:                     # SIM: A and B's onsets coincide -- one transient
            on2[f0:f1] = True
            cond[f0:f1] = ci
        else:                          # SEQ: B follows A after the gap `s`
            fb0 = int(round((t0 + probe + s) * FPS))
            fb1 = int(round((t0 + 2 * probe + s) * FPS))
            on2[fb0:fb1] = True
            cond[f0:fb1] = ci
    design = dict(kind='seqsim', trials=np.array(trials, np.float32), probe=probe,
                 soas=np.array(soas), cond=cond)
    buf, cur, active, last_active = [], None, False, False
    for f in range(nf):
        active = on1[f] or on2[f]
        if active and (cur is None or not last_active or f % 2 == 0):
            cur = noise(rng, 'pink', rms=0.5 * RMS_FULL)
        last_active = active
        img = cur.copy() if active else np.full((H, W), 0.5, np.float32)
        img[FIX] = 0.25
        buf.append(img)
        if len(buf) == 50 or f == nf - 1:
            if frame_sink is not None:
                frame_sink(np.stack(buf))
            buf = []
    return dict(gaze=gaze, design=design, events=np.zeros((0, 3)))


def reward_design(seed, duration=300.0):
    """Returns shots-like array (t_outcome, p, outcome, 0, 0)."""
    rng = np.random.default_rng(seed)
    out, t = [], 2.0
    while t + 4 < duration:
        p = rng.choice([0.2, 0.5, 0.8])
        out.append((t + 3.0, p, float(rng.random() < p), 0.0, 0.0))
        t += rng.uniform(7, 9)
    return np.array(out)
