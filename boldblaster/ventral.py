"""Ventral/temporal 'category-selective, large pRF' voxel population (round 6, part B).

Ground truth: response = baseline + categories . weights, where 'categories' are game-engine
content maps (NPC presence; 4 wall-texture identities; floor; ceiling) pooled through a large,
foveal-biased Gaussian pRF -- the fwRF recipe (one spatial field, several feature channels),
with weights sparse/skewed toward one preferred category per voxel (category selectivity).
No rectification: kept linear so recovery is a plain GLM, like the self-motion/MST population.

Content maps reuse the exact 0.5 deg / gaze-shift machinery already built for V1 band energy
(bands.retinal, bands.PAD/BLOCK/CANVAS) -- they are zero-padded and block-averaged to the same
(MH, MW) grid as band_energy() so bands.retinal() can be called unchanged.
"""
import numpy as np
from . import bands
from .game import W, H, FPS

CH_NAMES = ['npc', 'wall1', 'wall2', 'wall3', 'wall4', 'floor', 'ceiling']
NCH = len(CH_NAMES)


def content_features(cat_frames):
    """(n,NCH,H,W) raw 0/1 maps -> (n,NCH,MH,MW), same padding/blocking as bands.band_energy."""
    n = cat_frames.shape[0]
    c = np.zeros((n, NCH, bands.PH, bands.PW), np.float32)
    c[:, :, bands.PAD:bands.PAD + H, bands.PAD:bands.PAD + W] = cat_frames
    return c.reshape(n, NCH, bands.MH, bands.BLOCK, bands.MW, bands.BLOCK).mean((3, 5))


class ContentSink:
    def __init__(self):
        self.chunks = []

    def __call__(self, cat_chunk):
        self.chunks.append(content_features(cat_chunk).astype(np.float16))

    def result(self):
        return np.concatenate(self.chunks)


def pool_content(maps, gaze, Gc, chunk=200):
    """maps (n,NCH,MH,MW) content features, gaze (n,2) deg -> (n,V,NCH) pooled at 10 Hz."""
    n = maps.shape[0]
    V = Gc.shape[0]
    out = np.empty((n, V, NCH), np.float32)
    for s in range(0, n, chunk):
        r = bands.retinal(np.asarray(maps[s:s + chunk], np.float32), gaze[s:s + chunk])  # (m,NCH,RET*RET)
        m = r.shape[0]
        out[s:s + m] = np.einsum('nkp,vp->nvk', r, Gc)
    return out


def sample_ventral(rng, n=200, ecc_max=10.0):
    """Large, foveal-biased pRFs: smaller max eccentricity, much larger sigma-vs-eccentricity
    slope and a bigger baseline sigma than V1 (sample_v1 in models.py), consistent with known
    ventral-temporal pRF scaling (Kay, Weiner & Grill-Spector 2015): big receptive fields that
    still prefer the fovea."""
    u = rng.uniform(0, 1, n) ** 1.6                 # density falls off with eccentricity, foveal bias
    ecc = u * ecc_max
    pa = rng.uniform(0, 2 * np.pi, n)
    sigma = (1.5 + 0.35 * ecc) * np.exp(rng.normal(0, 0.15, n))
    return dict(x=ecc * np.cos(pa), y=ecc * np.sin(pa), ecc=ecc, sigma=sigma)


def plant_weights(rng, n):
    """Each voxel prefers one of NCH categories (weight ~U(0.6,1.3)); small nonneg weights on
    the rest (sparse but not pure one-hot, like a real category-selective fwRF voxel)."""
    pref = rng.integers(0, NCH, n)
    W_ = rng.uniform(0.05, 0.25, (n, NCH))
    W_[np.arange(n), pref] = rng.uniform(0.6, 1.3, n)
    b = rng.uniform(0.1, 0.4, n)
    return dict(pref=pref, weights=W_.astype(np.float32), baseline=b.astype(np.float32))
