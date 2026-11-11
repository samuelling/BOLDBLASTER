# Noise model

All noise is added in percent signal change at TR = 1.5 s (200 TRs per 5-min run),
independently per voxel, in `boldblaster/sim.py::nuisance_bold` (rounds 1–4) with session
effects in `run_long.py` (round 4).

## What is added to every voxel

| Component | Applies to | Form | Parameters | Shared across voxels? | What the fitter does |
|---|---|---|---|---|---|
| Measurement noise | all runs | AR(1), ρ = 0.3 | SD per voxel ~ U(0.6, 1.2) % | no (independent) | nothing: least squares, no prewhitening |
| Slow drift | all runs | Legendre order 1–3 | each coefficient ~ N(0, 0.5) % per voxel per run | no | Legendre 0–3 regressors per run |
| Event-locked arousal | game runs (hits/explosions); reward-task wins | gamma response peaking at 8 s, peak-normalized, one per event | per-voxel weight ~ N(0.5, 0.2), min 0.05 | shape shared, scaled per voxel | regressor from logged events with a 7 s kernel (deliberately mismatched) + its derivative |
| Slow global fluctuation | all runs | random walk smoothed by a 20 s boxcar, SD 0.3 | same per-voxel weight as arousal | yes | not modeled (partly absorbed by drift terms) |
| HRF timing | all voxels | double-gamma with per-voxel time scaling | scale ~ N(1, 0.08), clipped 0.8–1.2 | no | round 1: canonical HRF; rounds 2–4: per-voxel HRF scale fitted |
| Session effects (round 4) | each 2 h session | response gain ~ N(1, 0.1); noise SD × U(0.85, 1.15) | — | per session | one gain per session fitted |
| Eye-tracker error | game runs | per-run offset N(0, 0.5°), linear drift N(0, 0.3°), gain error N(0, 4 %), sample noise 0.3°, blink dropouts (~1.2 % of frames, held) | — | — | not corrected |

## Signal against it

- V1 response amplitude: U(1.5, 3) % at the 95th percentile of the bar-driven response.
  Numerosity, self-motion and striatal populations: U(1, 2) %.
- Round 1 medians of the SD across time: visual signal 0.60 % (bars) and 0.14 % (gameplay),
  systemic 0.29 %, noise 0.91 %.
- ~0.9 % noise per TR is roughly tSNR ≈ 110 for a single voxel: moderate, plausible for
  7T at ~2 mm or a good 3T voxel. Many 3T 2 mm voxels are noisier (tSNR 40–80).

## Not simulated (where the testbed is optimistic)

- Spatially correlated noise (physiology, motion, draining veins); noise here is independent per voxel.
- Head motion, including controller-induced motion.
- Separate cardiac and respiratory components (RETROICOR-style corrections would remove some, not all).
- Region-specific or time-varying systemic responses; arousal is one shared shape.
- Noise that scales with voxel size or field strength.

## Sensitivity

Hours-to-recovery scales roughly with noise variance: doubling the noise SD roughly
quadruples the gameplay time needed for the same recovery. A sweep over tSNR (50/100/200)
with correlated noise and motion is the natural next simulation.
