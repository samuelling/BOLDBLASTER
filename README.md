# BOLDBLASTER

A simulation testbed for one question: **can model-based fMRI (here, V1 pRF
mapping) be fit from free gameplay instead of a controlled localizer, and what
does it take?**

It plays a sandboxed first-person game, simulates a V1 voxel population with
known pRFs responding to what lands on the retina, generates BOLD, and then fits
pRFs back out under realistic conditions. Recovered parameters are scored against
ground truth and cross-validated on held-out game runs and a held-out bar run.

## Pipeline

| Stage | What | Module |
|---|---|---|
| Game | Numpy raycaster: procedural rooms + corridors, 4 wall textures spanning low→high SF, textured floor/ceiling, distance fog, wandering NPCs, tag events with explosions + screen flashes, static weapon + crosshair. 24×18° screen, 10 Hz, scripted walk/turn/stop/look-around trajectory. | `boldblaster/game.py` |
| Gaze | Fixation/saccade sampler: crosshair-biased, NPC-seeking, captured by explosions. | `game.GazeModel` |
| Bars | Standard 8-direction bar sweep (2.25° wide, 9° aperture), 5 Hz checkerboard, fixation with jitter. | `stimuli.simulate_bar_run` |
| Front end | Multi-scale DoG contrast energy; gaze-contingent shift into retinal coordinates (exact, because filtering commutes with translation). | `boldblaster/stimuli.py` |
| Truth | 500 V1 voxels (cortical-magnification sampling, σ ≈ 0.25 + 0.13·ecc, CSS exponent 0.5–0.9), per-voxel HRF timing jitter, event-locked systemic arousal + slow global drift, AR(1) noise, polynomial drift. Truth runs at 0.25°/10 Hz; the fitter at 0.5°/2 Hz with the canonical HRF, so the fitter is never handed the true generative model. | `boldblaster/sim.py` |
| Eye tracker | Per-run calibration offset (σ 0.5°), drift, 4% gain error, 0.3° sample noise, blink dropouts. | `run_experiment.eyetracker` |
| Fit | CSS pRF: 64,512-candidate grid → 3 rounds of batched local refinement; nuisance (Legendre 0–3, optional arousal regressors built from logged game events) projected out. | `boldblaster/fit.py` |

## Conditions

`bars`, `game_truegaze`, `game_eyetracker`, `game_nogaze`, `game_et_noarousal`,
`bars+game`, plus a data-quantity curve (1/2/4/6 game runs, 1/2/3 bar runs).
Each run is 300 s (200 TRs at 1.5 s). Game: train runs 0–5, test 6–7. Bars:
train 0–2, test 3.

## Run

```bash
pip install -r requirements.txt
python run_experiment.py            # ~45 min on 2 cores; stages cache to data/
python run_experiment.py bars       # or any subset of conditions
python -c "import run_experiment as R; R.extend_game_runs(26)"   # +18 game runs for the 2 h point
python run_experiment.py game_et_2h
python decompose_check.py           # noise vs systemic-arousal split
python transient_check.py           # sensitivity: sustained vs transient V1 channel
python make_figures.py              # figures/ + sandbox_clip.mp4
python build_report.py              # report/index.html
```

## Known simplifications (things to add next)

- Spatial-only V1 model: no temporal channels, so saccade/motion transients are
  missing. This probably *underestimates* gameplay-driven modulation.
- Single contrast-energy front end; no SF tuning per voxel (a pSFT layer would
  slot in at `stimuli.contrast_energy`).
- Gaze-contingent shift is integer-quantized at 0.25°.
- Head motion from controller use is not simulated.

## Results (first pass)

| Condition | train min | median pRF position error | error at 9–12° |
|---|---|---|---|
| Bars, fixation | 15 | 0.42° | 1.87° |
| Game, perfect gaze | 30 | 2.16° | 1.14° |
| Game, in-bore eye tracker | 30 | 2.71° | 1.10° |
| Game, gaze ignored | 30 | 4.08° | 3.92° |
| Bars + game (ET) | 45 | 0.37° | 0.51° |
| Game, eye tracker, 2 h | 120 | 1.01° | 0.55° |

Noise-free game data recovers pRFs to 0.19°, so the limit is SNR, not
identifiability: gameplay modulates each V1 pRF ~0.2× as much as bars (sustained
model). Figures in `figures/`.
