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

## Round 2: many models from one game

`run_v2.py` plants parameters for four populations in one simulated subject and asks
whether gameplay recovers them as well as dedicated experiments:

- **V1 (300 voxels):** pRF + SF tuning (8 half-octave bands), contrast response with
  surround normalization, transient channel (|ΔE| in retinal coordinates), attention gain
  around salient events. Modules: `bands.py` (front end), `models.py` (truth + fitter).
- **Numerosity (150), self-motion (150), striatal RPE (100)**, driven by engine logs.
- Game v2 (`game.simulate_run_v2`): coin clusters (1–8) as a numerosity lever,
  probabilistic shots with logged p(hit), salient-onset and self-motion logs.
- Dedicated designs (`controlled.py`): band-pass noise, contrast series, discs/annuli,
  refresh-rate series, attend-left/right blocks, dot numerosity, optic-flow blocks, cued
  reward task (2 × 5 min each).

Spearman r between planted and fitted parameters (game uses eye-tracker gaze and
bar-fitted pRFs; oracle uses true gaze and pRFs):

| Parameter | Dedicated 10 min | V1 battery joint 50 min | Game 1 h | Game 2 h | Game 2 h oracle |
|---|---|---|---|---|---|
| pSFT peak | 0.72 | 0.75 | 0.38 | 0.47 | 0.65 |
| pSFT bandwidth | 0.31 | 0.39 | 0.00 | 0.11 | 0.21 |
| pCRF effective c50 | 0.49 | 0.50 | 0.43 | 0.58 | 0.72 |
| pCRF exponent | 0.20 | 0.17 | 0.15 | 0.26 | 0.42 |
| Surround k | 0.10 | 0.07 | 0.21 | 0.25 | 0.51 |
| Transient τ | 0.04 | 0.48 | 0.61 | 0.65 | 0.70 |
| Attention gain | 0.50 | 0.57 | 0.10 | 0.22 | 0.24 |
| Numerosity preference | 0.91 | — | 0.99 | 0.99 | — |
| Numerosity width | 0.58 | — | 0.89 | 0.93 | — |
| Rotation preference | 0.92 | — | 0.88 | 0.94 | — |
| Expansion share | 0.95 | — | 0.92 | 0.93 | — |
| Striatal RPE β | 0.19 | — | 0.66 | 0.72 | — |

Pipeline lessons: apply model nonlinearities at the frame rate before binning; fit a
per-voxel HRF timescale; interacting V1 parameters (k, τ, q) need joint estimation, and a
contrast experiment measures effective c50 = c50/(1+k)^(1/q).

```bash
python run_v2.py data      # ~25 min on 2 cores (needs round-1 data/ for pRFs)
python run_v2.py fit       # ~15 min
python make_figures_v2.py && python build_report.py
```

## Round 3: design levers

`simulate_run_v2(..., levers=True)` adds covert-attention cues (central arrow → peripheral
location, gaze held at the crosshair for 2 s) and SF/contrast-decorrelated walls
(contrast-equalized band-pass textures, random per-segment contrast, weaker fog).
With 2 h of play: pSFT peak 0.47 → 0.64 (dedicated 0.72; 0.79 with oracle gaze/pRFs),
attention gain 0.22 → 0.33 (dedicated 0.50), CRF exponent 0.26 → 0.37.
`python run_v2.py data3 && python run_v2.py fit levers_1h levers_2h levers_2h_oracle`.

## Round 4 (in progress): 20 h across ten themed levels

`run_long.py`: ten 2 h sessions (Warehouse, Night ops, Open arena, Cathedral, Sprint,
Treasure hunt, Fog bank, Mixed day, Boss rush, Dusk maze) with session-specific gain and
noise; fits at 2/5/10/20 h and a 4 h varied-vs-single-level comparison.
