# BOLDBLASTER

> Interactive dashboard: `docs/dashboard/` (served at https://samuelling.github.io/boldblaster/dashboard/ once GitHub Pages is on; rebuild with `python dashboard/standalone/build.py`). Source: [`dashboard/`](dashboard/) (page + scripts over the CSVs; `test_harness.js` runs it headless with jsdom). Noise assumptions: [NOISE.md](NOISE.md). Tidy CSVs of every result: [`dashboard_data/`](dashboard_data/) (regenerate with `python export_dashboard_data.py`).

> **Headline (round 5):** against dedicated experiments of realistic length (30–60 min each), gameplay is not a per-parameter shortcut. At a matched 5 h budget the full dedicated battery recovers 3 of 7 V1 measures better, gameplay 1 (surround suppression), and 3 tie. Gameplay's case is breadth: one dataset serves every model at once. Earlier rounds used a 10-min dedicated baseline, which flattered gameplay.

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

| Parameter | Short dedicated (10 min) | V1 battery joint 50 min | Game 1 h | Game 2 h | Game 2 h oracle |
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

## Round 4: 20 h across ten themed levels

`run_long.py`: ten 2 h sessions (Warehouse, Night ops, Open arena, Cathedral, Sprint,
Treasure hunt, Fog bank, Mixed day, Boss rush, Dusk maze) with session-specific gain and
noise (one gain per session in the fit); 120 V1 voxels. Spearman r (dedicated scored on
the same voxels):

| Parameter | Short dedicated (10 min) | 2 h | 5 h | 10 h | 20 h |
|---|---|---|---|---|---|
| pSFT peak | 0.69 | 0.37 | 0.55 | 0.78 | 0.77 |
| pSFT bandwidth | 0.26 | 0.14 | 0.07 | 0.18 | 0.32 |
| pCRF effective c50 | 0.45 | 0.35 | 0.55 | 0.61 | 0.61 |
| pCRF exponent | 0.17 | 0.23 | 0.36 | 0.44 | 0.49 |
| Surround k | 0.18 | 0.48 | 0.57 | 0.60 | 0.60 |
| Transient τ | 0.14 | 0.65 | 0.70 | 0.82 | 0.84 |
| Attention gain | 0.58 | 0.41 | 0.54 | 0.63 | 0.70 |

At equal time (4 h), varied levels vs one level type: all differences within ±0.08, all
bootstrap CIs span zero. `python run_long.py data && python run_long.py fit && python long_summary.py`.

## Round 5: a fair dedicated baseline

Rounds 2–4 compared gameplay against dedicated experiments of only 10 min (2 × 5-min runs),
which flatters gameplay: real pRF/pSFT/pCRF mapping takes 30–60+ min. `run_fair.py` extends
each dedicated V1 experiment to 12 runs (60 min) and fits it at 10/30/60 min, alone (other
parameters at population defaults, as in round 2) and as a joint battery of all five
experiments (the 5 h battery spans three sessions with round-4 session gain/noise jitter).
Everything is scored on the round-4 voxels (first 120); head-to-heads use a paired
bootstrap over voxels (`export_fair.py`).

| Parameter | Ded. 10 min | Ded. 30 min | Ded. 60 min | Battery 50 min | Battery 2.5 h | Battery 5 h | Game+levers 1 h | Play 5 h | Play 20 h |
|---|---|---|---|---|---|---|---|---|---|
| pSFT peak | 0.70 | 0.81 | 0.78 | 0.81 | 0.80 | 0.88 | 0.51 | 0.55 | 0.77 |
| pSFT bandwidth | 0.26 | 0.30 | 0.25 | 0.45 | 0.52 | 0.50 | 0.10 | 0.07 | 0.32 |
| pCRF effective c50 | 0.45 | 0.45 | 0.54 | 0.48 | 0.52 | 0.57 | 0.28 | 0.55 | 0.61 |
| pCRF exponent | 0.17 | 0.33 | 0.32 | 0.18 | 0.24 | 0.40 | 0.32 | 0.36 | 0.49 |
| Surround k | 0.19 | 0.17 | 0.16 | 0.14 | 0.17 | 0.19 | 0.16 | 0.57 | 0.60 |
| Transient τ | 0.15 | 0.21 | 0.30 | 0.42 | 0.60 | 0.59 | 0.48 | 0.70 | 0.84 |
| Attention gain | 0.58 | 0.69 | 0.80 | 0.60 | 0.70 | 0.81 | 0.16 | 0.54 | 0.70 |

**Matched 1 h** (one dedicated experiment vs 1 h of the game with levers):

| Parameter | Dedicated | Gameplay | Δ r [95% CI] | Verdict |
|---|---|---|---|---|
| pSFT peak | 0.78 | 0.51 | -0.27 [-0.49, -0.05] | Dedicated |
| pSFT bandwidth | 0.25 | 0.10 | -0.15 [-0.39, +0.09] | Tie |
| pCRF effective c50 | 0.54 | 0.28 | -0.26 [-0.45, -0.08] | Dedicated |
| pCRF exponent | 0.32 | 0.32 | -0.00 [-0.23, +0.22] | Tie |
| Surround suppression k | 0.16 | 0.16 | +0.01 [-0.22, +0.22] | Tie |
| Transient weight | 0.30 | 0.48 | +0.19 [-0.06, +0.41] | Tie |
| Attention gain | 0.80 | 0.16 | -0.64 [-0.82, -0.44] | Dedicated |

**Matched 5 h** (full dedicated battery vs 5 h of varied-level play):

| Parameter | Dedicated | Gameplay | Δ r [95% CI] | Verdict |
|---|---|---|---|---|
| pSFT peak | 0.88 | 0.55 | -0.33 [-0.50, -0.19] | Dedicated |
| pSFT bandwidth | 0.50 | 0.07 | -0.43 [-0.63, -0.24] | Dedicated |
| pCRF effective c50 | 0.57 | 0.55 | -0.02 [-0.14, +0.10] | Tie |
| pCRF exponent | 0.40 | 0.36 | -0.03 [-0.25, +0.17] | Tie |
| Surround suppression k | 0.19 | 0.57 | +0.38 [+0.20, +0.56] | Gameplay |
| Transient weight | 0.59 | 0.70 | +0.11 [-0.03, +0.25] | Tie |
| Attention gain | 0.81 | 0.54 | -0.27 [-0.44, -0.10] | Dedicated |

Hour for hour, dedicated experiments win on SF peak, effective c50 and attention, and no V1
measure is clearly better from an hour of play. At a matched 5 h budget the battery wins
3 (SF peak, SF bandwidth, attention), gameplay wins 1 (surround suppression) and 3 tie
(effective c50, CRF exponent, transient weight, which leans toward gameplay). Longer scans
help the noise-limited parameters; one-at-a-time fits of transient weight, surround and SF
bandwidth stay poor at 60 min because the other parameters are fixed, and only the joint
battery rescues the transient weight. Non-visual (no CIs): RPE weight is better from 1 h of
play (0.66) than 1 h of the reward task (0.50); dedicated optic flow beats play on self-motion
(0.99 vs 0.88–0.92); numerosity ties. The case for gameplay is breadth, plus the quantities
games sample richly, rather than per-parameter efficiency.
`python run_fair.py data && python run_fair.py fit && python run_fair.py other && python export_fair.py`.

## Round 6: attention dynamics, a ventral/temporal gaze-tolerance test, and one game lever

Three follow-ups (`run_round6.py`), reusing round-5's voxels, cached game logs and band
energy wherever possible (no re-rendering needed for part A).

**A. Can the *time course* of attention be recovered, not just its gain?** Every V1 voxel
gets a rise/decay kernel (τ_rise ~ U(0.1,0.5) s, τ_decay ~ U(0.4,2.5) s) applied to
event-locked (NPC-appear/kill/covert-cue) attention instead of the old fixed box; a new
`AttnDynFitter` recovers (a, τ_rise, τ_decay) per voxel, other parameters held at a prior
joint-fit estimate (not ground truth — the same "isolate one mechanism" convention as the
dedicated fits). Gain is weakly but increasingly recoverable from gameplay (oracle: 0.13 →
0.17 → 0.21 spearman at 20/40/60 min); the *shape* is not — τ_decay stays ~0.08–0.15 and
τ_rise is indistinguishable from noise (−0.08 to +0.03) at every duration, oracle or
realistic. The reason is visible in the ground truth itself: the attention term is only
~4.5% of the core visual drive's SD (median voxel), a small multiplicative ripple on a much
larger signal plus comparable measurement noise — more naturalistic event timing can't fix
a signal that's intrinsically this small relative to noise and to TR/HRF smoothing at the
sub-2-s scale these kernels live on.

The dedicated attn block design is worse off on principle, not just power: its ground truth
(`attention_blocks`) has *no temporal shape at all* within a 15 s block, so nothing about it
can inform a decay constant. Convolving its block-onsets with the same kernel and
correlating against the true box confirms this analytically — correlation climbs
monotonically across the entire biologically-plausible range (τ_decay 0.4→2.5 s: r
0.24→0.46) and only peaks once τ_decay approaches the block length itself (r 0.84 at 10–15
s), with a broad plateau either side (r 0.80–0.84 for 8–20 s). A fit to this design would be
pulled toward a 10–15 s decay regardless of the true value, and even that peak isn't sharply
resolved. Gameplay's irregular timing is the only one of the two designs that could ever see
transient dynamics — it just doesn't see them well *in this model*, because the effect size
is small.

**B. Do large, foveal-biased receptive fields buy gaze tolerance?** A new category-selective
"ventral" population (200 voxels, `boldblaster/ventral.py`) pools game-engine content maps
(NPC presence, 4 wall-texture identities, floor, ceiling — a toy stand-in for object/scene
category, via new `content_sink` support in `simulate_run_v2`) through large (σ ≈ 1.5–4+ deg),
foveally-biased pRFs instead of V1's small ones, with each voxel preferring one category.
Recovering the 7-way weight vector by per-voxel GLM, at a matched 40 min, the oracle-vs-
eye-tracker-gaze gap is tiny (preferred-weight spearman 0.33 → 0.31; full-vector correlation
0.543 → 0.535), versus V1's much larger gap at the *same* 40 min (pSFT peak 0.38 → 0.26,
computed fresh for this comparison since no prior round used exactly 40 min) — gaze
measurement error costs V1 six times more spearman than it costs the large-RF population.
Pairwise regressor correlations stay low (|r| ≤ 0.19) except the mechanically-forced
floor/ceiling anti-correlation (−0.55); NPC-selectivity in particular shows no inflation from
the gaze-content confound (people fixate NPCs) at this pooling scale. Absolute recovery is
still modest at 40 min (r ≈ 0.3–0.5) — category selectivity from gameplay is data-hungry —
but the gaze-tolerance prediction holds.

**C. One game-measure-improvement lever, tested rather than just proposed.** Of the six
ideas for narrowing the gap in round 5 (embedding probes in mechanics, adaptive design,
crosshair-anchored stimuli, stronger attention cues, more response variance, hierarchical
fitting), the cheapest to test against existing infrastructure was strengthening the covert
cue: tightening the inter-cue interval from 6–10 s to 3–5 s. It didn't help — attention-gain
spearman on 1 h of that gameplay (0.20) is no better than the original schedule (0.23),
plausibly because cues this frequent are hard to space efficiently against an ~8 s
hemodynamic response (the same event-spacing/collinearity trade-off that limits any
closely-packed fMRI design) — a reminder that "more cues" isn't free, and that the remaining
five levers (especially embedding probes directly in mechanics, and adaptive per-session
design) are likely to matter more than cue frequency alone.

`python run_round6.py pop && python run_round6.py attn_data && python run_round6.py attn_fit
&& python run_round6.py attn_identify && python run_round6.py vent && python run_round6.py
lever`. Results JSON: `results_attn.json`, `results_attn_identify.json`, `results_vent.json`,
`results_lever.json`, `results_v1_match40.json` (the matched-duration V1 comparison for B).

Results JSON for every round is in `results/`; the built report is in `docs/`
(enable GitHub Pages from `main` / `/docs` to serve it).
