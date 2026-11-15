# BOLDBLASTER

> **Summary and dashboard, in one place:** `docs/dashboard/` (served at https://samuelling.github.io/boldblaster/dashboard/ once GitHub Pages is on; `docs/index.html` now just points there). It opens on an Overview of all seven experiments, each a card with its headline result, and one tab per experiment with what we did, a picture of the stimulus, the result, its key chart and a folded "More detail"; a Methods tab holds the simulation, noise and voxel-level views. Rebuild with `python dashboard/standalone/build.py`. Source: [`dashboard/`](dashboard/) (page + scripts over the CSVs; stimulus pictures from `make_stimulus_figs.py`; `test_harness.js` runs it headless with jsdom). Noise assumptions: [NOISE.md](NOISE.md). Tidy CSVs of every result: [`dashboard_data/`](dashboard_data/) (regenerate with `python export_dashboard_data.py`).

> **Latest (round 9):** navigation codes need free movement in every direction, not a particular game, and cognitive parameters that behaviour recovers well are often weak in BOLD; see Round 9 and [MECHANICS.md](MECHANICS.md).

> **Headline (round 8):** the game matters. At a matched hour with eye-tracker gaze, all three real games beat the raycaster on contrast response (effective c50 0.43 on the raycaster, 0.68 Betamax, 0.51 Brick Ninja, 0.54 Animal Sanctuary) and on attention gain (0.10; 0.42, 0.16, 0.37). Betamax, a realistic shooter, is best overall.

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

## Round 7: spatiotemporal (CST) pRF — can gameplay separate sustained from transient?

Following Kim/Kupers/Grill-Spector's compressive spatiotemporal pRF, the V1 drive's old
instantaneous transient term is replaced by two memory channels: a sustained low-pass of the
centre drive (time constant τ_s) and a transient low-pass of the frame-difference drive
(τ_t), mixed by a per-voxel weight w_s, each still passing through the existing compressive
nonlinearity (`boldblaster/models.py`: `causal_filter`, `core_dcs`, `cst_combine`,
`cst_drive`, `CSTFitter`). As with round 6's attention isolation, mu/sig/c50/q/k come from
the existing joint fit (or generic defaults for SEQ-SIM), not ground truth, and attention is
switched off — this test is about the new mechanism alone.

A dedicated SEQ-SIM design (`boldblaster/controlled.py`: `run_seqsim_experiment`) stands in
for Kupers' own sequential-vs-simultaneous retinotopic-bar paradigm: two full-field noise
flashes, 0.35 s each, shown either together or offset by a gap (0, 0.15, 0.3, 0.6 s). The
first version of this design used two small, fixed-location disc probes instead — a mistake
worth recording: with pRFs scattered across 0–12° eccentricity and all polar angles, two
small apertures only drive the handful of voxels whose pRF happens to overlap them, starving
the rest of signal and producing spuriously near-zero recoverability unrelated to the actual
timing question. Switching to full-field probes fixed that, and is the version reported here.

Even full-field, the dedicated design does not win. Across 20/40/60 minutes, the
channel-mixing weight w_s is the only one of the three parameters gameplay (oracle gaze) gets
real traction on at all, and SEQ-SIM does no better — if anything slightly worse — at every
duration; both time constants (τ_s, τ_t) stay indistinguishable from noise under every
design. A ground-truth diagnostic (no fitting) explains why: filtering the sustained and
transient drive channels at the population-median time constants and correlating them over
time shows SEQ-SIM's channels are *no less collinear* than gameplay's (median voxel
correlation comparable under both). The sequential-vs-simultaneous manipulation, in this
full-field, mostly-blank form, doesn't buy the channel separation Kupers' retinotopic bar
version does — its low duty cycle (brief probes against a mostly-blank screen) plausibly
costs more total driven signal than its clean timing control buys back, while gameplay's
continuous, always-on content drives both channels throughout the run. This is a genuine,
surprising result, reported as found rather than adjusted to fit the original hypothesis
that a controlled design should win on temporal identifiability; it's also only a test of
this specific full-field stand-in, not of Kupers' actual retinotopic-bar version, which could
behave differently.

`python run_round7.py pop && python run_round7.py cst_data && python run_round7.py cst_fit
&& python run_round7.py seqsim_data && python run_round7.py seqsim_fit && python
run_round7.py diagnostic`. Results JSON: `results_cst.json`, `results_cst_seqsim.json`,
`results_cst_diagnostic.json`.

## Round 8: does the game matter? Three real games against the raycaster

Every earlier round used BOLDBLASTER's own numpy raycaster. Round 8 swaps in three real games, each
played by a bot for twelve 5-minute runs on the same 24 x 18 deg screen at 192 x 144 px, and refits
every model unchanged (`run_round8.py`, one game at a time via `GAME=betamax|brickninja|sanctuary`):

- [Betamax](https://github.com/samuelling/betamax)'s Mayflower Beach battle, played **forward**
  (time runs in order, kills stay down, no rewind and no taking bodies; if you fall you carry on as
  the nearest man of your side).
- [Brick Ninja](https://github.com/samuelling/Brick-Ninja), the ten side-scrolling worlds.
- [Animal Sanctuary](https://github.com/samuelling/Animal-Sanctuary), the five wild zones, walking
  up to creatures and playing the befriending timing game.

*Betamax rendering* (`tools/betamax/`): rebuilt with one hook (`betamax-harness-hook.patch`,
exposing `window.BWfwd(actor, tapeTime)`) and run headed under Xvfb in Chrome on Mesa's llvmpipe
(3.5x faster than SwiftShader here), stepped with the game's own `BWstep` at 2 x 50 ms ticks per
10 Hz frame, VCR filter off, off-screen soldier instances culled. The bot (`bot.js`) plays a Cape
Cod defender: it acquires the nearest visible Texan with a 150-400 ms reaction delay, turns with
first-order pursuit plus tremor, fires bursts on target, strafes, crouches and scans. Frames come
off `readPixels`, about 1 s/frame/worker, so a 5-min run takes about 50 min.

*2D rendering* (`tools/games2d/`): both games are single-file canvas games wrapped in an IIFE, so a
one-line patch exposes `window.__eval` inside it. A virtual clock replaces `performance.now` and
`requestAnimationFrame`, the bot (`bots/brickninja.js`, `bots/sanctuary.js`) runs in the game's
scope, and each frame is letterboxed onto a mid-grey 4:3 screen. A run takes seconds. Brick
Ninja's bot runs, jumps, throws aimed stars and swings its sword; Animal Sanctuary's walks to the
nearest creature and presses during the catch bar with human-like timing error (SD 0.13).

*Fitting*: each game's logs become exactly the files `run_v2.py` writes for the raycaster. Gaze is
the same `GazeModel`, fed the game's on-screen targets (enemies or creatures); items for
numerosity are enemies within 60 m, studs or creatures; self-motion is speed and turn rate in
Betamax, and horizontal camera scroll as turning with zero forward motion in the 2D games, zeroed
across camera cuts; the RPE prediction is each shot's hit probability from a per-game logistic fit
(bursts, stars, catch presses). CST uses each game's own joint V1 fit as plug-in tuning. Matched
comparisons use the first six (30 min) or twelve (60 min) raycaster runs.

*Result, matched 60 min, eye-tracker gaze* (Spearman r; raycaster / Betamax / Brick Ninja / Animal
Sanctuary):

| Parameter | Raycaster | Betamax | Brick Ninja | Animal Sanctuary |
|---|---|---|---|---|
| Effective c50 | 0.43 | 0.68 | 0.51 | 0.54 |
| SF peak | 0.37 | 0.43 | 0.36 | 0.27 |
| SF bandwidth | -0.01 | 0.22 | 0.16 | 0.17 |
| CRF exponent | 0.15 | 0.43 | 0.36 | 0.27 |
| Surround k | 0.22 | 0.24 | 0.23 | 0.25 |
| Transient weight | 0.61 | 0.62 | 0.62 | 0.49 |
| Attention gain | 0.10 | 0.42 | 0.16 | 0.37 |
| Striatal RPE beta | 0.66 | 0.96 | 0.96 | 0.63 |
| Self-motion rotation | 0.88 | 0.87 | 0.45 | 0.19 |
| CST sustained weight | 0.30 | 0.53 | 0.37 | 0.36 |

Betamax is best overall: dawn glare, haze, dark silhouettes and muzzle flash spread V1's contrast
drive over the widest range, its spectrum falls off like a natural scene, and its firefight gives
the most salient events. Animal Sanctuary's slow, flat pixel art costs it SF peak and transient
weight, though its constant creature encounters help attention. RPE follows the number of
outcomes: about 40 shots a minute in the action games, a dozen catch presses in Animal Sanctuary,
which ties the raycaster. The 2D games cannot map self-motion (no forward flow, and scrolling
barely varies in direction). With oracle gaze every game improves; the 30-min comparison shows the
same ordering. Full numbers: `dashboard_data/round8_v1.csv`, `round8_other.csv`, `round8_stats.csv`.

*Caveats:* scripted bots, not people, with model-generated gaze; each game's restarts replay the
same levels or battle, so an hour of one is less varied than an hour of fresh raycaster levels; a
one-off scale bug (band energy computed before `V2.calibrate()`) briefly made Betamax look 10x
weaker and was fixed before any result here.

Renders: `bash tools/betamax/queue.sh 0 12 & bash tools/betamax/queue.sh 1 12` (see the header of
`tools/betamax/run.mjs`), and `tools/games2d/README.md` for the 2D games. Then
`GAME=<game> python run_round8.py all 12` for each game, `python export_round8.py` and
`python make_stimulus_figs.py`. Results JSON: `results/round8.json`, `round8_stats.json` and
`round8_<game>_phit.json`.

## Round 9: navigation and cognitive models, driven by the games' own dynamics

The Palace builds a test and dresses it as a game. Round 9 goes the other way: it takes the four
games as they are and asks which models their dynamics can fit, with every parameter given a
candidate brain area (`dashboard_data/areas.csv`). Medial temporal voxels get 1.4-1.6x cortical
noise for susceptibility loss (`run_round9.py`, `export_round9.py`).

*Navigation, voxel by voxel* (150 voxels per area, from logged position, heading and gaze;
raycaster poses are regenerated from its seeds, the others come from the harness logs):

- entorhinal hexadirectional codes for movement direction (Doeller 2010) and eye-movement
  direction (Nau 2018)
- retrosplenial facing-direction tuning
- hippocampal path vs Euclidean distance to the current goal (Howard 2014)
- boundary-distance tuning
- a hippocampal event-boundary response

Grid orientation is recovered wherever movement is free: phase recovery is .58 on the
raycaster, .52 on Betamax and .46 on Animal Sanctuary, against chance (.12) on Brick Ninja,
which only runs left or right. Facing direction is recovered at .98 with free heading.
- The eye-movement grid code is weak everywhere (at most .35).
- Path and Euclidean distance only come apart in the raycaster maze (path weight recovered at
  .74, though even there the two correlate .97).
- Event boundaries come through on Betamax (.58) but not in Animal Sanctuary (.09): its boundaries
  are befriendings, which the arousal nuisance model removes.

*Cognitive models from The Palace and Mind Arcade, player by player* (40 simulated players per
game, planted parameters, behaviour with 4% lapses and motor jitter, region-mean BOLD with noise
shared across voxels):

- drift diffusion for each engage decision (pre-SMA / IPS accumulator)
- Rescorla-Wagner value learning (ventral striatum RPE)
- place familiarity with encoding and forgetting (hippocampal novelty)

Behaviour recovers drift and boundary at .84-.99 in every game. BOLD recovers them at most .46,
because the accumulator signal needs spaced decisions over a wide range of difficulty, and
Betamax's targets barely vary in difficulty. Learning rate from striatal BOLD follows outcomes
per minute: .68 on Brick Ninja, .42 on Betamax, .33-.36 on the raycaster and Animal Sanctuary.
Place familiarity is recovered best with many revisits (Brick Ninja .94/.95). Path integration
and map updating could not be tested: none of the games has homing or layout changes.

Game-design side notes, including candidate puzzle elements for Betamax drawn from Prey,
Dishonored, Doom, Dark Souls, Antichamber and The Palace: [MECHANICS.md](MECHANICS.md).
`python run_round9.py all && python export_round9.py`. Results JSON: `results/round9_nav.json`,
`results/round9_cog.json`.

## Round 10: puzzle elements in Betamax (The Diggings)

Round 9 found that Betamax's open beach starves the navigation and memory models, so Betamax got
tunnels: a braided maze under the parking lot and behind the dune crest, reached by hatches, with
ammunition runs, cave-ins, sector blackouts and friend-or-foe figures (branch `diggings` of
samuelling/betamax; the bot and harness are in `tools/betamax/`). The bot played the same hour
(12 x 5 min) with occasional tunnel missions, and every model was refitted (`run_round10.py`,
`export_round10.py`, with `GAME=betamaxt python run_round8.py` for the visual models).

Three models that no earlier game could test were added:

- path integration on blackout homing walks (The Palace's lantern model: heading noise, distance
  gain, leak), from where the player thinks home is and from entorhinal/hippocampal uncertainty
- map updating on junction choices after cave-ins (The Palace's East Wing model: update rate,
  route-choice temperature), from choices and from hippocampal surprise
- drift diffusion on the friend-or-foe calls

Results at 60 min, Betamax vs Betamax with tunnels:

- Path vs Euclidean goal distance: untestable to .94 (the two distances correlate .77 in the warren).
- Boundary distance: untestable to .73. Place familiarity encoding .45 to .52, forgetting .66 to .70.
- Path integration heading noise: behaviour .89, BOLD .43 (agreement .30); distance gain .71 and
  leak .86 from behaviour.
- But the bot spent 55% of the hour below ground: firefight decisions fell from 38 to 14 a
  minute, V1 SF peak from .43 to .12, effective c50 from .68 to .57, and striatal learning rate
  from .42 to .19.
- Map updating and friend-or-foe calls were too rare (13 cave-ins met, 64 calls an hour):
  behaviour recovers route-choice temperature (.97) and drift (.85), BOLD recovers neither.

How the estimates miss (`diag_round10.py`, planted vs fitted per voxel or player): behaviour
estimates sit on the diagonal with small, explainable biases (lapses and motor jitter make drift
read low, the boundary high and choice consistency low). BOLD estimates are rarely biased on
average but stick to the edges of the fitted range (70% of striatal learning rates with tunnels).
Eye-tracker error makes V1 SF peak read half an octave low; most of the tunnels' SF loss is gaze
error, not image content. Map updating has a ceiling (no estimate above .65) because each cave-in
is met about once.

Next: shorter, denser tunnel trips (MECHANICS.md). Results JSON: `results/round10_*.json`.

Results JSON for every round is in `results/`; the summary and dashboard are in `docs/dashboard/`
(enable GitHub Pages from `main` / `/docs` to serve it).
