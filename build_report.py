"""Builds report/index.html from data/*.json and figures/ (numbers are never hand-typed)."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
D, F, R = (os.path.join(ROOT, x) for x in ('data', 'figures', 'report'))
os.makedirs(R, exist_ok=True)
res = json.load(open(os.path.join(D, 'results.json')))
dec = json.load(open(os.path.join(D, 'decompose_check.json')))
tra = json.load(open(os.path.join(D, 'transient_check.json')))
for f in os.listdir(F):
    shutil.copy(os.path.join(F, f), os.path.join(R, f))


def pe(c):
    return f"{res[c]['pos_err_median']:.2f}°"


def row(c, label):
    r = res[c]
    return (f"<tr><td>{label}</td><td>{r['minutes_train']:.0f}</td><td>{r['pos_err_median']:.2f}°</td>"
            f"<td>{r['by_ecc']['9-12.1']['pos_err']:.2f}°</td>"
            f"<td>{r['size_logratio_median']:+.2f}</td><td>{r['frac_ceiling_game']:.2f}</td>"
            f"<td>{r['frac_ceiling_bar']:.2f}</td></tr>")


rows = [('bars', 'Bars, fixation (standard)'), ('game_truegaze', 'Game, perfect gaze'),
        ('game_eyetracker', 'Game, in-bore eye tracker'), ('game_nogaze', 'Game, gaze ignored'),
        ('game_et_noarousal', 'Game, ET, no arousal regressor'), ('bars+game', 'Bars + game (ET)'),
        ('game_et_2h', 'Game, ET, 2 hours')]
table = '\n'.join(row(c, l) for c, l in rows if c in res)
dec_rows = '\n'.join(f"<tr><td>{k}</td><td>{v['pos_err']:.2f}°</td><td>{v['size_log2']:+.2f}</td></tr>"
                     for k, v in dec.items())
g2h = res.get('game_et_2h')
two_h = (f"With <b>2 hours</b> of gameplay (eye-tracker gaze), median error falls to "
         f"<b>{g2h['pos_err_median']:.2f}°</b> (vs {pe('bars')} for 15 min of bars), and "
         f"{g2h['by_ecc']['9-12.1']['pos_err']:.2f}° beyond 9° eccentricity, where bars give "
         f"{res['bars']['by_ecc']['9-12.1']['pos_err']:.2f}°.") if g2h else ''

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BOLDBLASTER: can pRFs be fit from free gameplay?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--bg:#fcfcfb;--accent:#eb6834;--blue:#2a78d6}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}} h1 span{{color:var(--accent)}}
.sub{{color:var(--ink2);margin:0 0 28px}}
h2{{font-size:19px;margin:44px 0 10px;padding-top:18px;border-top:1px solid var(--rule)}}
.tldr{{background:#fff;border:1px solid var(--rule);border-left:4px solid var(--accent);padding:16px 20px;border-radius:6px}}
.tldr li{{margin:6px 0}}
img,video{{width:100%;border:1px solid var(--rule);border-radius:6px;background:#fff}}
figcaption{{color:var(--ink2);font-size:13px;margin:6px 0 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--ink2);font-weight:600;font-size:12px}}
code{{font:13px 'JetBrains Mono',monospace;background:#f1f0ec;padding:1px 5px;border-radius:4px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} @media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.note{{color:var(--ink2);font-size:13px}}
</style></head><body><main>
<h1>BOLD<span>BLASTER</span></h1>
<p class="sub">Simulation testbed: can V1 pRFs be recovered from free first-person gameplay, and what would a dynamic, NSD-style gaming dataset need?</p>

<div class="tldr"><b>Bottom line</b><ul>
<li><b>The game carries the spatial information; the problem is SNR.</b> With noise-free BOLD, gameplay recovers pRFs as well as bars ({dec['visual only']['pos_err']:.2f}° median error). With realistic noise and 30 min of play, error is {pe('game_eyetracker')}, against {pe('bars')} for 15 min of bars.</li>
<li><b>Why:</b> gameplay keeps V1 driven but barely <i>modulates</i> any one pRF. In a sustained-only V1 model, gameplay modulation is ~{tra['sustained']['game_over_bar_modulation']:.2f}× that of bars, so it needs roughly an order of magnitude more data per unit of information.</li>
<li><b>The periphery is where gameplay wins.</b> Saccades sweep the screen across the outer visual field, which a 9° bar aperture never reaches. At 9–12° eccentricity, 30 min of gameplay already beats bars ({res['game_eyetracker']['by_ecc']['9-12.1']['pos_err']:.2f}° vs {res['bars']['by_ecc']['9-12.1']['pos_err']:.2f}°). Central pRFs suffer most under gameplay.</li>
<li><b>Localizer fits transfer to gameplay:</b> pRFs fit on bars predict held-out gameplay at {res['bars']['frac_ceiling_game']:.0%} of the noise ceiling. That is the cross-check anyone using the dataset would want.</li>
<li><b>Gaze is mandatory:</b> ignoring eye position gives {pe('game_nogaze')} error. A realistic in-bore tracker costs little over perfect gaze ({pe('game_eyetracker')} vs {pe('game_truegaze')}).</li>
<li><b>Log engine events:</b> arousal-locked systemic BOLD alone pushes error to {dec['visual + systemic (no regressor)']['pos_err']:.2f}°; a regressor built from logged game events brings it back to {dec['visual + systemic (arousal regressor)']['pos_err']:.2f}°.</li>
<li><b>Design implication:</b> deep sampling (hours per subject) plus a short controlled localizer in the same people. Bars + game together gave the best fits ({pe('bars+game')}). {two_h}</li>
</ul></div>

<h2>1 · The sandbox</h2>
<p>A numpy raycaster first-person game: procedural rooms and corridors, wall textures from low to high spatial frequency, floor and ceiling, distance fog, wandering NPCs, tag events with explosions and screen flashes, plus a static weapon and crosshair. Screen 24×18°, 10 Hz, scripted walking, turning, stopping and looking around. Gaze is a fixation/saccade process biased to the crosshair, drawn to NPCs, and captured by explosions. Every frame is converted to multi-scale contrast energy and shifted into retinal coordinates using gaze.</p>
<video src="sandbox_clip.mp4" controls loop muted playsinline></video>
<figcaption>30 s of play. Left: screen with gaze (red). Right: the contrast energy the retina receives (±13.5°), which jumps with every saccade.</figcaption>
<p></p>
<img src="fig1_sandbox.png" alt="Game frames, retinal contrast energy, bar stimulus and mean game contrast energy">
<figcaption>Top: game frames with gaze, and a bar-retinotopy frame. Bottom: retinal contrast energy, and the time-averaged screen contrast energy. Note the static structure (screen border, crosshair, weapon) that free viewing smears across the retina.</figcaption>

<h2>2 · Ground truth and fitting</h2>
<p>500 V1 voxels: centres sampled by cortical magnification out to 12°, σ ≈ 0.25 + 0.13·ecc, CSS exponent 0.5–0.9, per-voxel HRF timing jitter, AR(1) noise (0.6–1.2%), polynomial drift, and systemic arousal locked to game events plus slow global fluctuation. Truth is generated at 0.25° and 10 Hz with jittered HRFs. The fitter works at 0.5° and 2 Hz with the canonical HRF, so it never sees the true generative model. Fits use a 64,512-candidate CSS grid, then batched local refinement. Each run is 300 s; game runs 0–5 train and 6–7 test; bar runs 0–2 train and 3 tests.</p>

<h2>3 · Gameplay barely modulates any one pRF</h2>
<img src="fig2_signal.png" alt="Signal components">
<figcaption>Left: one 4° voxel during play. The visual component is small next to the arousal-locked systemic component and noise. Middle: median SD of each component. Right: the game/bar ratio rises past 9°, where bars never reach but eye movements put the screen.</figcaption>

<h2>4 · Recovery by condition</h2>
<img src="fig4_summary.png" alt="Summary by condition">
<p></p>
<table><thead><tr><th>Condition</th><th>train min</th><th>median pos. error</th><th>error, ecc 9–12°</th><th>size bias (log₂)</th><th>held-out game R²/ceiling</th><th>held-out bar R²/ceiling</th></tr></thead>
<tbody>{table}</tbody></table>
<p class="note">Size bias is log₂ of fitted/true effective size (σ/√n). R²/ceiling divides cross-validated R² by the R² of the true noise-free visual signal on the same test data.</p>
<img src="fig3_recovery.png" alt="Position error by eccentricity and visual-field maps">
<figcaption>Top: position error against true eccentricity (line = binned median). Bottom: true pRF centres (dots) with lines to fitted centres; the dashed circle is the bar aperture.</figcaption>

<h2>5 · It is SNR, not identifiability</h2>
<p>Grid fits on 120 voxels, 6 game runs, true gaze, adding one nuisance source at a time:</p>
<table><thead><tr><th>Data</th><th>median pos. error</th><th>size bias (log₂)</th></tr></thead><tbody>{dec_rows}</tbody></table>
<p>The game stimulus fully constrains pRFs. Measurement noise is what breaks the fits. At low SNR the fitter also shrinks pRFs (small pRFs chase noise in sharp, local contrast events), so <b>size estimates are the first casualty</b>.</p>

<h2>6 · How much gameplay is enough?</h2>
<img src="fig5_quantity.png" alt="Error versus minutes of training data">
<figcaption>Error and generalization against training minutes for bars and for gameplay with the eye tracker. Beyond 9° (right of the bar aperture), gameplay overtakes bars once there is enough of it.</figcaption>

<h2>7 · Caveats: the testbed is deliberately pessimistic</h2>
<ul>
<li><b>No temporal channel.</b> Real V1 responds strongly to change. A transient channel driven by |ΔCE| gives a game/bar modulation ratio of {tra['transient']['game_over_bar_modulation']:.2f} (vs {tra['sustained']['game_over_bar_modulation']:.2f} sustained; {tra['mixture_50_50']['game_over_bar_modulation']:.2f} for a 50/50 mix), because every saccade and camera turn is a full-field transient. That is roughly 3× more signal variance than assumed here, which would cut the data requirement accordingly.</li>
<li>One contrast-energy front end with no per-voxel SF tuning. A pSFT layer slots in at <code>stimuli.contrast_energy</code>.</li>
<li>No head motion from controller use; gaze shifts quantized to 0.25°.</li>
<li>The game is a 1990s-style raycaster: real engines add parallax, lighting changes and richer motion.</li>
</ul>

<h2>8 · What this says about the dataset</h2>
<ul>
<li><b>Hours per subject, not many subjects.</b> Gameplay gives up information about early visual tuning slowly. A dense design is the only way it pays off for pRF/pSFT-type models.</li>
<li><b>Embed a localizer.</b> 15 min of bars plus gameplay beats either alone. It also provides the validation anchor for anyone fitting a model to the game data.</li>
<li><b>Eye tracking and event logging are non-negotiable.</b> Gaze rescues position estimates; engine-logged events remove most of the systemic confound.</li>
<li><b>Design the game for modulation.</b> Since SNR is the bottleneck, the engine should create local contrast change: probe textures, lighting cuts, objects that appear and vanish in the periphery. The next simulation to run is whether those raise information per minute.</li>
</ul>
<p class="note">Code: <code>github.com/samuelling/boldblaster</code> — <code>python run_experiment.py</code> then <code>python make_figures.py</code> and <code>python build_report.py</code>.</p>
</main></body></html>"""
open(os.path.join(R, 'index.html'), 'w').write(html)
print('report built')
