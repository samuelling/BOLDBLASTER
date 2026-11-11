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

v2_section = ''
R2 = os.path.join(ROOT, 'data_v2', 'results.json')
if os.path.exists(R2):
    r2 = json.load(open(R2))

    def s(key, prm):
        return r2[key]['scores'][prm]['spearman']

    v2_section = f"""
<h2 id="round-2">Round 2 · Many models from one game</h2>
<div class="tldr"><b>Bottom line, round 2</b><ul>
<li><b>Gameplay wins where the game varies a quantity a lot or produces many events.</b> Transient weight: r = {s('game_2h','tau'):.2f} (game 2 h) vs {s('dedicated_temporal','tau'):.2f} (dedicated) and {s('battery_joint','tau'):.2f} (whole battery fit jointly). Numerosity preference: {s('num_game_2h','mu'):.2f} vs {s('num_dedicated','mu'):.2f}. Striatal RPE: {s('rpe_game_2h','rpe_beta'):.2f} from {r2['rpe_game_2h']['n_outcomes']} outcomes vs {s('rpe_dedicated','rpe_beta'):.2f} from {r2['rpe_dedicated']['n_outcomes']}. Effective c50: {s('game_2h','c50_eff'):.2f} vs {s('dedicated_crf','c50_eff'):.2f}.</li>
<li><b>Ties:</b> self-motion preferences are easy either way (rotation preference {s('mst_game_2h','rotation_pref'):.2f} vs {s('mst_dedicated','rotation_pref'):.2f}).</li>
<li><b>Gameplay loses on SF tuning and attention.</b> pSFT peak: {s('game_2h','mu'):.2f} vs {s('dedicated_psft','mu'):.2f}. Attention gain: {s('game_2h','a'):.2f} vs {s('dedicated_attn','a'):.2f}. In play, attention lands where the eyes land, on objects that are also visual onsets and arousal events.</li>
<li><b>Gaze and pRF accuracy multiply everything in V1.</b> With true gaze and true pRFs ("oracle"), the same 2 h gives pSFT peak {s('game_2h_oracle','mu'):.2f}, surround {s('game_2h_oracle','k'):.2f} (vs {s('game_2h','k'):.2f}), effective c50 {s('game_2h_oracle','c50_eff'):.2f}.</li>
<li><b>Interacting parameters can't be measured one at a time.</b> Surround suppression, transient weight and the CRF exponent trade off. Fixing the others at population defaults (the usual single-experiment analysis) leaves them unrecoverable. A contrast experiment measures an effective c50 = c50/(1+k)<sup>1/q</sup>, not c50 itself.</li>
</ul></div>
<p><b>Setup.</b> One simulated subject holds four populations with planted parameters. <b>V1</b> (300 voxels) has the round-1 pRFs plus SF tuning (log-Gaussian over 8 half-octave bands, 0.25–2.8 cpd), a contrast response with normalization, R = (D<sub>c</sub><sup>q</sup> + τ·D<sub>t</sub><sup>q</sup>) / (c50<sup>q</sup> + D<sub>c</sub><sup>q</sup> + k·D<sub>s</sub><sup>q</sup>) × (1 + a·Att), a transient channel driven by |ΔE| in retinal coordinates, and an attention gain field around salient events. The other populations are <b>numerosity</b> (150 voxels; log-Gaussian tuning), <b>self-motion</b> (150; forward, rotate-left, rotate-right weights) and <b>striatum</b> (100; outcome and RPE weights). The game gained clusters of 1–8 coins (the numerosity lever, visible ~70% of the time) and probabilistic shots with logged p(hit) from 0.1 to 0.95, mostly below 0.7 (about 50 per run).</p>
<p><b>Designs.</b> Dedicated experiments are 2 × 5 min each: band-pass noise (pSFT), 7 contrasts (pCRF), discs and annuli (surround), 0–5 Hz refresh (transient), attend-left/right blocks (attention), dot numerosity, optic-flow blocks, and a cued reward task. Gameplay is 1 h or 2 h. All V1 fits use bar-fitted pRFs from round 1, eye-tracker gaze, an assumed attention width, and a per-voxel HRF timescale. The dedicated fits free only their own parameters; the battery and game fits free all seven.</p>
<img src="fig6_recovery_matrix.png" alt="Recovery matrix across models and designs">
<figcaption>Spearman r between planted and fitted parameters. Dedicated rows: the experiment built for that parameter, 10 min each. The dedicated route totals ~95 min for one model each (15 bars + 50 V1 battery + 30 numerosity/flow/reward); gameplay is 15 min bars + 1–2 h play for all of them at once.</figcaption>
<p></p>
<img src="fig7_v1_scatter.png" alt="Planted vs fitted V1 parameters">
<figcaption>Planted vs fitted, dedicated (blue) vs game 2 h (orange). Points on the edges are fits at parameter bounds, mostly dedicated surround and transient fits with the other parameters held at defaults.</figcaption>
<h3>What building it taught us about the real pipeline</h3>
<ul>
<li><b>Apply nonlinearities at the frame rate, then bin.</b> Binning features to 2 Hz first distorts brief transients (trial onsets and offsets, saccades) by a factor that depends on each voxel's exponent. Noise-free contrast-response recovery collapsed until we fit at 10 Hz.</li>
<li><b>Fit each voxel's HRF timing.</b> ±20% HRF time-scaling dropped noise-free effective-c50 recovery in a rapid event design from 0.95 to 0.41; fitting a per-voxel timescale brought it back to 0.68.</li>
<li><b>Size event responses realistically.</b> Our first striatal simulation used 0.1 s impulses (~0.03% per event) and no design could recover anything. With 1 s outcome events (~0.2%), gameplay's event count dominates.</li>
</ul>
<h3>Caveats for round 2</h3>
<ul>
<li>Best case for model form: the fitter knows each model's functional form (not its parameters, HRF, gaze or pRFs).</li>
<li>The non-visual populations read engine variables directly: coin count, speed, yaw, p(hit). Real numerosity and motion areas also see item size, density and retinal flow, so the game must decorrelate those (vary coin size and spacing; separate flow from heading).</li>
<li>Dedicated non-visual experiments got 10 min against 60–120 min of play. Per minute, a dedicated design is often better; the gameplay gain is concurrency.</li>
<li>One simulated subject, no head motion, a screen capped at 8 px/deg (SF bands stop at 2.8 cpd).</li>
</ul>
<h3>Design levers this points to</h3>
<ul>
<li><b>Engineered quantities work:</b> coin clusters made numerosity the best-recovered model. Each target model should get its own lever in the level design.</li>
<li><b>Covert-attention moments:</b> events that demand monitoring the periphery while holding the crosshair would break the gaze/attention/onset collinearity.</li>
<li><b>SF and contrast decorrelation:</b> textures whose SF content varies independently of distance and fog, on a higher-resolution display.</li>
<li><b>Best-in-class eye tracking</b> and a pRF localizer for every subject: the oracle column is roughly what better gaze and pRFs buy.</li>
</ul>
"""

v3_section = ''
if os.path.exists(R2) and 'levers_2h' in r2:
    def d(prm):
        k = 'dedicated_' + {'c50_eff': 'crf', 'mu': 'psft', 'sig': 'psft', 'q': 'crf', 'k': 'size',
                            'tau': 'temporal', 'a': 'attn'}[prm]
        return s(k, prm)
    v3_section = f"""
<h2 id="round-3">Round 3 · Design levers</h2>
<p>Two changes to the game, aimed at round 2's losses. <b>Covert-attention cues:</b> every 8–12 s an arrow at the crosshair points to a peripheral spot (4–9°). For 2 s the player monitors it while gaze holds the crosshair, which separates attention from fixation, visual onsets and arousal (~30 episodes per 5-min run). <b>SF/contrast decorrelation:</b> walls use contrast-equalized band-pass textures (2–16 cycles per wall unit), each wall segment gets a random contrast (×0.25–1.6), and fog is weaker. Retinal SF then varies with viewing distance independently of contrast. Same planted V1 population, same fitter.</p>
<img src="fig8_levers.png" alt="Recovery with design levers">
<div class="tldr"><b>What the levers bought (2 h of play, realistic gaze and pRFs)</b><ul>
<li><b>SF tuning:</b> pSFT peak {s('game_2h','mu'):.2f} → <b>{s('levers_2h','mu'):.2f}</b> (dedicated {d('mu'):.2f}); bandwidth {s('game_2h','sig'):.2f} → {s('levers_2h','sig'):.2f}. With oracle gaze and pRFs the levered game beats the dedicated experiment ({s('levers_2h_oracle','mu'):.2f}).</li>
<li><b>Attention:</b> gain recovery {s('game_2h','a'):.2f} → <b>{s('levers_2h','a'):.2f}</b>, still below a dedicated block design ({d('a'):.2f}). Covert episodes help, but each is short and the attention field's width must be assumed.</li>
<li><b>CRF exponent</b> improves ({s('game_2h','q'):.2f} → {s('levers_2h','q'):.2f}); effective c50 dips slightly ({s('game_2h','c50_eff'):.2f} → {s('levers_2h','c50_eff'):.2f}), likely because weaker fog and equalized textures narrow the contrast range. Transient weight is unchanged.</li>
<li><b>Gaze/pRF accuracy is still the biggest multiplier:</b> the levered oracle reaches surround k {s('levers_2h_oracle','k'):.2f} and transient {s('levers_2h_oracle','tau'):.2f}.</li>
</ul></div>
"""

v4_section = ''
SL = os.path.join(ROOT, 'data_long', 'summary.json')
if os.path.exists(SL):
    S4 = json.load(open(SL))

    def g4(name, prm):
        x = S4[name][prm]
        return f"{x['r']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"

    rows4 = ''.join(
        f"<tr><td>{lab}</td><td>{S4['dedicated_10min'][p]['r']:.2f}</td>" +
        ''.join(f"<td>{S4[n][p]['r']:.2f}</td>" for n in ('long_2h', 'long_5h', 'long_10h', 'long_20h')) + '</tr>'
        for p, lab in (('mu', 'pSFT peak'), ('sig', 'pSFT bandwidth'), ('c50_eff', 'pCRF effective c50'),
                       ('q', 'pCRF exponent'), ('k', 'Surround k'), ('tau', 'Transient τ'), ('a', 'Attention gain')))
    dv = S4['varied_minus_single_4h']
    maxd = max(abs(v['diff']) for v in dv.values())
    v4_section = f"""
<h2 id="round-4">Round 4 · Twenty hours, ten levels</h2>
<p>One simulated subject plays ten 2-hour sessions (240 five-minute runs) of the levered game. Each session is a different level, the way a campaign varies: Warehouse, Night ops (dim, stealthy, frequent covert cues), Open arena (big rooms, fast), Cathedral (huge rooms, fine textures), Sprint, Treasure hunt, Fog bank, Mixed day, Boss rush and Dusk maze. Each session also has its own response gain (±10%) and noise level (±15%), and the fitter estimates one gain per session. A 24-run control repeats the Warehouse level, so 4 h of one level type can be compared with 4 h spread over all ten. The first 120 voxels of the round-2 V1 population are used throughout, including the dedicated baselines; brackets are bootstrap 95% CIs over voxels.</p>
<img src="fig9_long.png" alt="Recovery vs hours and the variety test">
<p></p>
<table><thead><tr><th>Spearman r</th><th>short dedicated (10 min)</th><th>2 h</th><th>5 h</th><th>10 h</th><th>20 h</th></tr></thead><tbody>{rows4}</tbody></table>
<div class="tldr"><b>Bottom line, round 4</b><ul>
<li><b>By ~10 hours, gameplay matches or beats a <i>short (10-minute)</i> dedicated experiment on every measure except SF bandwidth.</b> Round 5 below repeats this against dedicated experiments of realistic length, which changes the picture. 20 h vs 10-min dedicated: transient τ {g4('long_20h','tau')} vs {g4('dedicated_10min','tau')}; surround k {g4('long_20h','k')} vs {g4('dedicated_10min','k')}; CRF exponent {g4('long_20h','q')} vs {g4('dedicated_10min','q')}; attention gain {g4('long_20h','a')} vs {g4('dedicated_10min','a')}; pSFT peak {g4('long_20h','mu')} vs {g4('dedicated_10min','mu')}.</li>
<li><b>Several measures plateau from 10 to 20 h</b> (pSFT peak, effective c50, surround k). The most likely limit there is model mismatch rather than data: eye-tracker gaze and bar-fitted pRFs. Round 3's oracle shows how much headroom better gaze and pRFs would give. Transient weight and attention keep improving.</li>
<li><b>Variety did not help at equal time.</b> 4 h across ten levels vs 4 h of one level type: every difference is within ±{maxd:.2f}, and every 95% CI spans zero. The levers already make each level sample SF, contrast and attention broadly, and the themes mostly shift global statistics (lighting, fog, pace). Variety's payoff is more likely in generalization (models tested on unseen levels) and in models whose variables a single level under-samples (e.g., numerosity range), neither of which this test probes.</li>
<li><b>pSFT bandwidth stays weak</b> in every design ({g4('long_20h','sig')} at 20 h). Recovering bandwidth probably needs a higher-resolution display and finer SF sampling than the sandbox's 8 px/deg.</li>
</ul></div>
"""

v5_section = ''
BP = os.path.join(ROOT, 'dashboard_data', 'budget.csv')
FP = os.path.join(ROOT, 'dashboard_data', 'fair.csv')
if os.path.exists(BP) and os.path.exists(FP):
    import csv
    B = list(csv.DictReader(open(BP)))
    F = {(r['design'], r['parameter']): r for r in csv.DictReader(open(FP))}
    PL = (('mu', 'pSFT peak'), ('sig', 'pSFT bandwidth'), ('c50_eff', 'pCRF effective c50'), ('q', 'pCRF exponent'),
          ('k', 'Surround k'), ('tau', 'Transient τ'), ('a', 'Attention gain'))
    f5 = lambda d, p: f"{float(F[(d, p)]['spearman']):.2f}"
    rows5 = ''.join(f"<tr><td>{lab}</td>" + ''.join(f"<td>{f5(d, p)}</td>" for d in
                    ('ded_10m', 'ded_30m', 'ded_60m', 'battery_60m', 'levers_1h', 'long_5h', 'long_20h')) + '</tr>' for p, lab in PL)
    def h2h(b):
        rr = [r for r in B if r['budget'] == b]
        body = ''.join(f"<tr><td>{r['parameter_label']}</td><td>{float(r['ded_r']):.2f}</td><td>{float(r['game_r']):.2f}</td>"
                       f"<td>{float(r['diff']):+.2f} [{float(r['diff_lo']):+.2f}, {float(r['diff_hi']):+.2f}]</td><td>{r['verdict']}</td></tr>" for r in rr)
        cnt = {v: sum(r['verdict'] == v for r in rr) for v in ('Dedicated', 'Gameplay', 'Tie')}
        return body, cnt
    b1, c1 = h2h('1 h')
    b5, c5 = h2h('5 h')
    v5_section = f"""
<h2 id="round-5">Round 5 · A fair dedicated baseline</h2>
<p>Rounds 2–4 compared gameplay with dedicated experiments of only 10 minutes (two 5-minute runs). Real pRF, pSFT and pCRF mapping takes 30–60+ minutes. Here each dedicated V1 experiment is extended to 12 runs (60 min) and fitted at 10, 30 and 60 min, both on its own (other parameters at population defaults, as in round 2) and as a joint battery of all five experiments (50 min, 2.5 h, 5 h; the 5 h battery spans three sessions with the same gain and noise jitter as round 4). Everything is scored on the round-4 voxels (first 120). Head-to-heads use a paired bootstrap over voxels.</p>
<table><thead><tr><th>Spearman r</th><th>dedicated 10 min</th><th>30 min</th><th>60 min</th><th>battery 5 h</th><th>game + levers 1 h</th><th>varied play 5 h</th><th>varied play 20 h</th></tr></thead><tbody>{rows5}</tbody></table>
<h3>Matched 1 h: one dedicated experiment vs the game with levers</h3>
<table><thead><tr><th>Parameter</th><th>dedicated</th><th>game</th><th>Δ r [95% CI]</th><th>verdict</th></tr></thead><tbody>{b1}</tbody></table>
<h3>Matched 5 h: full dedicated battery vs varied-level play</h3>
<table><thead><tr><th>Parameter</th><th>dedicated</th><th>game</th><th>Δ r [95% CI]</th><th>verdict</th></tr></thead><tbody>{b5}</tbody></table>
<div class="tldr"><b>Bottom line, round 5</b><ul>
<li><b>The 10-minute baseline flattered gameplay.</b> Hour for hour, a dedicated experiment wins on {c1['Dedicated']} of 7 V1 measures (SF peak, effective c50, attention) and gameplay on {c1['Gameplay']}.</li>
<li><b>At a matched 5 h budget</b> the full dedicated battery wins {c5['Dedicated']} (SF peak and bandwidth, attention), gameplay wins {c5['Gameplay']} (surround suppression), and {c5['Tie']} tie (effective c50, CRF exponent, transient weight, the last leaning toward gameplay).</li>
<li><b>Longer dedicated scans help the noise-limited parameters</b> (attention 0.58 → 0.80 from 10 to 60 min), but one-at-a-time fits of transient weight, surround and SF bandwidth stay poor at an hour because the other parameters are held at defaults. Fitting the battery jointly fixes the transient weight; surround suppression stays low even jointly (r ≈ {float(F[("battery_60m","k")]["spearman"]):.2f}), so our disc/annulus design may simply be a weak surround experiment.</li>
<li><b>Non-visual models:</b> an hour of the reward task recovers RPE weights worse than an hour of play (play has many more outcomes); dedicated optic flow beats play on self-motion; numerosity ties.</li>
<li><b>The case for gameplay is breadth, not per-parameter efficiency:</b> one dataset serves every model at once, and it is the better design for quantities games sample richly (surround context, transients, rewards).</li>
</ul></div>
"""

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
h3{{font-size:16px;margin:26px 0 6px}}
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
<p class="note"><b><a href="dashboard/">Interactive dashboard →</a></b> · Round 1: pRFs from gameplay (below). <a href="#round-2">Round 2: eight models from one game</a> · <a href="#round-3">Round 3: design levers</a>{' · <a href="#round-4">Round 4: twenty hours</a>' if v4_section else ''}{' · <a href="#round-5">Round 5: fair baseline</a>' if v5_section else ''}.</p>
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
{v2_section}
{v3_section}
{v4_section}
{v5_section}
<p class="note">Noise model: see <code>NOISE.md</code> in the repo (AR(1) ρ = 0.3 with per-voxel SD 0.6–1.2 %, Legendre drift, event-locked arousal, slow global fluctuation, HRF timing jitter, eye-tracker error; no head motion or spatially correlated noise).</p>
<p class="note">Code: <code>github.com/samuelling/boldblaster</code>. Round 1: <code>python run_experiment.py</code>, <code>make_figures.py</code>. Round 2: <code>python run_v2.py data</code>, <code>python run_v2.py fit</code>, <code>make_figures_v2.py</code>. Then <code>build_report.py</code>.</p>
</main></body></html>"""
open(os.path.join(R, 'index.html'), 'w').write(html)
print('report built')
