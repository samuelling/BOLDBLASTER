"""Adds TL;DR boxes, explainers and the dynamic best-design sentence to index.html."""
h = open('index.html').read()

h = h.replace("[data-dash-source-state=refreshing] { opacity: .5; }", """[data-dash-source-state=refreshing] { opacity: .5; }
.bb-tldr { display: flex; gap: var(--cds-gap-md); align-items: flex-start; background: color-mix(in srgb, #6250d6 7%, var(--color-panel)); border: 1px solid color-mix(in srgb, #6250d6 28%, var(--color-border-line)); border-radius: var(--cds-radius); padding: var(--cds-pad-md) var(--cds-pad-lg); margin: 0 0 var(--cds-gap-md); line-height: 1.55; }
.bb-tldr-tag { flex: none; font-size: 11px; font-weight: 700; letter-spacing: .12em; color: #fff; background: linear-gradient(90deg, #2a78d6, #6250d6); border-radius: 999px; padding: 3px 9px; margin-top: 2px; }
.bb-tldr p { margin: 0 0 6px; }
.bb-tldr p:last-child { margin-bottom: 0; }
.bb-tldr b { font-variant-numeric: tabular-nums; }
.bb-explain { margin-top: var(--cds-gap-md); border-top: 1px dashed var(--color-border-line); padding-top: var(--cds-gap-sm); font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); line-height: 1.6; }
.bb-explain summary { cursor: pointer; color: var(--color-fg); font-weight: var(--cds-font-weight-medium); list-style: none; display: inline-flex; align-items: center; gap: 6px; }
.bb-explain summary::-webkit-details-marker { display: none; }
.bb-explain summary::before { content: "+"; display: inline-grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; border: 1px solid var(--color-border-line); font-size: 13px; line-height: 1; }
.bb-explain[open] summary::before { content: "\\2212"; }
.bb-explain h4 { margin: var(--cds-gap-md) 0 2px; font-size: var(--cds-font-size-caption); text-transform: uppercase; letter-spacing: .08em; color: var(--color-fg); }
.bb-explain p, .bb-explain ul { margin: 0 0 4px; }
.bb-explain ul { padding-left: 18px; }
.bb-signal { border-left: 3px solid #1baf7a; padding-left: 10px; }
.bb-dyn { margin: 0 0 var(--cds-gap-md); }""")


def explain(what, read, signal, caveat=None):
    s = '<details class="bb-explain"><summary>What am I looking at?</summary>\n'
    s += f'<h4>What it shows</h4>{what}\n<h4>How to read it</h4>{read}\n'
    s += f'<div class="bb-signal"><h4>Signal vs noise</h4>{signal}</div>\n'
    if caveat:
        s += f'<h4>Caveats</h4>{caveat}\n'
    return s + '</details>'


def sub(old, new):
    global h
    assert old in h, old[:80]
    h = h.replace(old, new, 1)


TLDR = '<div class="bb-tldr"><span class="bb-tldr-tag">TL;DR</span><div>\n    <p>{}</p>\n  </div></div>\n'

# ---------- hero
sub('''  <p class="bb-stat" style="margin-top: var(--cds-gap-md)">Every value is Spearman r between planted and fitted parameters across simulated voxels: 1 means the ordering of voxels is recovered perfectly.</p>''',
    '''  <p class="bb-stat" style="margin-top: var(--cds-gap-md)">Every value is Spearman r between planted and fitted parameters across simulated voxels: 1 means the ordering of voxels is recovered perfectly.</p>
  ''' + explain(
        "<p>BOLDBLASTER is a simulation. We play a first-person game, plant known tuning parameters in simulated voxels (the ground truth), generate BOLD with realistic noise, and fit the parameters back out exactly as one would with real data. Because the truth is known, we can score how well each experimental design recovers it.</p>",
        "<p>The tiles compare the best gameplay result with the classic dedicated experiment for the same parameter (10 minutes, built for that one quantity). Spearman r measures whether voxels are put in the right order: 0.9 means the fitted values rank voxels almost exactly as the truth does; 0.3 means only a weak trend.</p>",
        "<p>Treat differences under about 0.1 as noise for the 300-voxel rounds and under about 0.15 for the 120-voxel round 4. The sections below show 95% intervals so any comparison can be checked.</p>",
        "<p>Everything here is simulated. The fitter knows each model's functional form (not its parameters), and some real-world noise sources are not simulated; see section 05.</p>"))

# ---------- 01
sub('''  <div class="bb-card">
    <div class="bb-chart" id="forest"''',
    '  ' + TLDR.format("Gameplay wins where a game naturally varies a quantity a lot or produces many events: transient responses (saccades, camera turns), numerosity (coin clusters) and reward outcomes. It needs hours, design levers or both to match dedicated experiments on SF tuning and attention. Parameters that interact (surround suppression, transient weight, CRF exponent) are barely recoverable one experiment at a time, so fitting them jointly is what matters.") +
    '''  <div class="bb-card">
    <p class="bb-dyn bb-stat">For <b data-draw="sel-param" data-source="recovery" data-field="parameter_label">—</b>, the best realistic design (no oracle) is <b data-draw="best-design" data-source="recovery" data-field="design_label">—</b> at r = <b data-draw="best-r" data-source="recovery" data-field="spearman">—</b>; the dedicated 10-minute experiment gives <b data-draw="ded-r" data-source="recovery" data-field="spearman">—</b>.</p>
    <div class="bb-chart" id="forest"''')
sub('''      <span>Ten levels (round 4)</span>
    </div>
  </div>''', '''      <span>Ten levels (round 4)</span>
    </div>
    ''' + explain(
    "<p>For the parameter you pick, one row per experimental design and the Spearman r between planted and fitted values across voxels.</p><ul><li><b>Dedicated experiment</b>: the classic design built for that parameter (band-pass noise for pSFT, a contrast series for pCRF, discs and annuli for surround suppression, refresh rates for the transient channel, attend-left/right blocks for attention, dot arrays, optic-flow blocks, a reward task), 10 minutes, fitting only that parameter with the others fixed at population defaults.</li><li><b>Battery fit jointly</b>: all five V1 experiments (50 min) fitted together with every parameter free.</li><li><b>Game</b>: 1–2 h of the round-2 game. <b>Game + design levers</b>: the round-3 game with covert-attention cues and SF/contrast-decorrelated textures.</li><li><b>Oracle</b>: the same data fitted with true gaze and true pRFs, an upper bound on what better eye tracking and localizers would buy.</li><li><b>Ten levels</b>: round 4, 2–20 h across ten level types on a 120-voxel subset (its dedicated row is re-scored on those same voxels).</li></ul>",
    "<p>Dots further right recover the parameter better. Horizontal lines separate rounds. Hover any row for minutes of data, voxel count and the interval.</p>",
    "<p>Whiskers are bootstrap 95% intervals over voxels. The shaded column is the dedicated experiment's interval: a design whose whisker sits entirely outside it is convincingly better or worse; overlap means treat it as a tie. As a rule of thumb, r is uncertain by about ±0.1 with 300 voxels and ±0.15 with 120.</p>",
    "<p>Compare within a round where possible (rounds 2–3 use 300 voxels, round 4 uses 120). Numerosity, self-motion and striatal rows have no intervals because their per-voxel fits were not saved, and their dedicated runs had 10 min against 60–120 min of play.</p>") + '''
  </div>''')

# ---------- 02
sub('''  <div class="bb-grid2">
    <div class="bb-card">
      <p class="bb-note" data-when="nonv1" hidden>Rounds 3 and 4''',
    '  ' + TLDR.format('Recovery climbs steeply up to about 10 hours, then flattens for SF peak, effective c50 and surround suppression: past that point the limit is gaze and pRF accuracy, not more play. Spreading the same 4 hours over ten level types instead of one changed recovery by at most <b data-draw="variety" data-source="variety">—</b> (r), well inside the noise.') +
    '''  <div class="bb-grid2">
    <div class="bb-card">
      <p class="bb-note" data-when="nonv1" hidden>Rounds 3 and 4''')
sub('''      <div class="bb-chart" id="hours-chart" data-draw="hours" data-source="recovery" style="min-height: 300px"></div>
    </div>''', '''      <div class="bb-chart" id="hours-chart" data-draw="hours" data-source="recovery" style="min-height: 300px"></div>
      ''' + explain(
    "<p>Spearman r for the chosen parameter as sessions accumulate: 2 h is session 1 (Warehouse), 5 h adds Night ops and half of Open arena, 10 h is five levels, 20 h all ten. Orange is gameplay with its 95% band; the blue dashed line and band are the dedicated 10-minute experiment on the same 120 voxels.</p>",
    "<p>Where the orange line rises above the blue band, gameplay has overtaken the dedicated experiment. A line that keeps rising means more hours still help; a flat line means something other than data quantity is limiting.</p>",
    "<p>Trust the overall trend from 2 to 10 to 20 h. Single steps can wobble: the 5 h point adds a dim, low-contrast level (Night ops), and with 120 voxels each point is uncertain by roughly ±0.15, so a dip of 0.1 between neighbours is not a real reversal.</p>",
    "<p>One simulated subject. Each session also has its own response gain and noise level, and the fitter estimates one gain per session.</p>") + '''
    </div>''')
sub('''      <div class="bb-chart" id="variety-chart" data-draw="varietychart" data-source="recovery" style="min-height: 300px"></div>
    </div>''', '''      <div class="bb-chart" id="variety-chart" data-draw="varietychart" data-source="recovery" style="min-height: 300px"></div>
      ''' + explain(
    "<p>A controlled test of level variety at equal time. Grey: 4 h of the Warehouse level (two sessions with different random seeds). Orange: 4 h made of about 24 minutes from each of the ten levels. Same voxels, same fitter.</p>",
    "<p>Each row is one V1 parameter; the grey bar joins the two estimates. A long bar would mean variety matters for that parameter.</p>",
    "<p>Every pair overlaps almost completely (hover for intervals): no parameter shows a detectable variety effect. The design levers already make each level sample SF, contrast and attention broadly, and the level themes mostly change global statistics such as lighting, fog and pace.</p>",
    "<p>This does not test generalization (fitting on some levels and predicting an unseen one) or models whose variables one level under-samples, such as the range of coin counts. Variety may pay off there.</p>") + '''
    </div>''')

# ---------- 03
sub('''  <div class="bb-card">
    <p class="bb-stat">Spearman r for this design and parameter (all voxels):''',
    '  ' + TLDR.format("This is where you see <i>how</i> a design succeeds or fails. A tight cloud along the diagonal is good recovery. A flat, horizontal cloud means the parameter is not identifiable from that data. Dots piled against the top or bottom edge are fits stuck at a parameter bound, common for dedicated surround and transient fits whose other parameters are held fixed.") +
    '''  <div class="bb-card">
    <p class="bb-stat">Spearman r for this design and parameter (all voxels):''')
sub('''    <div class="bb-chart" id="scatter" data-draw="scatter" data-source="v1_voxels" style="min-height: 380px"></div>
  </div>''', '''    <div class="bb-chart" id="scatter" data-draw="scatter" data-source="v1_voxels" style="min-height: 380px"></div>
    ''' + explain(
    "<p>One dot per simulated V1 voxel: its planted (true) value on the x-axis and the value fitted from the chosen design's data on the y-axis. Colour is eccentricity. Units: pSFT peak in log2 cycles/degree, bandwidth in octaves, effective c50 in contrast-energy units (log axes), CRF exponent unitless, surround suppression and transient weight as weights, attention as a fractional gain.</p>",
    "<ul><li><b>Along the dashed diagonal</b>: recovered.</li><li><b>Diagonal but tilted flatter</b>: shrinkage, fitted values pulled toward the middle (typical when the data are weak).</li><li><b>Horizontal smear</b>: the data barely constrain the parameter.</li><li><b>Rows of dots at the top or bottom edge</b>: fits hit a parameter bound.</li></ul><p>Use the eccentricity filter to see whether failures concentrate in the fovea or periphery.</p>",
    "<p>The r above the plot is for all voxels of this design and parameter, from the results file; the filter changes what you see, not that number. A visible trend among a few dozen voxels in one eccentricity band can easily be chance.</p>",
    "<p>Effective c50 is c50 / (1+k)<sup>1/q</sup>, the half-saturation point a contrast experiment can actually measure; raw c50 trades off with surround suppression and the exponent.</p>") + '''
  </div>''')

# ---------- 04
sub('''  <div class="bb-grid2">
    <div class="bb-card bb-tablewrap">
      <table class="bb-table" id="prf-table"''',
    '  ' + TLDR.format('For plain pRF mapping, 30 minutes of gameplay gives <b data-draw="cell" data-source="prf_round1" data-field="pos_err_median" data-where="design=game_eyetracker">—</b>° median position error, against <b data-draw="cell" data-source="prf_round1" data-field="pos_err_median" data-where="design=bars">—</b>° for 15 minutes of bars, so bars win overall. Beyond 9°, where bars never reach, gameplay wins (<b data-draw="cell" data-source="prf_round1" data-field="pos_err_ecc_9_12" data-where="design=game_eyetracker">—</b>° vs <b data-draw="cell" data-source="prf_round1" data-field="pos_err_ecc_9_12" data-where="design=bars">—</b>°). Ignoring eye position is catastrophic (<b data-draw="cell" data-source="prf_round1" data-field="pos_err_median" data-where="design=game_nogaze">—</b>°), and bars plus gameplay together is best (<b data-draw="cell" data-source="prf_round1" data-field="pos_err_median" data-where="design=bars%2Bgame">—</b>°).') +
    '''  <div class="bb-grid2">
    <div class="bb-card bb-tablewrap">
      <table class="bb-table" id="prf-table"''')
sub('''        <tbody data-draw="prftable"></tbody>
      </table>
    </div>''', '''        <tbody data-draw="prftable"></tbody>
      </table>
      ''' + explain(
    "<p>Round 1, the original question: can pRF centres and sizes be fitted from gameplay? This simulation used a simpler V1 (a compressive pRF, no SF tuning or transient channel) and 500 voxels. Rows: bars at fixation (5–15 min), gameplay with perfect gaze, a realistic in-bore eye tracker, or gaze ignored, without the arousal regressor, bars plus game, and the eye-tracker game from 5 min to 2 h.</p>",
    "<ul><li><b>Median error</b>: distance between fitted and true pRF centre, in degrees, over all voxels.</li><li><b>Error 9–12°</b>: the same for the far periphery.</li><li><b>Held-out bar R² / ceiling</b>: how well the fitted pRFs predict a bar run they never saw, as a fraction of the best possible (1 = as good as the true pRFs).</li></ul>",
    "<p>Below about 0.5° is excellent; above about 2° the pRF position is essentially not recovered. A ceiling fraction near 0 means the fit carries no usable information about responses to new stimuli.</p>") + '''
    </div>''')
sub('''      <div class="bb-chart" id="prf-chart" data-draw="prfscatter" data-source="prf_voxels prf_round1" style="min-height: 320px"></div>
    </div>''', '''      <div class="bb-chart" id="prf-chart" data-draw="prfscatter" data-source="prf_voxels prf_round1" style="min-height: 320px"></div>
      ''' + explain(
    "<p>Each dot is one of 500 voxels: its true eccentricity against how far its fitted pRF centre landed from the truth (log scale). The black steps are the median error in each eccentricity band; the dashed line is the edge of the bar stimulus at 9°.</p>",
    "<p>Click a row in the table to switch designs. Bars are low centrally and jump beyond 9°; gameplay is higher centrally but stays low in the periphery, because saccades carry the game screen out there.</p>",
    "<p>The steps summarise each band. Individual dots scatter by an order of magnitude even within a good design, so judge a design by its steps, not by outliers.</p>") + '''
    </div>''')

# ---------- 05
sub('''  <div class="bb-grid2">
    <div class="bb-card">
      <p class="bb-stat">Median SD over time across 500 round-1 voxels (%)</p>''',
    '  ' + TLDR.format('Gameplay keeps V1 driven but modulates any one voxel only <b data-draw="cell" data-source="signal_medians" data-field="median_sd" data-where="component=visual_game">—</b>% (SD over time), against <b data-draw="cell" data-source="signal_medians" data-field="median_sd" data-where="component=visual_bars">—</b>% for bars, below both the arousal signal (<b data-draw="cell" data-source="signal_medians" data-field="median_sd" data-where="component=systemic">—</b>%) and the noise (<b data-draw="cell" data-source="signal_medians" data-field="median_sd" data-where="component=noise">—</b>%). With noise removed, gameplay recovers pRFs to <b data-draw="cell" data-source="diagnostics" data-field="value" data-where="condition=visual only">—</b>°: the information is there, and the bottleneck is signal-to-noise. The noise model is moderate (about tSNR 110) and leaves out head motion and spatially correlated noise, so real data will need more time.') +
    '''  <div class="bb-grid2">
    <div class="bb-card">
      <p class="bb-stat">Median SD over time across 500 round-1 voxels (%)</p>''')
sub('''      <div class="bb-chart" id="signal-chart" data-draw="signal" data-source="signal_medians" style="min-height: 240px"></div>
    </div>''', '''      <div class="bb-chart" id="signal-chart" data-draw="signal" data-source="signal_medians" style="min-height: 240px"></div>
      ''' + explain(
    "<p>How big each part of a simulated voxel's time series is, as its standard deviation over time, median across 500 round-1 voxels: the visual response to bars, the visual response to gameplay, the arousal-linked systemic signal during gameplay, and the measurement noise.</p>",
    "<p>The ratio of a visual bar to the noise bar is the per-voxel signal-to-noise. Gameplay's visual bar is about a quarter of bars', so each minute of play carries far less information about a pRF than a minute of bars; that is why the hours curve in section 02 matters.</p>",
    "<p>These are medians; individual voxels vary (peripheral voxels get relatively more gameplay modulation). The systemic signal is shared across voxels, so an event-locked regressor removes most of it; the noise is not removable.</p>",
    "<p>Round 1 used a sustained-only V1. The transient channel added in later rounds raises gameplay's modulation (see the table beside this).</p>") + '''
    </div>''')
sub('''        <tbody data-draw="diagtable"></tbody>
      </table>
    </div>''', '''        <tbody data-draw="diagtable"></tbody>
      </table>
      ''' + explain(
    "<p>Two checks from round 1. The first five rows fit gameplay pRFs (120 voxels, 30 min, true gaze) adding one nuisance at a time: visual only (no noise at all), plus systemic arousal with and without the event-based regressor, plus measurement noise, and everything together. The last three rows give gameplay's visual modulation relative to bars for a sustained channel, a transient (change-driven) channel, and a 50/50 mix.</p>",
    "<p>In the first block, compare each row with “visual only”: the jump tells you which nuisance does the damage. In the second, a ratio of 1 would mean gameplay modulates V1 as strongly as bars.</p>",
    "<p>Noise, not arousal, is what breaks gameplay fits: the error jumps to several degrees only when noise is added. Arousal alone costs little once the logged-event regressor is used. A transient channel roughly triples gameplay's relative modulation, so the sustained-only round-1 numbers are pessimistic.</p>") + '''
    </div>''')
sub('''      <tbody data-draw="noisetable"></tbody>
    </table>
  </div>''', '''      <tbody data-draw="noisetable"></tbody>
    </table>
    ''' + explain(
    "<p>Every noise and nuisance component in the simulation, where it applies, its form and parameters, whether it is shared across voxels, and what the fitter does about it.</p>",
    "<p>Read across a row: a component the fitter models (drift, logged arousal, HRF timing, session gain) costs little; one it does not model (measurement noise, the slow global fluctuation) goes straight into the error. The last row lists what is not simulated at all.</p>",
    "<p>Measurement noise of 0.6–1.2% per TR (AR(1), ρ = 0.3) is about tSNR 110: plausible for 7T at about 2 mm or a good 3T voxel, while many 3T voxels are noisier. The hours needed scale roughly with noise variance, so doubling the noise SD roughly quadruples the time.</p>",
    "<p>Independent noise per voxel, no head motion and no separate cardiac or respiratory signals make this an optimistic noise model. Full details: NOISE.md in the repository.</p>") + '''
  </div>''')

open('index.html', 'w').write(h)
print(len(h.encode()), 'bytes;', h.count('class="bb-explain"'), 'explainers;', h.count('class="bb-tldr"'), 'tldrs')
