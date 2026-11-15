"""One-off restructure of dashboard/index.html into an overview plus one tab per experiment.

Every chart, table and explainer is kept as it was; each experiment becomes a <section> (all but the open
one hidden): what we did, what it looked like, the result, then the key chart, with secondary material
folded into "More detail". Round 8 (three real games) is written fresh in round8_tab.html."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = '''
.bb-tabs { position: sticky; top: 0; z-index: 4; display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; padding: var(--cds-gap-sm) 0; margin: 0 0 var(--cds-gap-lg); background: var(--color-bg); border-bottom: 1px solid var(--color-border-line); }
.bb-tabs::-webkit-scrollbar { display: none; }
.bb-tabbtn { flex: none; font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); background: transparent; border: 1px solid transparent; border-radius: 999px; padding: 6px 12px; cursor: pointer; white-space: nowrap; }
.bb-tabbtn:hover { color: var(--color-fg); border-color: var(--color-border-line); }
.bb-tabbtn.on { color: var(--color-fg); background: var(--color-panel); border-color: var(--color-border-line); font-weight: var(--cds-font-weight-medium); }
.bb-intro { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap: var(--cds-gap-md); margin-bottom: var(--cds-gap-md); }
.bb-h3 { font-size: var(--cds-font-size-body); font-weight: var(--cds-font-weight-medium); margin: 0 0 var(--cds-gap-sm); }
.bb-did p, .bb-did li { line-height: 1.55; margin: 0 0 6px; }
.bb-did ul { margin: 0 0 6px; padding-left: 18px; }
.bb-fig { margin: 0; display: flex; flex-direction: column; gap: var(--cds-gap-sm); }
.bb-fig img { width: 100%; height: auto; border-radius: 6px; image-rendering: auto; background: var(--color-bg); min-height: 60px; }
.bb-fig figcaption { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); line-height: 1.5; }
.bb-ctl-row { margin: 0 0 var(--cds-gap-md); }
.bb-more, .bb-sub { margin-top: var(--cds-gap-md); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-md) var(--cds-pad-lg); background: color-mix(in srgb, var(--color-panel) 60%, transparent); }
.bb-more > summary, .bb-sub > summary { cursor: pointer; font-weight: var(--cds-font-weight-medium); }
.bb-more[open] > summary, .bb-sub[open] > summary { margin-bottom: var(--cds-gap-md); }
.bb-glance { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: var(--cds-gap-md); }
.bb-gcard { display: flex; flex-direction: column; gap: 6px; }
.bb-gcard p { margin: 0; line-height: 1.55; }
.bb-gcard .bb-h3 { margin: 0; }
.bb-gwide { grid-column: 1 / -1; }
.bb-open { align-self: flex-start; margin-top: auto; font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg); background: transparent; border: 1px solid var(--color-border-line); border-radius: 999px; padding: 5px 12px; cursor: pointer; }
.bb-open:hover { background: color-mix(in srgb, var(--color-fg) 6%, transparent); }
.bb-gap { margin-top: var(--cds-gap-xl); }
.bb-tab { scroll-margin-top: 60px; }
'''
IMGS = ['prf', 'models', 'hours', 'dedicated', 'ventral', 'seqsim', 'games']
src = open(os.path.join(HERE, 'index.html')).read()

head = src[:src.index('<div class="bb" data-read lang="en">')]
head = head.replace('</style>', CSS + '</style>', 1)
footer = src[src.index('<footer class="bb-foot" id="methods">'):src.index('</footer>') + len('</footer>')]
scripts = src[src.index('</div>\n\n<script src="core.js">'):]


def section(sid):
    a = src.index(f'<section class="bb-sec" id="{sid}">')
    body = src[a + len(f'<section class="bb-sec" id="{sid}">'):src.index('</section>', a)]
    t = body.index('<div class="bb-tldr">')
    h, tail = body[:t], body[t:]
    te = tail.index('</div></div>') + len('</div></div>')
    tldr, rest = tail[:te], tail[te:]
    desc = re.search(r'<p class="bb-desc">(.*?)</p>', h, re.S).group(1)
    ctl = re.search(r'(<div class="bb-controls">.*)\n  </div>\s*$', h, re.S)
    blocks = [b for b in re.split(r'\n(?=  <div class=)', rest) if b.strip()]
    return dict(desc=desc, controls=ctl.group(1) if ctl else '', tldr=tldr, blocks=blocks)


PARAM_SELECT = re.search(r'<label>Parameter\n.*?</label>', src, re.S).group(0)


def tab(tid, num, rnd, title, did, img, cap, tldr, primary, more=None, more_label='', controls=''):
    tl = tldr.replace('<span class="bb-tldr-tag">TL;DR</span>', '<span class="bb-tldr-tag">RESULT</span>')
    out = [f'<section class="bb-sec bb-tab" id="{tid}" hidden>',
           f'  <div class="bb-sec-head"><div><p class="bb-num">{num} · {rnd}</p><h2 class="bb-h2">{title}</h2></div></div>',
           '  <div class="bb-intro">',
           f'    <div class="bb-card bb-did"><h3 class="bb-h3">What we did</h3><p>{did}</p></div>']
    if img:
        out.append(f'    <figure class="bb-card bb-fig" id="{tid}-fig"><img data-draw="img" data-img="{img}" data-source="img_{img}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
    out.append('  </div>')
    out.append('  ' + tl)
    if controls:
        out.append(f'  <div class="bb-controls bb-ctl-row">{controls}</div>')
    out += primary
    if more:
        out.append(f'  <details class="bb-more" id="{tid}-more"><summary>More detail: {more_label}</summary>')
        out += more
        out.append('  </details>')
    out.append('</section>\n')
    return '\n'.join(out)


R = {k: section(k) for k in ('recovery', 'fair', 'hours', 'voxels', 'prf', 'noise', 'round6', 'round7')}
fix = lambda s: s.replace('Uses the parameter chosen above.', 'Uses the parameter picked here.').replace(' Uses the parameter chosen above', '')

tabs = []
tabs.append(tab('exp-prf', 'Experiment 1', 'Round 1', 'Can pRFs be mapped from free gameplay?',
                'A numpy first-person game is played at 10 Hz on a 24 × 18° screen while a gaze model (fixations, saccades, NPC pursuit) moves the eyes; 500 simulated V1 voxels with Gaussian pRFs respond, and their pRFs are fitted back from 5 minutes to 2 hours of play, with true gaze, a realistic in-bore eye tracker or no gaze at all, against 5–15 minutes of standard drifting bars.',
                'prf', 'Game frames with the gaze position (red cross) and a frame of the bar localizer, with the retinal contrast energy each produces.',
                R['prf']['tldr'], R['prf']['blocks']))
tabs.append(tab('exp-models', 'Experiment 2', 'Rounds 2–3', 'Can one gameplay dataset serve many models at once?',
                fix(R['recovery']['desc']) + ' Round 2 plants seven V1 tuning parameters (SF peak and bandwidth, contrast response, surround suppression, a transient channel, attention gain) plus numerosity, self-motion and striatal reward populations; round 3 adds design levers: covert-attention cues and textures that decorrelate SF from contrast.',
                'models', 'Top: the round-2 game. Bottom: the same game with round 3’s design levers (decorrelated textures, varied contrast).',
                R['recovery']['tldr'], R['recovery']['blocks'], controls=R['recovery']['controls']))
tabs.append(tab('exp-hours', 'Experiment 3', 'Round 4', 'How much play does it take, and does level variety help?',
                fix(R['hours']['desc']),
                'hours', 'Five of the ten level themes: Warehouse, Night ops, Open arena, Cathedral and Sprint.',
                R['hours']['tldr'], R['hours']['blocks'], controls=f'<div class="bb-controls">{PARAM_SELECT}</div>'))
tabs.append(tab('exp-fair', 'Experiment 4', 'Round 5', 'How does gameplay compare with dedicated experiments of realistic length?',
                fix(R['fair']['desc']),
                'dedicated', 'The dedicated V1 battery: band-pass noise (SF tuning), a contrast series, discs and annuli (surround), flicker rates (transient) and attention blocks.',
                R['fair']['tldr'], R['fair']['blocks'][:1], more=R['fair']['blocks'][1:], more_label='head-to-head tables with paired intervals',
                controls=f'<div class="bb-controls">{PARAM_SELECT}</div>'))
r6 = R['round6']['blocks']
tabs.append(tab('exp-attn', 'Experiment 5', 'Round 6', 'Attention dynamics, gaze tolerance of large receptive fields, and one game lever',
                fix(R['round6']['desc']),
                'ventral', 'A game frame and three of the content maps the ventral population pools: NPCs, one wall texture, the floor.',
                R['round6']['tldr'],
                ['  <details class="bb-sub" id="r6-a" open><summary>A · Can the time course of attention be recovered?</summary>', r6[0], '  </details>',
                 '  <details class="bb-sub" id="r6-b"><summary>B · Do large receptive fields buy gaze tolerance?</summary>', r6[1], '  </details>',
                 '  <details class="bb-sub" id="r6-c"><summary>C · Does a tighter attention cue help?</summary>', r6[2], '  </details>']))
tabs.append(tab('exp-cst', 'Experiment 6', 'Round 7', 'Spatiotemporal pRFs: can gameplay separate sustained from transient channels?',
                fix(R['round7']['desc']),
                'seqsim', 'One sequential SEQ-SIM trial, every other frame: two full-field flashes with a gap between them.',
                R['round7']['tldr'], R['round7']['blocks'][:1], more=R['round7']['blocks'][1:], more_label='why the controlled design doesn’t win'))
tabs.append(open(os.path.join(HERE, 'round8_tab.html')).read())
methods = ['<section class="bb-sec bb-tab" id="methods-tab" hidden>',
           '  <div class="bb-sec-head"><div><p class="bb-num">Reference</p><h2 class="bb-h2">How the simulation works</h2></div></div>',
           '  <div class="bb-card bb-did">' + footer.replace('<footer class="bb-foot" id="methods">', '').replace('</footer>', '')
           .replace('<details>', '<details open>') + '</div>',
           '  <h3 class="bb-h3 bb-gap">Signal, noise and what the fitter removes</h3>',
           f'  <p class="bb-desc">{R["noise"]["desc"]}</p>', '  ' + R['noise']['tldr']] + R['noise']['blocks'] + [
           '  <h3 class="bb-h3 bb-gap">Planted vs fitted, voxel by voxel</h3>',
           f'  <p class="bb-desc">{fix(R["voxels"]["desc"])}</p>', f'  {R["voxels"]["controls"].replace("<label>Design", PARAM_SELECT + "\n      <label>Design", 1)}',
           '  ' + R['voxels']['tldr']] + R['voxels']['blocks'] + ['</section>\n']
tabs.append('\n'.join(methods))

overview = open(os.path.join(HERE, 'overview.html')).read()
page = head + '<div class="bb" data-read lang="en">\n\n' + overview + '\n' + '\n'.join(tabs) + scripts
open(os.path.join(HERE, 'index.html'), 'w').write(page)
print('wrote', len(page))
