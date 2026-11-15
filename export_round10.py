"""Round 10 tidy CSVs: Betamax before and after the Diggings, every model, plus what the tunnels add."""
import csv
import json
import os

import numpy as np

import run_round9 as R9

D = 'dashboard_data'
GL = {'betamax': 'Betamax', 'betamaxt': 'Betamax + tunnels'}
rows = []


def add(part, model, ml, meas, mel, area, source, game, mins, value, metric='spearman', cond='realistic'):
    v = '' if value in ('', None) or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), 4)
    rows.append(dict(part=part, model=model, model_label=ml, measure=meas, measure_label=mel, area=area, source=source,
                     game=game, game_label=GL[game], minutes=mins, condition=cond, value=v, metric=metric))


# vision and the other round-8 models, from the round-8 tables (betamax and betamaxt rows)
for r in csv.DictReader(open(f'{D}/round8_v1.csv')):
    if r['game'] in GL and r['parameter'] != 'c50':
        add('vision', 'v1', 'V1 tuning', r['parameter'], r['parameter_label'], 'V1', 'bold', r['game'], int(r['minutes']), r['spearman'], cond=r['condition'])
AREA_O = {'Numerosity': 'Posterior parietal numerosity maps', 'Self-motion': 'hMT+ / MST', 'Striatal RPE': 'Ventral striatum',
          'Spatiotemporal (CST)': 'V1-V3'}
for r in csv.DictReader(open(f'{D}/round8_other.csv')):
    if r['game'] in GL:
        add('other', r['model'], r['model'], r['measure'], r['measure_label'], AREA_O.get(r['model'], ''), 'bold', r['game'], int(r['minutes']), r['spearman'], cond=r['condition'])
# navigation and cognition: betamax from round 9, betamaxt from round 10
nav9 = json.load(open('data_v2/results_round9_nav.json')); cog9 = json.load(open('data_v2/results_round9_cog.json'))
nav10 = json.load(open('data_v2/results_round10_nav.json')); cog10 = json.load(open('data_v2/results_round10_cog.json'))
new10 = json.load(open('data_v2/results_round10_new.json')) if os.path.exists('data_v2/results_round10_new.json') else {}
NAVM = [('grid_move', 'Grid code, movement', 'phi', 'Grid orientation', 'phase'), ('grid_gaze', 'Grid code, eye movements', 'phi', 'Grid orientation', 'phase'),
        ('facing', 'Facing direction', 'psi', 'Preferred direction', 'phase'), ('facing', 'Facing direction', 'kappa', 'Tuning sharpness', 'spearman'),
        ('goal', 'Distance to goal', 'w', 'Path vs Euclidean weight', 'spearman'), ('boundary', 'Boundary distance', 'd', 'Preferred boundary distance', 'spearman'),
        ('evbound', 'Event boundaries', 'b', 'Boundary response gain', 'spearman')]
COGM = [('ddm', 'Drift diffusion (firefight)', 'v', 'Drift scale v'), ('ddm', 'Drift diffusion (firefight)', 'a', 'Boundary a'),
        ('rl', 'Value learning', 'alpha', 'Learning rate'), ('novelty', 'Place familiarity', 'e', 'Encoding strength'),
        ('novelty', 'Place familiarity', 'f', 'Forgetting rate')]
for g, nav, cog in (('betamax', nav9, cog9), ('betamaxt', nav10, cog10)):
    for mins in (30, 60):
        k = f'{g}_{mins}'
        if k not in nav:
            continue
        for mo, ml, me, mel, met in NAVM:
            add('nav', mo, ml, me, mel, R9.AREA[mo][0], 'bold', g, mins, nav[k][mo].get(me), met)
        for mo, ml, me, mel in COGM:
            for src in ('behaviour', 'bold', 'agree'):
                v = cog[k][mo][me].get(src)
                if v is not None:
                    add('cog', mo, ml, me, mel, R9.AREA[mo][0], src, g, mins, v)
NEWM = [('ddm_id', 'Drift diffusion (friend or foe)', [('v', 'Drift scale v'), ('a', 'Boundary a')], 'Pre-SMA / intraparietal sulcus'),
        ('pathint', 'Path integration', [('sh', 'Heading noise'), ('gd', 'Distance gain'), ('la', 'Leak')], 'Entorhinal cortex / hippocampus'),
        ('mapupd', 'Map updating', [('alpha', 'Update rate'), ('tau', 'Route-choice temperature')], 'Hippocampus')]
for mins in (30, 60):
    k = f'betamaxt_{mins}'
    if k not in new10:
        continue
    for mo, ml, meas, area in NEWM:
        r = new10[k].get(mo, {})
        for me, mel in meas:
            if me not in r:
                continue
            for src in ('behaviour', 'bold', 'agree'):
                if src in r[me]:
                    add('new', mo, ml, me, mel, area, src, 'betamaxt', mins, r[me][src])
F = ['part', 'model', 'model_label', 'measure', 'measure_label', 'area', 'source', 'game', 'game_label', 'minutes', 'condition', 'value', 'metric']
with open(f'{D}/round10.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, F); w.writeheader(); w.writerows(rows)

# what the tunnels add over an hour
def tunnel_stats(n=12):
    tot = dict(under=0, frames=0, junctions=0, stale=0, homing=0, figures=0, deliveries=0, collapses=0, lights=0, descents=0)
    for i in range(n):
        f = f'/home/claude/bbx/runs_dig/betamaxt{i}_events.json'
        if not os.path.exists(f):
            continue
        ev = json.load(open(f))
        R = [json.loads(l) for l in open(f'/home/claude/bbx/runs_dig/betamaxt{i}_log.jsonl')]
        tot['under'] += sum(r['under'] for r in R); tot['frames'] += len(R)
        tot['junctions'] += len(ev['junctions']); tot['stale'] += len(ev['stale'])
        tot['homing'] += sum(1 for h in ev['homing'] if h[10] != 0); tot['figures'] += len(ev['figures'])
        for e in ev['digev']:
            tot[{'deliveries': 'deliveries', 'collapses': 'collapses', 'lightsOut': 'lights', 'descents': 'descents'}.get(e[1], 'x')] = tot.get({'deliveries': 'deliveries', 'collapses': 'collapses', 'lightsOut': 'lights', 'descents': 'descents'}.get(e[1], 'x'), 0) + 1
    return tot
T = tunnel_stats()
mins = T['frames'] / 600 or 1
st = [('under', 'Time below ground', '', f"{100 * T['under'] / max(T['frames'], 1):.0f}%"),
      ('descents', 'Trips down a hatch per hour', '0', f"{T['descents'] * 60 / mins:.0f}"),
      ('junctions', 'Route choices at junctions per hour', '0', f"{T['junctions'] * 60 / mins:.0f}"),
      ('collapses', 'Cave-ins per hour', '0', f"{T['collapses'] * 60 / mins:.0f}"),
      ('stale', 'Cave-ins run into per hour', '0', f"{T['stale'] * 60 / mins:.0f}"),
      ('homing', 'Blackout homing walks usable as trials per hour', '0', f"{new10.get('betamaxt_60', {}).get('pathint', {}).get('trials', 0) * 60 / mins:.0f}"),
      ('figures', 'Friend-or-foe decisions per hour', '0', f"{T['figures'] * 60 / mins:.0f}"),
      ('deliveries', 'Ammunition deliveries per hour', '0', f"{T['deliveries'] * 60 / mins:.0f}")]
for g, nav, cog in (('betamax', nav9, cog9), ('betamaxt', nav10, cog10)):
    pass
extra = [('path_eucl', 'Path vs Euclidean distance correlation', lambda nav, cog, g: f"{nav[g + '_60']['goal']['path_eucl_corr']:.2f}"),
         ('bounds', 'Event boundaries per minute', lambda nav, cog, g: f"{nav[g + '_60']['evbound']['per_min']:.1f}"),
         ('entries', 'New places entered per minute', lambda nav, cog, g: f"{cog[g + '_60']['novelty']['entries_per_min']:.0f}"),
         ('revisit', 'Share of entries that are revisits', lambda nav, cog, g: f"{cog[g + '_60']['novelty']['revisit_frac']:.2f}"),
         ('dec', 'Firefight decisions per minute', lambda nav, cog, g: f"{cog[g + '_60']['ddm']['per_min']:.1f}")]
with open(f'{D}/round10_stats.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['measure', 'label', 'betamax', 'betamaxt'])
    for k, l, a, b in st:
        w.writerow([k, l, a or '0', b])
    for k, l, fn in extra:
        w.writerow([k, l, fn(nav9, cog9, 'betamax'), fn(nav10, cog10, 'betamaxt') if 'betamaxt_60' in nav10 else ''])
print(len(rows), 'rows')

# brain areas: replace round 9's "not tested" rows with round 10's models
rows_a = [r for r in csv.DictReader(open(f'{D}/areas.csv')) if not r['model'].endswith('(not tested)') and r['round'] != '10']
rows_a += [
    dict(round='10', model='Path integration', parameter='Heading noise, distance gain, leak', area='Entorhinal cortex / hippocampus',
         basis="The Palace's lantern model, on Betamax blackout walks; BOLD is positional uncertainty while walking in the dark", noise_mult='1.5'),
    dict(round='10', model='Map updating', parameter='Update rate, route-choice temperature', area='Hippocampus',
         basis="The Palace's East Wing model, on Betamax junction choices after cave-ins; BOLD is surprise at a cave-in and believed distance at junctions", noise_mult='1.4'),
    dict(round='10', model='Drift diffusion (friend or foe)', parameter='Drift scale, boundary', area='Pre-SMA / intraparietal sulcus',
         basis='Accumulator activity for identification decisions on figures in the tunnels', noise_mult='1.0')]
with open(f'{D}/areas.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows_a[0])); w.writeheader(); w.writerows(rows_a)
