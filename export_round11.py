"""Round 11 tidy CSVs: Halcyon, the game built around the models, against Betamax and both tunnel
versions, every model; plus what an hour of Halcyon contains against its design targets."""
import csv
import json
import os

import numpy as np

import run_round9 as R9

D = 'dashboard_data'
GL = {'betamax': 'Betamax', 'betamaxt': 'Betamax + tunnels', 'betamaxu': 'Tuned tunnels', 'halcyon': 'Halcyon'}
rows = []


def add(part, model, ml, meas, mel, area, source, game, mins, value, metric='spearman', cond='realistic'):
    v = '' if value in ('', None) or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), 4)
    rows.append(dict(part=part, model=model, model_label=ml, measure=meas, measure_label=mel, area=area, source=source,
                     game=game, game_label=GL[game], minutes=mins, condition=cond, value=v, metric=metric))


# everything round 10 already tabulated for the three Betamax conditions
for r in csv.DictReader(open(f'{D}/round10.csv')):
    rows.append(r)
# Halcyon: the visual models from the round-8 tables
for r in csv.DictReader(open(f'{D}/round8_v1.csv')):
    if r['game'] == 'halcyon' and r['parameter'] != 'c50':
        add('vision', 'v1', 'V1 tuning', r['parameter'], r['parameter_label'], 'V1', 'bold', 'halcyon', int(r['minutes']), r['spearman'], cond=r['condition'])
AREA_O = {'Numerosity': 'Posterior parietal numerosity maps', 'Self-motion': 'hMT+ / MST', 'Striatal RPE': 'Ventral striatum',
          'Spatiotemporal (CST)': 'V1-V3'}
for r in csv.DictReader(open(f'{D}/round8_other.csv')):
    if r['game'] == 'halcyon':
        add('other', r['model'], r['model'], r['measure'], r['measure_label'], AREA_O.get(r['model'], ''), 'bold', 'halcyon', int(r['minutes']), r['spearman'], cond=r['condition'])
ld = lambda f: json.load(open(f)) if os.path.exists(f) else {}
nav = ld('data_v2/results_round11_nav.json'); cog = ld('data_v2/results_round11_cog.json'); new = ld('data_v2/results_round11_new.json')
NAVM = [('grid_move', 'Grid code, movement', 'phi', 'Grid orientation', 'phase'), ('grid_gaze', 'Grid code, eye movements', 'phi', 'Grid orientation', 'phase'),
        ('facing', 'Facing direction', 'psi', 'Preferred direction', 'phase'), ('facing', 'Facing direction', 'kappa', 'Tuning sharpness', 'spearman'),
        ('goal', 'Distance to goal', 'w', 'Path vs Euclidean weight', 'spearman'), ('boundary', 'Boundary distance', 'd', 'Preferred boundary distance', 'spearman'),
        ('evbound', 'Event boundaries', 'b', 'Boundary response gain', 'spearman')]
# Halcyon's drift diffusion and value learning run on the calls (Betamax's on firefight bursts):
# the same model and model id, so the rows line up across games
COGM = [('ddm', 'Drift diffusion (firefight)', 'v', 'Drift scale v'), ('ddm', 'Drift diffusion (firefight)', 'a', 'Boundary a'),
        ('rl', 'Value learning', 'alpha', 'Learning rate'), ('novelty', 'Place familiarity', 'e', 'Encoding strength'),
        ('novelty', 'Place familiarity', 'f', 'Forgetting rate')]
for mins in (30, 60):
    k = f'halcyon_{mins}'
    if k in nav:
        for mo, ml, me, mel, met in NAVM:
            add('nav', mo, ml, me, mel, R9.AREA[mo][0], 'bold', 'halcyon', mins, nav[k][mo].get(me), met)
    if k in cog:
        for mo, ml, me, mel in COGM:
            for src in ('behaviour', 'bold', 'agree'):
                v = cog[k][mo][me].get(src)
                if v is not None:
                    add('cog', mo, ml, me, mel, R9.AREA[mo][0], src, 'halcyon', mins, v)
    if k in new:
        for mo, ml, meas, area in [('pathint', 'Path integration', [('sh', 'Heading noise'), ('gd', 'Distance gain'), ('la', 'Leak')], 'Entorhinal cortex / hippocampus'),
                                   ('mapupd', 'Map updating', [('alpha', 'Update rate'), ('tau', 'Route-choice temperature')], 'Hippocampus'),
                                   ('bandit', 'Stairwell bandit', [('alpha', 'Learning rate'), ('lam', 'Risk weight')], 'Ventral striatum')]:
            r = new[k].get(mo, {})
            for me, mel in meas:
                for src in ('behaviour', 'bold', 'agree'):
                    if me in r and src in r[me]:
                        add('new', mo, ml, me, mel, area, src, 'halcyon', mins, r[me][src])
F = ['part', 'model', 'model_label', 'measure', 'measure_label', 'area', 'source', 'game', 'game_label', 'minutes', 'condition', 'value', 'metric']
with open(f'{D}/round11.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, F); w.writeheader(); w.writerows(rows)

# an hour of Halcyon against its design targets
rt = ld('data_v2/results_round11_rates.json')
c60 = cog.get('halcyon_60', {}); n60 = nav.get('halcyon_60', {}); w60 = new.get('halcyon_60', {})
S = [('calls', 'Shoot-or-hold calls per hour', '300+', f"{rt.get('calls', 0):.0f}"),
     ('stairs', 'Stair crossings (bandit trials) per hour', '100+', f"{w60.get('bandit', {}).get('trials', rt.get('stairs', 0)):.0f}"),
     ('stale', 'Shut security doors run into per hour', '60+', f"{rt.get('stale', 0):.0f}"),
     ('homing', 'Blackout walks usable as homing trials per hour', '25+', f"{w60.get('pathint', {}).get('trials', rt.get('homing', 0)):.0f}"),
     ('pickups', 'Guests reached per hour', '30+', f"{rt.get('pickups', 0):.0f}"),
     ('rescued', 'Guests walked out per hour', '', f"{rt.get('rescued', 0):.0f}"),
     ('junctions', 'Route choices at junctions per hour', '', f"{w60.get('mapupd', {}).get('junctions', rt.get('junctions', 0)):.0f}"),
     ('dark', 'Time in the dark', 'under 15%', f"{100 * rt.get('dark', 0):.0f}%"),
     ('outdoors', 'Time on the terrace', '10-20%', f"{100 * rt.get('outdoors', 0):.0f}%"),
     ('deaths', 'Deaths per hour', '', f"{rt.get('deaths', 0):.0f}"),
     ('dec', 'Decisions per minute (DDM opportunities)', '', f"{c60.get('ddm', {}).get('per_min', 0):.1f}"),
     ('gap', 'Median gap between decisions (s)', '4-10', f"{c60.get('ddm', {}).get('median_gap', 0):.1f}"),
     ('entries', 'New places entered per minute', '', f"{c60.get('novelty', {}).get('entries_per_min', 0):.0f}"),
     ('bounds', 'Event boundaries per minute', '', f"{n60.get('evbound', {}).get('per_min', 0):.1f}")]
with open(f'{D}/round11_stats.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['measure', 'label', 'target', 'halcyon']); w.writerows(S)

rows_a = [r for r in csv.DictReader(open(f'{D}/areas.csv')) if r['round'] != '11']
rows_a.append(dict(round='11', model='Stairwell bandit', parameter='Learning rate, risk weight', area='Ventral striatum',
                   basis="Halcyon's stair choices: each flight's ambush risk drifts; BOLD is the prediction error on reaching the landing", noise_mult='1.3'))
with open(f'{D}/areas.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows_a[0])); w.writeheader(); w.writerows(rows_a)
print(len(rows), 'rows')
