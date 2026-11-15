"""Round 9 results -> tidy CSVs for the dashboard, plus the brain-area table for every round."""
import csv
import json
import os

import numpy as np

import run_round9 as R9

D = 'dashboard_data'
nav = json.load(open('data_v2/results_round9_nav.json'))
cog = json.load(open('data_v2/results_round9_cog.json'))
GL = R9.LABEL


def num(x):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


rows = []
NAVM = [  # model, model label, measure, measure label, metric
    ('grid_move', 'Grid code, movement', 'phi', 'Grid orientation', 'phase'),
    ('grid_move', 'Grid code, movement', 'm', 'Hexadirectional depth', 'spearman'),
    ('grid_gaze', 'Grid code, eye movements', 'phi', 'Grid orientation', 'phase'),
    ('grid_gaze', 'Grid code, eye movements', 'm', 'Hexadirectional depth', 'spearman'),
    ('facing', 'Facing direction', 'psi', 'Preferred direction', 'phase'),
    ('facing', 'Facing direction', 'kappa', 'Tuning sharpness', 'spearman'),
    ('goal', 'Distance to goal', 'w', 'Path vs Euclidean weight', 'spearman'),
    ('boundary', 'Boundary distance', 'd', 'Preferred boundary distance', 'spearman'),
    ('evbound', 'Event boundaries', 'b', 'Boundary response gain', 'spearman'),
]
for g in R9.GAMES:
    for mins in (30, 60):
        r = nav[f'{g}_{mins}']
        for mod, ml, meas, mel, met in NAVM:
            rows.append(dict(part='nav', model=mod, model_label=ml, measure=meas, measure_label=mel,
                             area=R9.AREA[mod][0], source='bold', game=g, game_label=GL[g], minutes=mins,
                             value=num(r[mod].get(meas)), metric=met))
        rows.append(dict(part='nav', model='grid_gaze', model_label='Grid code, eye movements', measure='phi_oracle',
                         measure_label='Grid orientation (true gaze)', area=R9.AREA['grid_gaze'][0], source='bold',
                         game=g, game_label=GL[g], minutes=mins, value=num(nav[f'{g}_{mins}_oraclegaze']['grid_gaze']['phi']),
                         metric='phase'))
COGM = [
    ('ddm', 'Drift diffusion', 'v', 'Drift scale v', ('behaviour', 'bold', 'agree')),
    ('ddm', 'Drift diffusion', 'a', 'Boundary a', ('behaviour', 'bold', 'agree')),
    ('ddm', 'Drift diffusion', 'va', 'Drift x boundary', ('behaviour', 'bold', 'agree')),
    ('rl', 'Value learning', 'alpha', 'Learning rate', ('behaviour', 'bold', 'agree')),
    ('rl', 'Value learning', 'beta', 'Choice temperature', ('behaviour',)),
    ('novelty', 'Place familiarity', 'e', 'Encoding strength', ('bold',)),
    ('novelty', 'Place familiarity', 'f', 'Forgetting rate', ('bold',)),
]
for g in R9.GAMES:
    for mins in (30, 60):
        r = cog[f'{g}_{mins}']
        for mod, ml, meas, mel, srcs in COGM:
            for src in srcs:
                rows.append(dict(part='cog', model=mod, model_label=ml, measure=meas, measure_label=mel,
                                 area=R9.AREA[mod][0], source=src, game=g, game_label=GL[g], minutes=mins,
                                 value=num(r[mod][meas].get(src)), metric='spearman'))
F = ['part', 'model', 'model_label', 'measure', 'measure_label', 'area', 'source', 'game', 'game_label', 'minutes', 'value', 'metric']
with open(f'{D}/round9.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, F)
    w.writeheader()
    w.writerows(rows)

# what each game offers these models, over the hour
st = []
def srow(key, label, fn, fmt='{:.2f}'):
    d = dict(measure=key, label=label)
    for g in R9.GAMES:
        v = fn(g)
        d[g] = '' if v is None or not np.isfinite(v) else fmt.format(v)
    st.append(d)
srow('dir_coverage', 'Movement directions sampled (1 = balanced)', lambda g: max(0.0, nav[f'{g}_60']['grid_move']['dir_coverage']))
srow('gaze_coverage', 'Eye-movement directions sampled', lambda g: nav[f'{g}_60']['grid_gaze']['dir_coverage'])
srow('path_eucl', 'Path vs Euclidean distance correlation', lambda g: nav[f'{g}_60']['goal']['path_eucl_corr'])
srow('goal_time', 'Share of time with a goal', lambda g: nav[f'{g}_60']['goal']['goal_time'])
srow('bounds', 'Event boundaries per minute', lambda g: nav[f'{g}_60']['evbound']['per_min'], '{:.1f}')
srow('dec', 'Engage decisions per minute', lambda g: cog[f'{g}_60']['ddm']['per_min'], '{:.1f}')
srow('gap', 'Median gap between decisions (s)', lambda g: cog[f'{g}_60']['ddm']['median_gap'], '{:.1f}')
srow('diff_sd', 'Spread of decision difficulty (SD)', lambda g: cog[f'{g}_60']['ddm']['strength_sd'])
srow('hit', 'Outcome success rate', lambda g: cog[f'{g}_60']['rl']['hit_rate'])
srow('entries', 'New places entered per minute', lambda g: cog[f'{g}_60']['novelty']['entries_per_min'], '{:.0f}')
srow('revisit', 'Share of entries that are revisits', lambda g: cog[f'{g}_60']['novelty']['revisit_frac'])
with open(f'{D}/round9_stats.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['measure', 'label'] + R9.GAMES)
    w.writeheader()
    w.writerows(st)

# candidate brain area for every parameter the testbed recovers
A = [
    (1, 'pRF', 'pRF position and size', 'V1-V3', 'Retinotopic maps; pRF mapping (Dumoulin and Wandell 2008)'),
    ('2-8', 'pSFT', 'SF peak and bandwidth', 'V1', 'Population SF tuning (Aghajari et al. 2020)'),
    ('2-8', 'pCRF', 'Contrast response (effective c50, exponent)', 'V1', 'Population contrast response'),
    ('2-8', 'Surround', 'Surround suppression k', 'V1', 'Divisive normalization'),
    ('2-8', 'Transient', 'Transient weight', 'V1 (with hMT+)', 'Transient channels'),
    ('2-8', 'Attention', 'Attention gain', 'V1, controlled from IPS / FEF', 'Attentional gain in early visual cortex'),
    (6, 'Attention dynamics', 'Rise and decay time', 'V1, controlled from IPS / FEF', 'Event-locked attention kernel'),
    (6, 'Ventral category', 'Preferred-category weight', 'Ventral occipitotemporal (FFA, PPA)', 'Large, foveally biased pRFs'),
    (7, 'CST', 'Sustained/transient time constants and weight', 'V1-V3', 'Compressive spatiotemporal pRF (Kim et al. 2024)'),
    ('2-8', 'Numerosity', 'Preferred numerosity and width', 'Posterior parietal numerosity maps', 'Numerosity tuning (Harvey et al. 2013)'),
    ('2-8', 'Self-motion', 'Rotation preference, expansion weight', 'hMT+ / MST, V6, CSv', 'Optic-flow selective areas'),
    ('2-8', 'Striatal RPE', 'Prediction-error weight', 'Ventral striatum', 'Reward prediction error'),
]
for mod, ml, meas, mel, _ in NAVM:
    if meas in ('phi', 'psi', 'w', 'd', 'b'):
        A.append((9, ml, mel, R9.AREA[mod][0], {
            'grid_move': 'Hexadirectional running-direction signal (Doeller et al. 2010)',
            'grid_gaze': 'Hexadirectional gaze-direction signal (Nau et al. 2018)',
            'facing': 'Facing-direction tuning in RSC and SPL during free navigation',
            'goal': 'Path and Euclidean distance to goal (Howard et al. 2014)',
            'boundary': 'Boundary vector and border cells',
            'evbound': 'Hippocampal response at event boundaries (Ben-Yakov and Henson 2018)'}[mod]))
A += [(9, 'Drift diffusion', 'Drift scale and boundary', R9.AREA['ddm'][0], 'Evidence accumulation; boundary setting in pre-SMA'),
      (9, 'Value learning', 'Learning rate', R9.AREA['rl'][0], 'Reward prediction error with learned value'),
      (9, 'Place familiarity', 'Encoding and forgetting', R9.AREA['novelty'][0], 'Hippocampal novelty response'),
      (9, 'Path integration (not tested)', 'Heading noise, distance gain, leak', 'Entorhinal cortex, hippocampus, RSC',
       "The Palace's lantern model; needs homing trials, which none of these games has"),
      (9, 'Map updating (not tested)', 'Update rate, route-choice noise', 'Hippocampus, prefrontal cortex',
       "The Palace's East Wing model; needs route choices and changes to the layout")]
with open(f'{D}/areas.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['round', 'model', 'parameter', 'area', 'basis', 'noise_mult'])
    for rd, ml, mel, area, basis in A:
        mult = next((v[1] for k, v in R9.AREA.items() if v[0] == area), 1.0) if rd == 9 else 1.0
        w.writerow([rd, ml, mel, area, basis, mult])
print(len(rows), 'rows;', len(st), 'stats;', len(A), 'areas')
