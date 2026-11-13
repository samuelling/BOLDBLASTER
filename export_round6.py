"""Export round-6 results (attention dynamics, ventral gaze tolerance, cue lever) as tidy CSVs
for the dashboard (dashboard_data/round6.csv, dashboard_data/round6_identify.csv)."""
import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'dashboard_data')
R = os.path.join(ROOT, 'results')

PLAB = dict(mu='pSFT peak', sig='pSFT bandwidth', c50_eff='pCRF effective c50', q='pCRF exponent',
            k='Surround suppression k', tau='Transient weight', a='Attention gain')


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.4g}' if isinstance(x, float) else x for x in r])
    print(name, len(rows), 'rows')


def main():
    rows = []

    # ---- Part A: attention-dynamics kernel fits, gameplay, oracle vs realistic gaze, 20/40/60 min
    attn = json.load(open(os.path.join(R, 'round6_attn.json')))
    ALAB = dict(a='Attention gain', tr='Rise time (tau_rise)', td='Decay time (tau_decay)')
    for key, v in attn.items():
        if key.startswith('_'):
            continue
        cond, mins = v['cond'], v['minutes']
        design = f'attn_{cond}_{mins}m'
        clab = 'oracle gaze' if cond == 'oracle' else 'eye tracker'
        dlab = f'Attention dynamics, {clab}, {mins} min'
        for p, sc in v['scores'].items():
            rows.append(['attn', design, dlab, cond, mins, p, ALAB[p], sc['spearman'], sc['median_abs_err'], ''])

    # ---- Part B: ventral population, oracle vs realistic (eye-tracker) gaze, 10/20/40 min
    vent = json.load(open(os.path.join(R, 'round6_vent.json')))
    for key, v in vent.items():
        if key.startswith('_'):
            continue
        cond, mins = v['cond'], v['minutes']
        design = f'vent_{cond}_{mins}m'
        clab = 'oracle gaze+pRF' if cond == 'oracle' else 'eye tracker'
        dlab = f'Ventral population, {clab}, {mins} min'
        sc = v['scores']
        rows.append(['vent', design, dlab, cond, mins, 'pref_weight', 'Preferred-category weight',
                     sc['pref_weight']['spearman'], sc['pref_weight']['median_abs_err'], ''])
        rows.append(['vent', design, dlab, cond, mins, 'full_vector_corr', 'Full weight-vector correlation (median)',
                     sc['full_vector_corr_median'], '', ''])
        rows.append(['vent', design, dlab, cond, mins, 'pref_category_frac', 'Fraction with correct preferred category',
                     sc['pref_category_recovered_frac'], '', ''])

    # ---- V1-vs-ventral matched comparison at 40 minutes (fresh V1 fit for fair comparison)
    v1m = json.load(open(os.path.join(R, 'round6_v1_match40.json')))
    for key, cond in (('game_40m', 'realistic'), ('game_40m_oracle', 'oracle')):
        v = v1m[key]
        dlab = f"V1, matched 40 min game ({'oracle gaze+pRF' if cond == 'oracle' else 'eye tracker'})"
        for p, sc in v['scores'].items():
            if p not in PLAB:
                continue
            rows.append(['v1match40', key, dlab, cond, 40, p, PLAB[p], sc['spearman'], sc['median_abs_err'], ''])

    # ---- Part C: tightened covert-cue lever vs the original round-3 cue schedule (from recovery.csv)
    lever = json.load(open(os.path.join(R, 'round6_lever.json')))['lever_cue_1h']
    for p, sc in lever['scores'].items():
        if p not in PLAB:
            continue
        rows.append(['lever', 'lever_cue_1h', 'Game + tightened cue (3-5 s), 1 h', 'tightened_cue', 60, p, PLAB[p],
                     sc['spearman'], sc['median_abs_err'], ''])
    with open(os.path.join(OUT, 'recovery.csv')) as f:
        for r in csv.DictReader(f):
            if r['design'] == 'levers_1h':
                rows.append(['lever', 'levers_1h', r['design_label'] + ' (original cue 6-10 s)', 'original_cue', 60,
                             r['parameter'], r['parameter_label'], float(r['spearman']), float(r['median_abs_err']), ''])

    write('round6.csv', ['part', 'design', 'design_label', 'condition', 'minutes', 'parameter', 'parameter_label',
                         'spearman', 'median_abs_err', 'extra'], rows)

    # ---- Part A diagnostic: analytic non-identifiability of decay time in the dedicated block design
    ident = json.load(open(os.path.join(R, 'round6_attn_identify.json')))
    idrows = list(zip(ident['td_grid'], ident['corr']))
    write('round6_identify.csv', ['td', 'corr'], idrows)


if __name__ == '__main__':
    main()
