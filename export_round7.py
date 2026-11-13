"""Export round-7 results (compressive spatiotemporal/CST pRF) as tidy CSVs for the
dashboard (dashboard_data/round7.csv, dashboard_data/round7_diagnostic.csv)."""
import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'dashboard_data')
D = os.path.join(ROOT, 'data_v2')

PLAB = dict(taus='Sustained time constant (tau_s)', taut='Transient time constant (tau_t)',
            ws='Sustained-channel weight (w_s)')


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.4g}' if isinstance(x, float) else x for x in r])
    print(name, len(rows), 'rows')


def main():
    rows = []

    cst = json.load(open(os.path.join(D, 'results_cst.json')))
    for key, v in cst.items():
        cond, mins = v['cond'], v['minutes']
        clab = {'oracle': 'oracle gaze', 'real': 'eye tracker'}[cond]
        design = f'cst_{cond}_{mins}m'
        dlab = f'CST, gameplay, {clab}, {mins} min'
        for prm, sc in v['scores'].items():
            rows.append(['cst', design, dlab, cond, mins, prm, PLAB[prm], sc['spearman'], sc['median_abs_err'], ''])

    seqsim = json.load(open(os.path.join(D, 'results_cst_seqsim.json')))
    for key, v in seqsim.items():
        mins = v['minutes']
        design = f'cst_seqsim_{mins}m'
        dlab = f'CST, dedicated SEQ-SIM, {mins} min'
        for prm, sc in v['scores'].items():
            rows.append(['cst', design, dlab, 'seqsim', mins, prm, PLAB[prm], sc['spearman'], sc['median_abs_err'], ''])

    write('round7.csv', ['part', 'design', 'design_label', 'condition', 'minutes', 'parameter', 'parameter_label',
                         'spearman', 'median_abs_err', 'extra'], rows)

    diag = json.load(open(os.path.join(D, 'results_cst_diagnostic.json')))
    drows = [
        ['Gameplay', diag['sustained_transient_corr_gameplay']],
        ['Dedicated SEQ-SIM', diag['sustained_transient_corr_seqsim']],
    ]
    write('round7_diagnostic.csv', ['design', 'corr'], drows)


if __name__ == '__main__':
    main()
