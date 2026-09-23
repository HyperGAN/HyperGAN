#!/usr/bin/env python3
"""Compare finite warmup trajectories with their exact initialized baseline."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'research/startup_tuning/results/2026-09-22-healthy-control/logos/report.json'


def rows_for(case, report):
    real = report['real_bank_output_stats']
    schedule = {row['round']: row for row in report.get('proposal', {}).get('rounds', [])}
    for observation in report.get('observations', []):
        step = observation['step']
        if step > 128:
            continue
        counts = schedule.get(step, {'g_updates': step, 'd_updates': step, 'prior_updates': step})
        for bank in ('fixed_latent', 'evolving_prior'):
            measured = observation[bank]
            value = measured['output']
            activation = measured['final_affine']['activation']
            yield {'case': case, 'round': step, 'g_updates': counts['g_updates'],
                   'd_updates': counts['d_updates'], 'prior_updates': counts['prior_updates'],
                   'bank': bank, 'saturation_percent': 100 * value['absolute_above_0_99_fraction'],
                   'pixel_diversity_percent_of_real': 100 * value['sample_diversity_rms'] / real['sample_diversity_rms'],
                   'spatial_diversity_percent_of_real': 100 * value['spatial_sample_diversity_rms'] / real['spatial_sample_diversity_rms'],
                   'pooled_diversity_percent_of_real': 100 * value['pooled_4x4_sample_diversity_rms'] / real['pooled_4x4_sample_diversity_rms'],
                   'pre_tanh_rms': activation['rms'],
                   'mean_tanh_derivative': activation['tanh_response']['mean_derivative']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='case/report.json paths')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text())
    rows = list(rows_for('baseline', baseline))
    for path in args.reports:
        report = json.loads(path.read_text())
        if 'evaluation' not in report:
            print(path.parent.name, report['status'])
            continue
        assert report['initial_parameters_sha256'] == baseline['initial_parameters_sha256']
        assert report['prepared_parameters_sha256'] == baseline['prepared_parameters_sha256']
        for key in ('bank_sha256', 'measurement_rng_sha256', 'prior_rng_sha256'):
            assert report['evaluation'][key] == baseline['evaluation'][key], key
        if report['status'] == 'complete':
            assert report['restored'] and report['source_config_unchanged']
            assert (report['protected_before_sha256'] == report['protected_after_sha256'] ==
                    report['protected_after_restore_sha256'])
        rows.extend(rows_for(path.parent.name, report))
    print('case round G/D sat% pixel-div% spatial-div% pooled-div% pre-tanh-RMS')
    for row in rows:
        if row['bank'] == 'evolving_prior':
            print(f"{row['case']} {row['round']} {row['g_updates']}/{row['d_updates']} "
                  + ' '.join(f'{row[key]:.6g}' for key in (
                      'saturation_percent', 'pixel_diversity_percent_of_real',
                      'spatial_diversity_percent_of_real', 'pooled_diversity_percent_of_real', 'pre_tanh_rms')))
    if args.output:
        with args.output.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
