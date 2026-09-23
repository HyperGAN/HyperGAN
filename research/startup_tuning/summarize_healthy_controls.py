#!/usr/bin/env python3
"""Export real-referenced output metrics and batch-mean stage energy."""
import argparse
import csv
import json
import math
from pathlib import Path

CASES = ('cifar', 'logos', 'cifar_replicated', 'cifar_replicated_compensated', 'logos32_data')
METRICS = ('sample_diversity_rms', 'spatial_sample_diversity_rms',
           'sample_mean_color_diversity_rms', 'pooled_4x4_sample_diversity_rms')


def summarize(root):
    outputs, stages = [], []
    for case in CASES:
        report = json.loads((root / case / 'report.json').read_text())
        assert report['status'] == 'complete', case
        assert report['restored'] and report['source_config_unchanged'], case
        assert (report['protected_before_sha256'] == report['protected_after_sha256'] ==
                report['protected_after_restore_sha256']), case
        real = report['real_bank_output_stats']
        for observation in report['observations']:
            for bank in ('fixed_latent', 'evolving_prior'):
                measured = observation[bank]
                value = measured['output']
                row = {'case': case, 'step': observation['step'], 'bank': bank,
                       'saturation_percent': 100 * value['absolute_above_0_99_fraction'],
                       'pre_tanh_rms': measured['final_affine']['activation']['rms'],
                       'mean_tanh_derivative': measured['final_affine']['activation']['tanh_response']['mean_derivative']}
                for key in METRICS:
                    row[key] = value[key]
                    row[key + '_relative_to_real'] = value[key] / real[key]
                outputs.append(row)
                for path, value in measured['stages'].items():
                    rms, diversity = value['rms'], value['sample_diversity_rms']
                    stages.append({'case': case, 'step': observation['step'], 'bank': bank,
                                   'stage': path.removeprefix('models.generator.network.nodes.n_'),
                                   'rms': rms, 'sample_diversity_rms': diversity,
                                   'batch_mean_rms': math.sqrt(max(0., rms * rms - diversity * diversity)),
                                   'diversity_to_rms': value['diversity_to_rms']})
    return outputs, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outputs, stages = summarize(args.reports_root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('outputs.csv', outputs), ('stages.csv', stages)):
        with (args.output / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'healthy-controls-v1'
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    labels = ('CIFAR source', 'Logos128 source', 'CIFAR replicated FFNs',
              'CIFAR replicated + compensated Adam', 'CIFAR model on logos32 data')
    colors = ('#238b45', '#252525', '#d95f02', '#2b6cb0', '#8e44ad')
    for case, label, color in zip(CASES, labels, colors):
        rows = [r for r in outputs if r['case'] == case and r['bank'] == 'evolving_prior']
        for ax, metric, scale in zip(axes, ('saturation_percent', 'sample_diversity_rms_relative_to_real',
                                          'pooled_4x4_sample_diversity_rms_relative_to_real'), (1, 100, 100)):
            ax.plot([r['step'] for r in rows], [scale * r[metric] for r in rows], marker='o',
                    markersize=3, label=label, color=color)
    for ax, title in zip(axes, ('Saturated RGB values (%)', 'Pixel diversity (% of real)',
                               '4×4 pooled diversity (% of real)')):
        ax.set_title(title)
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlabel('Native training updates (symlog scale)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 105)
    for ax in axes[1:]:
        ax.axhline(100, color='gray', linestyle=':', linewidth=1)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Fixed 64-image monitor bank, online generator; diversity is not a quality score')
    destination = args.output / 'comparison.svg'
    fig.savefig(destination, metadata={'Date': None})
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
