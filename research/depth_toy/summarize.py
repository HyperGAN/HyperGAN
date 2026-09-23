"""Render a compact comparison from completed depth-toy reports."""
import argparse
import json
from pathlib import Path


def summarize(root):
    cases = [json.loads(path.read_text()) for path in root.glob('depth-*/report.json')]
    cases.sort(key=lambda case: case['depth'])
    if not cases:
        raise ValueError('no completed depth reports')
    for key in ('critic', 'prior', 'shared_generator', 'monitor'):
        if len({case['identity'][key] for case in cases}) != 1:
            raise ValueError(f'unmatched {key}')
    final_steps = {case['records'][-1]['step'] for case in cases}
    if len(final_steps) != 1:
        raise ValueError('final horizons differ')
    lines = ['# 100gaussians generator-depth comparison', '',
             f"Final step: {final_steps.pop()}. One seed, CPU, matched initial D/prior/shared G tensors and monitor bank.", '',
             '| Hidden layers | Seconds incl. diagnostics | Online modes | Online HQ | EMA modes | EMA HQ | EMA sliced W1 |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in cases:
        row = case['records'][-1]
        online, ema = row['online'], row['ema']
        lines.append(f"| {case['depth']} | {case['seconds']:.1f} | {online['modes']} | {online['hq']:.2%} | {ema['modes']} | {ema['hq']:.2%} | {ema['sliced_w1']:.4f} |")
    lines += ['', '## Initial contraction and first actual G update', '',
              '| Layers | Initial output spread | Requested shared energy | Actual shared energy | Movement cosine | Movement gain |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in cases:
        start, first = case['records'][:2]
        response = first['update']['g_only']
        lines.append(f"| {case['depth']} | {start['online']['between_sample_rms']:.6g} | {response['requested']['shared_energy_fraction']:.2%} | {response['actual']['shared_energy_fraction']:.2%} | {response['cosine']:.4f} | {response['gain']:.4g} |")
    lines += ['', 'Shared energy is a fraction, not absolute shared movement. Gain includes the',
              'native batch-mean gradient convention. A single first-step response is not a',
              'causal diagnosis. HQ/coverage do not validate within-mode variance.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output:
        args.output.write_text(result)
    else:
        print(result)
