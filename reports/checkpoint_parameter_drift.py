"""Read-only cumulative parameter displacement for the multidepth DINO testbed.

Group names recognize this recipe's HNDL paths, not arbitrary pretrained models.
No model reconstruction, optimizer update, or write to the run is performed.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from hypergan.checkpoints import read_checkpoint


def category(owner, name):
    if owner == 'prior':
        return 'prior'
    if name.startswith('models.generator.'):
        return 'generator'
    if '.n_backbone.' in name:
        return 'pretrained_backbone'
    if name.startswith('models.discriminator.'):
        return ('discriminator_feature_heads' if '.n_feature' in name
                else 'discriminator_pixel_and_other')
    return 'other'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifests = [p for p in (args.run / 'checkpoints').glob('*/manifest.json')
                 if not p.parent.name.startswith('.')]
    paths = sorted(manifests, key=lambda p: json.loads(p.read_text())['step'])
    if not paths:
        raise ValueError('No complete checkpoints')
    _, anchor, baseline = read_checkpoint(args.run, paths[0].parent.name)
    if baseline['step'] != 0:
        raise ValueError('Needs step zero baseline')
    records = []
    for path in paths[1:]:
        _, info, state = read_checkpoint(args.run, path.parent.name)
        if (info['run_id'], info['config_sha256']) != (anchor['run_id'], anchor['config_sha256']):
            raise ValueError('Run/config identity differs from anchor')
        groups, rows = {}, []
        for owner in ('graph', 'prior'):
            if state['trainable'][owner] != baseline['trainable'][owner]:
                raise ValueError('Parameter inventory or trainability changed')
            for name in baseline['trainable'][owner]:
                a, b = baseline[owner][name], state[owner][name]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError('Parameter shape/dtype changed')
                difference = b.double() - a.double()
                delta_square = float(difference.square().sum())
                initial_square = float(a.double().square().sum())
                group = category(owner, name)
                item = groups.setdefault(group, {
                    'count': 0, 'delta_square_sum': 0., 'initial_square_sum': 0.,
                    'changed_tensors': 0, 'tensors': 0})
                item['count'] += a.numel()
                item['delta_square_sum'] += delta_square
                item['initial_square_sum'] += initial_square
                item['changed_tensors'] += int(not torch.equal(a, b))
                item['tensors'] += 1
                if group != 'pretrained_backbone':
                    rows.append({'path': owner + '.' + name,
                                 'relative_l2_change': math.sqrt(delta_square / initial_square) if initial_square else None,
                                 'delta_rms': math.sqrt(delta_square / a.numel()),
                                 'initial_rms': math.sqrt(initial_square / a.numel())})
        for item in groups.values():
            item['relative_l2_change'] = (math.sqrt(item['delta_square_sum'] / item['initial_square_sum'])
                                          if item['initial_square_sum'] else None)
            item['delta_rms'] = math.sqrt(item['delta_square_sum'] / item['count'])
        records.append({'step': state['step'], 'checkpoint': path.parent.name,
                        'groups': groups, 'parameters': rows})
        print(json.dumps({'step': state['step'], 'groups': groups}), flush=True)
        del state
    report = {'baseline_step': 0,
              'interpretation': 'Cumulative parameter displacement, not one optimizer update or functional change. Backbone parameter equality is reported; buffers are outside this measurement.',
              'records': records}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
