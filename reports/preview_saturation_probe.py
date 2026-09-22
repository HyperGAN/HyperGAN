"""Read-only measurements of saved EMA preview PNGs, without model inference.

PNG quantization/clipping and EMA lag limit interpretation. These are output
statistics, not online gradient measurements or estimates of image quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def statistics(path, metadata, shape):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['sha256']:
        raise ValueError(f'Preview image checksum mismatch: {path}')
    count, channels, height, width = shape
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB'), dtype=np.float64) / 127.5 - 1
    if channels != 3 or pixels.shape != (metadata['height'], metadata['width'], 3):
        raise ValueError('Unexpected saved preview shape')
    rows, columns = pixels.shape[0] // height, pixels.shape[1] // width
    samples = pixels.reshape(rows, height, columns, width, channels).transpose(0, 2, 1, 3, 4)
    samples = samples.reshape(-1, height, width, channels)[:count]
    if len(samples) != count:
        raise ValueError('Preview grid has fewer cells than declared samples')
    colors = samples.mean((1, 2), keepdims=True)
    total_variance = samples.var(axis=0).mean()
    color_variance = colors.var(axis=0).mean()
    residual_variance = (samples - colors).var(axis=0).mean()
    if not np.isclose(total_variance, color_variance + residual_variance, atol=1e-12):
        raise ValueError('Between-sample variance decomposition failed')
    return {
        'absolute_value_at_least_0_98_fraction': float((np.abs(samples) >= .98).mean()),
        'absolute_value_above_0_99_fraction': float((np.abs(samples) > .99).mean()),
        'sample_diversity_rms': float(np.sqrt(total_variance)),
        'spatial_per_channel_std_rms': float(np.sqrt(samples.var(axis=(1, 2)).mean())),
        'mean_color_share_of_between_sample_variance': float(color_variance / total_variance) if total_variance else None,
        'mean_color_removed_sample_diversity_rms': float(np.sqrt(residual_variance)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    events = {}
    for line in (args.run / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:  # A live writer may have an incomplete last line.
            continue
        if event.get('event') == 'train' and event.get('metrics'):
            events[event['step']] = event
    rows = []
    for path in sorted((args.run / 'previews').glob('*/preview.json')):
        preview = json.loads(path.read_text())
        if preview['identity']['run_id'] != manifest['run_id']:
            raise ValueError('Preview belongs to a different run')
        event = events.get(preview['step'], {})
        row = {'step': preview['step'], 'preview_manifest': str(path),
               'preview_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'preview_seed': preview['seed'], 'particle_ids': preview.get('particle_ids'),
               'g_lr_warmup': event.get('g_lr_warmup'),
               'training_metrics': event.get('metrics', {})}
        for name in ('image_grid', 'real_image_grid'):
            grid = preview.get(name)
            if grid is not None:
                row[name] = statistics(Path(grid['path']), grid, preview['shape'])
        rows.append(row)
        print(json.dumps({'step': row['step'], **row['image_grid']}), flush=True)
    report = {'run_dir': str(args.run.resolve()), 'run_id': manifest['run_id'],
              'observed_status': manifest['status'], 'observed_step': manifest['steps'],
              'tuning': manifest['initialization_tuning'], 'rows': rows,
              'interpretation': [
                  'Quantized saved EMA previews; not online generator tensors or gradients.',
                  'Fractions count scalar color-channel values, not whole RGB pixels.',
                  'Population variances decompose exactly into sample mean-color and spatial-residual terms.',
                  'Preview particles/seed are recorded; the learned prior and EMA evolve, and real batches differ.',
                  'Large output spread does not imply useful or semantic diversity.',
                  'A single changing-rate run cannot establish a causal rate threshold or validate a fixed-rate alternative.']}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
