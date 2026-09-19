"""Bounded immutable EMA previews, separate from deployable inference bundles."""
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .run_state import atomic_json, sync_directory

MAX_COUNT = 16
MAX_ELEMENTS = 65536
MAX_BYTES = 2 * 1024 * 1024
MAX_KEEP = 100
_GENERATION = re.compile(r'(?:\.pending-)?\d{12,}-\d{4,}-[0-9a-f]{32}-step\d{8,}-[0-9a-f]{32}')


def _inputs(trainer, batch):
    names, needed = set(), set()
    def include(name):
        if name in names:
            return
        names.add(name)
        for binding in trainer.config['components'][name]['inputs'].values():
            if binding.startswith('components.'):
                include(binding.split('.')[1])
            elif binding.startswith('batch.'):
                needed.add(binding.split('.')[1])
    include('generator')
    return {key: batch[key] for key in needed}


def render_preview(trainer, batch, identity):
    """Copy EMA state before eval: custom forwards cannot mutate live buffers."""
    import copy
    import random
    import numpy as np
    import torch
    from .checkpoints import capture_rng, restore_rng
    rng = capture_rng()
    try:
        seed = trainer.config['sampling']['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        inputs = _inputs(trainer, batch)
        real = batch['real']
        per_sample = real[0].numel()
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or not len(value):
                raise ValueError(f'Preview input {key} must be a nonempty batched tensor')
            per_sample += value[0].numel()
        count = min(trainer.config['sampling']['count'], MAX_COUNT, MAX_ELEMENTS // max(1, per_sample))
        if count < 1:
            raise ValueError(f'One preview sample exceeds the {MAX_ELEMENTS}-element output/input budget')
        normalized = {key: value[torch.arange(count) % len(value)].detach().clone() for key, value in inputs.items()}
        recorded_inputs = {key: value.clone() for key, value in normalized.items()}
        graph = copy.deepcopy(trainer.ema_graph).eval().requires_grad_(False)
        prior = copy.deepcopy(trainer.ema_prior).eval().requires_grad_(False)
        with torch.inference_mode():
            latent, ids = prior.sample(count, generator=torch.Generator().manual_seed(seed))
            values = graph.generate(latent, normalized)['generated']
        if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
            raise ValueError('Preview generator must return a tensor with the requested batch size')
        if values.numel() + sum(value.numel() for value in recorded_inputs.values()) > MAX_ELEMENTS:
            raise ValueError(f'Preview exceeds the {MAX_ELEMENTS}-element output/input budget')
        if not torch.isfinite(values).all():
            raise ValueError('Preview contains nonfinite values')
        return {'schema_version': 1, 'kind': 'ema-preview', 'identity': dict(identity),
                'step': trainer.step, 'seed': seed, 'count': count,
                'requested_count': trainer.config['sampling']['count'],
                'count_limited_by': [name for name, limit in [('count-cap', MAX_COUNT), ('element-budget', MAX_ELEMENTS // max(1, per_sample))] if count < trainer.config['sampling']['count'] and count == limit],
                'shape': list(values.shape),
                'samples': values.tolist(), 'particle_ids': ids.tolist() if ids is not None else None,
                'inputs': {key: value.tolist() for key, value in recorded_inputs.items()},
                'conditioning': 'last-completed-batch-cycled' if normalized else 'unconditional',
                'resume_supported': False}
    finally:
        restore_rng(rng)


def _write_bounded(path, payload):
    size = 0
    with path.open('xb') as output:
        for chunk in json.JSONEncoder(allow_nan=False, separators=(',', ':')).iterencode(payload):
            encoded = chunk.encode('utf-8')
            size += len(encoded)
            if size + 1 > MAX_BYTES:
                raise ValueError(f'Preview JSON exceeds the {MAX_BYTES}-byte budget')
            output.write(encoded)
        output.write(b'\n')
        output.flush()
        os.fsync(output.fileno())
    return size + 1


def publish_preview(run_dir, trainer, batch, identity, keep=3):
    return _publish_preview(run_dir, identity, trainer.step,
                            lambda: render_preview(trainer, batch, identity), keep)


def publish_preview_payload(run_dir, payload, identity, step, keep=3):
    """Publish an already-rendered bounded JSON preview without numerical imports."""
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1
            or payload.get('kind') != 'ema-preview' or type(payload.get('step')) is not int
            or payload['step'] != step
            or json.dumps(payload.get('identity'), sort_keys=True) != json.dumps(identity, sort_keys=True)
            or type(payload.get('count')) is not int or not 1 <= payload['count'] <= MAX_COUNT
            or type(payload.get('shape')) is not list or not payload['shape']
            or payload['shape'][0] != payload['count']):
        raise ValueError('Rendered preview identity, step or shape is invalid')
    return _publish_preview(run_dir, identity, step, lambda: payload, keep)


def _publish_preview(run_dir, identity, step, render, keep):
    """Publish a complete directory, update its bounded index, then prune old previews.

    The trainer's run lock serializes producers. Only this managed preview directory
    is pruned; final inference bundles and recovery checkpoints are never removed.
    """
    if type(keep) is not int or not 1 <= keep <= MAX_KEEP:
        raise ValueError(f'preview_keep must be between 1 and {MAX_KEEP}')
    root = Path(run_dir) / 'previews'
    if root.is_symlink():
        raise ValueError('Managed preview directory must not be a symlink')
    root.mkdir(exist_ok=True)
    sync_directory(root.parent)
    # No other producer can own a pending directory while the trainer run lock is held.
    for pending in root.iterdir():
        if pending.name.startswith('.pending-') and _GENERATION.fullmatch(pending.name) and pending.is_dir() and not pending.is_symlink():
            shutil.rmtree(pending)
    name = f"{identity['sample_sequence']:012d}-{identity['attempt_id']}-step{step:08d}-{uuid.uuid4().hex}"
    temporary, target = root / ('.pending-' + name), root / name
    temporary.mkdir()
    completed = False
    try:
        payload = render()
        size = _write_bounded(temporary / 'preview.json', payload)
        record = {'schema_version': 1, 'kind': 'ema-preview', 'identity': dict(identity),
                  'step': step, 'path': str(target / 'preview.json'), 'bytes': size,
                  'count': payload['count'], 'shape': payload['shape']}
        atomic_json(temporary / 'manifest.json', record)
        sync_directory(temporary)
        temporary.rename(target)
        sync_directory(root)
        # Scan only managed generation directories. This also recovers publication
        # interrupted after rename but before index update in a prior attempt.
        records = []
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink() and _GENERATION.fullmatch(entry.name) and (entry / 'manifest.json').is_file():
                saved = json.loads((entry / 'manifest.json').read_text(encoding='utf-8'))
                if saved.get('kind') == 'ema-preview' and saved.get('identity', {}).get('run_id') == identity['run_id']:
                    saved['path'] = str(entry / 'preview.json')
                    records.append((saved, entry))
        records.sort(key=lambda pair: pair[0]['identity']['sample_sequence'])
        retained, expired = records[-keep:], records[:-keep]
        index = {'schema_version': 1, 'kind': 'ema-preview-index', 'run_id': identity['run_id'],
                 'keep': keep, 'previews': [saved for saved, _ in retained]}
        try:
            atomic_json(root / 'index.json', index)
        except Exception:
            # If index publication failed before replacement, do not accumulate an
            # unindexed generation on every failed observation. After replacement,
            # preserve it: readers may already have observed the new index.
            try:
                visible = json.loads((root / 'index.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                visible = None
            if visible != index:
                shutil.rmtree(target)
                sync_directory(root)
            raise
        errors = []
        for _, entry in expired:
            try:
                shutil.rmtree(entry)
            except OSError as exc:
                errors.append(f'Could not prune preview {entry.name}: {exc}')
        sync_directory(root)
        completed = True
        return record, index, errors
    finally:
        if not completed and target.exists():
            try:
                visible = json.loads((root / 'index.json').read_text(encoding='utf-8'))
                referenced = any(item.get('path') == str(target / 'preview.json') for item in visible.get('previews', []))
            except (OSError, ValueError, AttributeError, TypeError):
                referenced = False
            if not referenced:
                shutil.rmtree(target)
                sync_directory(root)
        if temporary.exists():
            shutil.rmtree(temporary)
