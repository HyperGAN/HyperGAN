"""Bounded immutable EMA previews, separate from deployable inference bundles."""
import atexit
import json
import base64
import hashlib
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import uuid

from .run_state import atomic_json, sync_directory

MAX_COUNT = 16
MAX_ELEMENTS = 65536
MAX_BYTES = 2 * 1024 * 1024
# PNG transport is bounded separately; published JSON never contains image bytes
# or large nested pixel lists. The total float image budget is 16 MiB.
MAX_IMAGE_ELEMENTS = 4_194_304
MAX_RENDER_BYTES = 24 * 1024 * 1024
MAX_INPUT_GRIDS = 4
MAX_EXTRA_GRIDS = 4
# Image history is what a reader scrubs through, so a run keeps a bounded but
# whole-run history rather than only its newest samples. When a run outgrows
# `keep`, the spacing between the older samples doubles (a run publishing every
# 500 steps is thinned to every 1,000, then every 2,000), which drops about half
# of them in one chunk and postpones the next prune by many publications. The
# first sample of the run and the newest window are never thinned, so the
# viewer's slider always reaches the beginning of the run. `KEEP_ALL` opts out
# and keeps every published generation for the life of the run.
KEEP_ALL = 0
DEFAULT_KEEP = 128
# The newest samples stay at full density inside a window that advances in whole
# chunks of this many sequences, so a sample leaving the dense window is decided
# together with its neighbours instead of one generation per publication.
DENSE_WINDOW = 16
# Short stable sample names: the EMA generator output is 'g' and the real batch
# it is compared against is 'x'. Names index a source across steps; they are not
# unique artifact identities.
DEFAULT_NAME = 'g'
REAL_NAME = 'x'
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,15}')
_GENERATION = re.compile(r'(?:\.(?:pending|expired)-)?\d{12,}-\d{4,}-[0-9a-f]{32}-step\d{8,}-[0-9a-f]{32}')


def sample_name(value, default=DEFAULT_NAME):
    """Validate a short stable sample name, falling back to the default."""
    if value is None:
        return default
    if not isinstance(value, str) or NAME.fullmatch(value) is None:
        raise ValueError('Sample name must be 1-16 letters, digits, dots, colons, underscores or hyphens')
    return value


def _identity_name(identity, payload=None):
    named = (payload or {}).get('name')
    if named is None:
        named = identity.get('name') if isinstance(identity, dict) else None
    return sample_name(named)


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
    from .config import sampling_bindings
    for binding in sampling_bindings(trainer.config['sampling'], preview=True):
        if binding.startswith('components.'):
            include(binding.split('.')[1])
        elif binding.startswith('batch.'):
            needed.add(binding.split('.')[1])
    return {key: batch[key] for key in needed}


def _conditioned(binding, specs):
    """Whether a particular output depends on batch input, independent of other views."""
    if binding == 'generated':
        binding = 'components.generator'
    if binding.startswith('batch.'):
        return True
    if binding.startswith('components.'):
        return any(_conditioned(path, specs) for path in specs[binding.split('.')[1]]['inputs'].values())
    return False


def _image(value):
    return value.ndim == 4 and value.shape[1] in (1, 3) and value.is_floating_point()


def preview_budget(trainer, batch, inputs):
    """Keep tensor fixtures small; allow bounded images without tensor JSON."""
    import torch
    real = batch['real']
    for name, value in dict(inputs, real=real).items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or not len(value):
            raise ValueError(f'Preview input {name} must be a nonempty batched tensor')
    per_sample = real[0].numel() + sum(value[0].numel() for value in inputs.values())
    image_only = _image(real) and per_sample * min(trainer.config['sampling']['count'], MAX_COUNT) > MAX_ELEMENTS
    budget = MAX_IMAGE_ELEMENTS if image_only else MAX_ELEMENTS
    count = min(trainer.config['sampling']['count'], MAX_COUNT, budget // max(1, per_sample))
    comparison = trainer.config['sampling'].get('comparison', [])
    if comparison and _image(real):
        from .image_grids import MAX_SIDE, MAX_PIXELS
        height, width = real.shape[2:]
        columns = len(comparison)
        count = min(count, (MAX_SIDE - 24) // height,
                    (MAX_PIXELS // (width * columns) - 24) // height)
    if count < 1:
        raise ValueError(f'One preview sample exceeds the {budget}-element output/input budget')
    if image_only and sum(count * value[0].numel() for value in inputs.values() if not _image(value)) > MAX_ELEMENTS:
        raise ValueError('Nonimage preview inputs exceed the tensor element budget')
    return count, per_sample, budget, image_only


def render_preview(trainer, batch, identity):
    """Copy EMA state before eval: custom forwards cannot mutate live buffers."""
    import copy
    import random
    import numpy as np
    import torch
    from .checkpoints import capture_rng, restore_rng
    from .recipes import generation_output, generation_particle_ids
    rng = capture_rng()
    try:
        seed = trainer.config['sampling']['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        inputs = _inputs(trainer, batch)
        real = batch['real']
        count, per_sample, budget, image_only = preview_budget(trainer, batch, inputs)
        normalized = {key: value[torch.arange(count) % len(value)].detach().clone() for key, value in inputs.items()}
        recorded_inputs = {key: value.clone() for key, value in normalized.items()}
        graph = copy.deepcopy(trainer.ema_graph).cpu().eval().requires_grad_(False)
        prior = copy.deepcopy(trainer.ema_prior).cpu().eval().requires_grad_(False)
        normalized = {key: value.cpu() for key, value in normalized.items()}
        recorded_inputs = {key: value.cpu() for key, value in recorded_inputs.items()}
        with torch.inference_mode():
            latent, ids = prior.sample(count, generator=torch.Generator().manual_seed(seed))
            context = graph.generate(latent, normalized, prior=prior)
            values = generation_output(graph, context, trainer.config['sampling'])
            ids = generation_particle_ids(graph, context, ids, trainer.config['sampling'])
            views = {name: graph.resolve(binding, context) for name, binding
                     in trainer.config['sampling'].get('views', {}).items()}
            comparison = [(column['label'], graph.resolve(column['binding'], context))
                          for column in trainer.config['sampling'].get('comparison', [])]
        if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
            raise ValueError('Preview generator must return a tensor with the requested batch size')
        if values.numel() + sum(value.numel() for value in recorded_inputs.values()) > budget:
            raise ValueError(f'Preview exceeds the {budget}-element output/input budget')
        if image_only and not _image(values):
            raise ValueError('Large previews require floating NCHW RGB/grayscale generator output')
        if not torch.isfinite(values).all():
            raise ValueError('Preview contains nonfinite values')
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': dict(identity),
                'name': _identity_name(identity),
                'step': trainer.step, 'seed': seed, 'count': count,
                'requested_count': trainer.config['sampling']['count'],
                'count_limited_by': [name for name, limit in [('count-cap', MAX_COUNT), ('element-budget', budget // max(1, per_sample))] if count < trainer.config['sampling']['count'] and count == limit],
                'shape': list(values.shape),
                'samples': None if image_only else values.tolist(), 'particle_ids': ids.tolist() if ids is not None else None,
                'representation': 'png' if image_only else 'tensor',
                'inputs': {key: {'shape': list(value.shape), 'representation': 'png'}
                           if image_only and _image(value) else value.tolist()
                           for key, value in recorded_inputs.items()},
                'conditioning': ('last-completed-batch-cycled' if _conditioned(
                    trainer.config['sampling'].get('generated', 'generated'), trainer.config['components']) else 'unconditional'),
                'resume_supported': False}
        if ids is not None:
            _, frequencies = ids.unique(return_counts=True)
            payload['routing'] = {'unique_particles': len(frequencies),
                                  'top_particle_share': float(frequencies.max().item() / len(ids))}
        from .metrics import preview_metrics_enabled
        if preview_metrics_enabled(trainer.config):
            from .diversity_metrics import batch_diversity
            # Unlike the display grid, never cycle a short real batch: repeated
            # rows would bias the unbiased sample variance and hide N=1.
            reference = real[:count].detach().cpu()
            diversity = batch_diversity(values, reference)
            coarse = batch_diversity(values, reference, pool_size=4)
            for field in ('metrics', 'unavailable'):
                diversity[field].update({'pooled4_' + key: value for key, value in coarse[field].items()})
            payload['diversity'] = dict(diversity, schema_version=1,
                generated_count=count, reference_count=len(reference),
                generated_binding=trainer.config['sampling'].get('generated', 'generated'))
        if (views or comparison) and not _image(values):
            raise ValueError('Additional preview views and comparisons require image output')
        if payload['name'] in views or (comparison and payload['name'] == 'comparison'):
            raise ValueError('Preview name collides with an additional view or comparison')
        if values.ndim == 4 and values.shape[1] in (1, 3):
            from .image_grids import tensor_grid
            provenance = {key: payload[key] for key in (
                'identity', 'step', 'seed', 'count', 'shape', 'particle_ids', 'conditioning')}
            encoded, grid = tensor_grid(values, dict(provenance, name=payload['name']))
            payload['image_grid'] = dict(grid, name=payload['name'],
                                         png_base64=base64.b64encode(encoded).decode('ascii'))
            # The comparable real batch is published beside it under its own name
            # so the viewer can index both sources; it is not regenerated data.
            rows = real[torch.arange(count) % len(real)].detach().cpu()
            if (rows.ndim == 4 and rows.shape[1] == values.shape[1] and rows.is_floating_point()
                    and rows.numel() <= budget and torch.isfinite(rows).all()):
                encoded, grid = tensor_grid(rows, dict(provenance, name=REAL_NAME,
                                                       shape=list(rows.shape), source='batch.real'))
                payload['real_image_grid'] = dict(grid, name=REAL_NAME, source='batch.real',
                                                  png_base64=base64.b64encode(encoded).decode('ascii'))
            image_inputs = [(key, value) for key, value in sorted(recorded_inputs.items())
                            if key != 'real' and _image(value)]
            if len(image_inputs) > MAX_INPUT_GRIDS:
                raise ValueError(f'Preview supports at most {MAX_INPUT_GRIDS} conditioning image grids')
            # Input names may equal the generator/real shelf names. Preserve the
            # original batch source, but give every image a distinct shelf name.
            occupied = {payload['name'], REAL_NAME, 'comparison', *views}
            reserved = occupied | {key for key, _ in image_inputs}
            for index, (key, value) in enumerate(image_inputs):
                name = key
                if NAME.fullmatch(name) is None or name in occupied:
                    suffix = index
                    name = f'input:{suffix}'
                    while name in reserved:
                        suffix += 1
                        name = f'input:{suffix}'
                occupied.add(name)
                reserved.add(name)
                encoded, grid = tensor_grid(value, dict(provenance, name=name,
                    shape=list(value.shape), source='batch.' + key))
                payload[f'input_image_grid_{index}'] = dict(grid, name=name, source='batch.' + key,
                    shape=list(value.shape), png_base64=base64.b64encode(encoded).decode('ascii'))
            for index, (name, value) in enumerate(views.items()):
                if (not isinstance(value, torch.Tensor) or not _image(value)
                        or len(value) != count or value.numel() > MAX_IMAGE_ELEMENTS):
                    raise ValueError('Additional preview views must be bounded image batches')
                encoded, grid = tensor_grid(value, dict(provenance, name=name,
                    particle_ids=None, source=trainer.config['sampling']['views'][name],
                    conditioning=('last-completed-batch-cycled' if _conditioned(
                        trainer.config['sampling']['views'][name], trainer.config['components']) else 'unconditional')))
                payload[f'extra_image_grid_{index}'] = dict(grid, name=name,
                    source=trainer.config['sampling']['views'][name],
                    png_base64=base64.b64encode(encoded).decode('ascii'))
            if comparison:
                if any(not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != count
                       for _, value in comparison):
                    raise ValueError('Comparison columns must match the requested preview count')
                from .image_grids import comparison_grid
                encoded, grid = comparison_grid(comparison, dict(provenance, name='comparison'))
                payload['comparison_image_grid'] = dict(grid, name='comparison',
                    sources=trainer.config['sampling']['comparison'],
                    png_base64=base64.b64encode(encoded).decode('ascii'))
        return payload
    finally:
        restore_rng(rng)


def _write_bounded(path, payload, max_bytes=None):
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    size = 0
    with path.open('xb') as output:
        for chunk in json.JSONEncoder(allow_nan=False, separators=(',', ':')).iterencode(payload):
            encoded = chunk.encode('utf-8')
            size += len(encoded)
            if size + 1 > max_bytes:
                raise ValueError(f'Preview JSON exceeds the {max_bytes}-byte budget')
            output.write(encoded)
        output.write(b'\n')
        output.flush()
        os.fsync(output.fileno())
    return size + 1


def publish_preview(run_dir, trainer, batch, identity, keep=DEFAULT_KEEP):
    return _publish_preview(run_dir, identity, trainer.step,
                            lambda: render_preview(trainer, batch, identity), keep)


def publish_preview_payload(run_dir, payload, identity, step, keep=DEFAULT_KEEP):
    """Publish an already-rendered bounded JSON preview without numerical imports."""
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1
            or payload.get('kind') != 'ema-preview' or type(payload.get('step')) is not int
            or payload['step'] != step
            or json.dumps(payload.get('identity'), sort_keys=True) != json.dumps(identity, sort_keys=True)
            or type(payload.get('count')) is not int or not 1 <= payload['count'] <= MAX_COUNT
            or type(payload.get('shape')) is not list or not payload['shape']
            or payload['shape'][0] != payload['count']
            or sample_name(payload.get('name')) != sample_name((identity or {}).get('name'))):
        raise ValueError('Rendered preview identity, step, name or shape is invalid')
    if payload.get('representation') == 'png' and (
            payload.get('samples') is not None or not isinstance(payload.get('image_grid'), dict)):
        raise ValueError('PNG-only previews require an image grid and no raw sample tensor')
    return _publish_preview(run_dir, identity, step, lambda: payload, keep)


GRIDS = (('image_grid', 'grid.png'), ('real_image_grid', 'real.png')) + tuple(
    (f'input_image_grid_{index}', f'input-{index}.png') for index in range(MAX_INPUT_GRIDS)) + tuple(
    (f'extra_image_grid_{index}', f'extra-{index}.png') for index in range(MAX_EXTRA_GRIDS)) + (
    ('comparison_image_grid', 'comparison.png'),)


def _publish_grids(temporary, target, payload):
    """Materialize renderer bytes inside the same atomic generation directory."""
    from .image_grids import MAX_BYTES as PNG_MAX_BYTES, inspect_png
    records = {}
    for field, filename in GRIDS:
        grid = payload.get(field)
        if grid is None:
            continue
        if (not isinstance(grid, dict) or not isinstance(grid.get('png_base64'), str)
                or len(grid['png_base64']) > 4 * ((PNG_MAX_BYTES + 2) // 3)):
            raise ValueError('Invalid bounded preview PNG payload')
        encoded = base64.b64decode(grid['png_base64'], validate=True)
        header = inspect_png(encoded)
        if any(grid.get(key) != value for key, value in header.items()):
            raise ValueError('Preview PNG dimensions differ from renderer metadata')
        metadata = {key: value for key, value in grid.items() if key != 'png_base64'}
        record = dict(metadata, path=str(target / filename), bytes=len(encoded),
                      sha256=hashlib.sha256(encoded).hexdigest(), media_type='image/png')
        record['name'] = sample_name(record.get('name'),
            REAL_NAME if field == 'real_image_grid' else sample_name(payload.get('name')))
        with (temporary / filename).open('xb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        payload = dict(payload, **{field: record})
        records[field] = record
    return payload, records


def _indexed_generations(root, run_id):
    """Map generation directory name to the record the published index holds.

    The index is this producer's own output under the run lock. It is still
    validated before reuse: anything unreadable or malformed simply falls back
    to rereading the generation manifests.
    """
    try:
        published = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if (not isinstance(published, dict) or published.get('schema_version') != 1
            or published.get('kind') != 'ema-preview-index'
            or published.get('run_id') != run_id
            or not isinstance(published.get('previews'), list)):
        return {}
    generations = {}
    for saved in published['previews']:
        identity = saved.get('identity') if isinstance(saved, dict) else None
        if (not isinstance(identity, dict) or type(identity.get('sample_sequence')) is not int
                or not isinstance(saved.get('path'), str) or type(saved.get('step')) is not int):
            return {}
        generations[Path(saved['path']).parent.name] = saved
    return generations


def thin(sequences, keep):
    """Choose which monotonic sample sequences a bounded run retains.

    The first sample of the run and the latest are always retained. The newest
    samples are retained at full density inside a window that advances in whole
    `DENSE_WINDOW` steps, and everything older is retained at a spacing that
    doubles each time the run outgrows `keep`. Both the doubling and the window's
    advance only ever remove sequences, so retention is nested: a later prune
    never wants back a sample an earlier one deleted. The answer depends only on
    the sequences that are on disk, so a history an older release already thinned
    is bounded from whatever its oldest surviving sample happens to be.
    """
    ordered = sorted(sequences)
    if keep == KEEP_ALL or len(ordered) <= keep:
        return set(ordered)
    if keep == 1:
        return {ordered[-1]}
    first, latest = ordered[0], ordered[-1]
    window = max(1, min(DENSE_WINDOW, keep - 1))
    dense = first + (latest - first) // window * window
    spacing = 1
    def retained(spacing):
        return {seq for seq in ordered if seq >= dense or (seq - first) % spacing == 0}
    kept = retained(spacing)
    # Beyond `latest - first` only the first sample satisfies the spacing, which
    # leaves `window + 1 <= keep` sequences, so this terminates inside the bound.
    while len(kept) > keep and spacing <= latest - first:
        spacing *= 2
        kept = retained(spacing)
    return kept


class _Pruner:
    """Deletes expired generation directories off the publishing path.

    Retention rewrites the index first and only then renames the generations it
    dropped, so a reader never holds a record for a directory this worker is
    about to delete, and the reindex scan skips the renamed ones by their dot
    prefix. A rename left behind by a crash is swept into the same worker by the
    next publication, and a generation whose rename never happened is simply
    rescanned and thinned away again by the same deterministic policy.
    """
    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._thread = None
        self._errors = []
        self._pending = 0

    def submit(self, path):
        with self._lock:
            self._pending += 1
            self._idle.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._drain_queue, daemon=True,
                                                name='hypergan-preview-pruner')
                self._thread.start()
        self._queue.put(path)

    def _drain_queue(self):
        while True:
            path = self._queue.get()
            try:
                shutil.rmtree(path, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as error:
                with self._lock:
                    self._errors = [*self._errors, f'Could not prune preview {path.name}: {error}'][-16:]
            finally:
                with self._lock:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.set()

    def errors(self):
        """Report and forget deletions that failed since the last publication."""
        with self._lock:
            reported, self._errors = self._errors, []
        return reported

    def drain(self, timeout=None):
        """Wait for queued deletions; the sweep recovers whatever does not finish."""
        return self._idle.wait(timeout)


_pruner = _Pruner()
atexit.register(_pruner.drain, 30)


def drain_pruning(timeout=None):
    """Wait for background preview deletion to settle (shutdown and tests)."""
    return _pruner.drain(timeout)


def _publish_preview(run_dir, identity, step, render, keep):
    """Publish a complete directory, update its index, then thin the history.

    The trainer's run lock serializes producers. A positive `keep` bounds the
    retained generations and thins the older ones in chunks (see `thin`);
    `KEEP_ALL` keeps the whole history. The index is rewritten before anything is
    removed and the expired directories are deleted by a background worker, so a
    prune never stalls the next publication. Only this managed preview directory
    is pruned; final inference bundles and recovery checkpoints are never removed.
    """
    if type(keep) is not int or keep < KEEP_ALL:
        raise ValueError('preview_keep must be a positive integer, '
                         f'or {KEEP_ALL} to keep every preview')
    root = Path(run_dir) / 'previews'
    if root.is_symlink():
        raise ValueError('Managed preview directory must not be a symlink')
    root.mkdir(exist_ok=True)
    sync_directory(root.parent)
    # No other producer can own a pending or expired directory while the trainer
    # run lock is held. An expired one is a prune an earlier process did not
    # finish; hand it back to the background worker instead of deleting it here.
    for stale in root.iterdir():
        if not (_GENERATION.fullmatch(stale.name) and stale.is_dir() and not stale.is_symlink()):
            continue
        if stale.name.startswith('.pending-'):
            shutil.rmtree(stale)
        elif stale.name.startswith('.expired-'):
            _pruner.submit(stale)
    name = f"{identity['sample_sequence']:012d}-{identity['attempt_id']}-step{step:08d}-{uuid.uuid4().hex}"
    temporary, target = root / ('.pending-' + name), root / name
    temporary.mkdir()
    completed = False
    try:
        payload = render()
        payload, grids = _publish_grids(temporary, target, payload)
        size = _write_bounded(temporary / 'preview.json', payload)
        record = {'schema_version': 1, 'kind': 'ema-preview', 'identity': dict(identity),
                  'name': _identity_name(identity, payload),
                  'step': step, 'path': str(target / 'preview.json'), 'bytes': size,
                  'sha256': hashlib.sha256((temporary / 'preview.json').read_bytes()).hexdigest(),
                  'count': payload['count'], 'shape': payload['shape'],
                  'representation': payload.get('representation', 'tensor')}
        record.update(grids)
        if 'diversity' in payload:
            record['diversity'] = payload['diversity']
        atomic_json(temporary / 'manifest.json', record)
        sync_directory(temporary)
        temporary.rename(target)
        sync_directory(root)
        # Scan only managed generation directories. A kept-forever history holds
        # thousands of them, so reuse the records the published index already
        # carries and read a manifest only for a directory the index does not
        # name. That also recovers a publication interrupted after rename but
        # before the index update in a prior attempt.
        published = _indexed_generations(root, identity['run_id'])
        records = []
        for entry in root.iterdir():
            # Directories handed to the background pruner carry a dot prefix and
            # are already absent from the index; skipping them here is what keeps
            # a prune in flight from being re-indexed.
            if entry.name.startswith('.'):
                continue
            if not (entry.is_dir() and not entry.is_symlink() and _GENERATION.fullmatch(entry.name)):
                continue
            saved = published.get(entry.name)
            if saved is None:
                try:
                    if not (entry / 'manifest.json').is_file():
                        continue
                    saved = json.loads((entry / 'manifest.json').read_text(encoding='utf-8'))
                except OSError:
                    continue  # Tolerate a directory that disappears mid-scan.
                if (saved.get('kind') != 'ema-preview'
                        or saved.get('identity', {}).get('run_id') != identity['run_id']):
                    continue
            # Paths are rewritten from the directory that holds them, so a run
            # copied to another location reindexes without rereading manifests.
            saved['path'] = str(entry / 'preview.json')
            saved['name'] = sample_name(saved.get('name'))
            for field, filename in GRIDS:
                if field in saved:
                    saved[field]['path'] = str(entry / filename)
            records.append((saved, entry))
        records.sort(key=lambda pair: pair[0]['identity']['sample_sequence'])
        kept = thin([saved['identity']['sample_sequence'] for saved, _ in records], keep)
        retained = [pair for pair in records if pair[0]['identity']['sample_sequence'] in kept]
        expired = [pair for pair in records if pair[0]['identity']['sample_sequence'] not in kept]
        index = {'schema_version': 1, 'kind': 'ema-preview-index', 'run_id': identity['run_id'],
                 'keep': keep, 'retention': 'all' if keep == KEEP_ALL else 'thinned',
                 'names': sorted({saved.get('name', DEFAULT_NAME) for saved, _ in retained}),
                 'previews': [saved for saved, _ in retained]}
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
        # The index above is the reader's view and already excludes these, so the
        # expensive part is only disk space: rename each one out of the scan (a
        # single cheap operation) and let the background worker delete it.
        errors = _pruner.errors()
        for _, entry in expired:
            retired = entry.parent / ('.expired-' + entry.name)
            try:
                entry.rename(retired)
            except OSError as exc:
                errors.append(f'Could not prune preview {entry.name}: {exc}')
            else:
                _pruner.submit(retired)
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
