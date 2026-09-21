"""Numerical snapshot evaluator, imported only inside its bounded worker."""
import hashlib
import importlib
import inspect
from pathlib import Path
import random

import numpy as np
import torch

from .artifacts import _restore_buffers
from .config import resolve_config
from .metric_plugins import _factory, finite_json
from .recipes import ComponentGraph, construct, execution_device, make_prior, move_tensors

MAX_BATCH_ELEMENTS = 16 * 1024 * 1024


def _sources(objects):
    result = {}
    for value in objects:
        module = inspect.getmodule(value)
        path = Path(getattr(module, '__file__', ''))
        if not path.is_file():
            raise ValueError('Evaluation sources must be inspectable files')
        result[module.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def evaluate_snapshot(spec, expected, snapshot, snapshot_sha256, identity):
    from .metric_evaluation import _sha256, MAX_SNAPSHOT_BYTES
    from .numerical_policy import apply_backend_policy, backend_info
    path = Path(snapshot)
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_SNAPSHOT_BYTES or _sha256(path) != snapshot_sha256:
        raise ValueError('Pinned evaluation snapshot identity changed')
    evaluation = spec['evaluation']
    seed = evaluation['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('schema_version') != 1 or saved.get('kind') != 'ema-inference':
        raise ValueError('Evaluation requires a current EMA inference bundle')
    owner = saved.get('identity', {})
    if owner.get('run_id') != identity['run_id'] or not isinstance(owner.get('attempt_id'), str) or type(saved.get('step')) is not int:
        raise ValueError('Snapshot lacks validated run/attempt/update provenance')
    if 'source_step' in identity and (saved['step'] != identity['source_step'] or
            any(owner.get(key) != identity.get(key) for key in ('attempt_id', 'attempt_index', 'evaluation_id'))):
        raise ValueError('Interval snapshot differs from the requested source step or attempt identity')
    config = resolve_config(saved['config'])
    generated_binding = evaluation.get('generated', config['sampling'].get('generated', 'generated'))
    if (generated_binding.startswith('components.')
            and generated_binding.split('.')[1] not in saved['components']):
        raise ValueError(f'Snapshot does not contain evaluation.generated component: {generated_binding}')
    apply_backend_policy({'training': {**config['training'], 'device': evaluation['device']}})
    device = execution_device(evaluation['device'])
    graph = ComponentGraph(saved['components']).float().eval().requires_grad_(False)
    for name, model in graph.models.items():
        model.load_state_dict(saved['model_states'][name])
        _restore_buffers(model, saved['model_buffers'][name])
    prior = make_prior(config['prior'], device='cpu').float().eval().requires_grad_(False)
    prior.load_state_dict(saved['prior'])
    _restore_buffers(prior, saved['prior_buffers'])
    graph, prior = graph.to(device), prior.to(device)
    data = construct(evaluation['data'])
    data_identity = data.resume_identity() if callable(getattr(data, 'resume_identity', None)) else None
    if data_identity is None:
        if ':' in evaluation['data']['factory']:
            raise ValueError('Custom evaluation data must expose resume_identity() for dataset/protocol provenance')
        data_identity = {'factory': evaluation['data']['factory'], 'args': evaluation['data']['args']}
    data_identity = finite_json(data_identity)
    instance, description = _factory(spec['factory'], spec['args'])
    if description != expected:
        raise ValueError('Metric source or descriptor changed since runtime preflight')
    constructors = [evaluation['data'], *saved['components'].values()]
    factory_modules = [importlib.import_module(value['factory'].split(':', 1)[0])
                       for value in constructors if ':' in value['factory']]
    code = _sources([type(data), type(prior), *[type(module) for module in graph.modules()],
                     *factory_modules, evaluate_snapshot, apply_backend_policy])
    data_rng = torch.Generator().manual_seed(seed + 1)
    prior_rng = torch.Generator(device=device).manual_seed(seed + 2)
    count = 0
    observed_backend = None
    runtime = {'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
               'backend': backend_info(), 'configured_training_backend': config['training']['backend']}
    def batches():
        nonlocal count, observed_backend
        while count < evaluation['sample_count']:
            size = min(evaluation['batch_size'], evaluation['sample_count'] - count)
            batch = data(size, generator=data_rng)
            if not isinstance(batch, dict) or not isinstance(batch.get('real'), torch.Tensor) or len(batch['real']) != size:
                raise ValueError('Evaluation data must return a batched real tensor')
            if any(not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != size for value in batch.values()):
                raise ValueError('Evaluation data must return only batched tensors')
            if sum(value.numel() for value in batch.values()) > MAX_BATCH_ELEMENTS:
                raise ValueError('Evaluation input batch exceeds element budget')
            batch = move_tensors(batch, device)
            z, _ = prior.sample(size, generator=prior_rng)
            effective_backend = backend_info()
            if observed_backend is not None and observed_backend != effective_backend:
                raise ValueError('Evaluation backend settings changed between generated batches')
            observed_backend = effective_backend
            runtime['backend'] = effective_backend
            generated = graph.resolve(generated_binding, graph.generate(z, batch, prior=prior))
            if (not isinstance(generated, torch.Tensor) or generated.ndim < 1 or len(generated) != size
                    or generated.numel() > MAX_BATCH_ELEMENTS or not torch.isfinite(generated).all()
                    or not torch.isfinite(batch['real']).all()):
                raise ValueError('Evaluation generated/reference tensors must be finite and bounded')
            context = {'evaluation.generated': generated, 'evaluation.reference': batch['real']}
            count += size
            yield {key: context[source] for key, source in spec['inputs'].items()}
    protocol = {'schema_version': 1, 'metric_factory': spec['factory'], 'factory_sources': description['factory_sources'],
                'args': spec['args'], 'inputs': spec['inputs'], 'data_identity': data_identity,
                'evaluation': evaluation, 'ema': True, 'sources': code,
                'generated_binding': generated_binding,
                'runtime': runtime}
    with torch.inference_mode():
        value = instance.evaluate(batches=batches(), context={'sample_count': evaluation['sample_count'],
                                  'seed': seed, 'step': saved['step'], 'snapshot_sha256': snapshot_sha256,
                                  'run_id': identity['run_id'], 'attempt_id': owner['attempt_id']})
    if count != evaluation['sample_count']:
        raise ValueError('Snapshot metric did not consume its complete declared evaluation sample count')
    value = finite_json(value)
    if description['descriptor']['kind'] == 'scalar':
        if type(value) not in (int, float):
            raise ValueError('Scalar snapshot metric must return a finite Python number')
    else:
        if not isinstance(value, dict) or set(value) != {'edges', 'counts'}:
            raise ValueError('Histogram metric must return edges and counts')
        edges, counts = value['edges'], value['counts']
        if (not isinstance(edges, list) or not isinstance(counts, list) or not 1 <= len(counts) <= 512
                or len(edges) != len(counts) + 1 or any(type(item) not in (int, float) for item in edges + counts)
                or any(a >= b for a, b in zip(edges, edges[1:])) or any(item < 0 for item in counts)):
            raise ValueError('Histogram requires finite ordered edges and at most 512 nonnegative counts')
    return finite_json({'value': value, 'protocol': protocol, 'snapshot_identity': owner, 'step': saved['step']})
