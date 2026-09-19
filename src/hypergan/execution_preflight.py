"""Construction-only CPU/CUDA runtime preflight in bounded, disposable workers.

Importing and calling the parent side loads no numerical runtime. Custom Python
constructors and identity hooks run only in supervised workers. No batch/forward,
optimizer update, checkpoint, inference artifact or run directory is produced.
"""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile

from .config import config_values, fingerprint, resolve_config
from .cpu_workers import launch_cpu_workers
from .cpu_worker_service import CPUWorkerService
from .execution_profiles import resolve_execution_profile


MAX_REPORT_BYTES = 1024 * 1024


def _encode(value):
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f'Preflight identity/report must contain finite JSON values: {exc}') from exc
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError(f'Preflight identity/report exceeds {MAX_REPORT_BYTES} bytes')
    return encoded


def _differences(expected, actual, path='identity'):
    if type(expected) is not type(actual):
        return [path + ' (type differs)']
    if isinstance(expected, dict):
        result = [f'{path}.{key} (missing or unexpected)' for key in sorted(expected.keys() ^ actual.keys(), key=str)]
        for key in sorted(expected.keys() & actual.keys(), key=str):
            result.extend(_differences(expected[key], actual[key], f'{path}.{key}'))
            if len(result) >= 16:
                break
        return result[:16]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [path + ' (length differs)']
        result = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(_differences(left, right, f'{path}[{index}]'))
            if len(result) >= 16:
                break
        return result[:16]
    return [] if expected == actual else [path]


def _resolve_profile(profile, config):
    if isinstance(profile, dict) and isinstance(profile.get('execution'), dict):
        derived = {'global_batch_size', 'local_batch_size', 'microbatch_size', 'accumulation_algorithm'}
        if derived & profile['execution'].keys():
            projected = {'schema_version': profile.get('schema_version'),
                         'execution': {key: profile['execution'][key] for key in ('name', 'world_size', 'accumulation_steps') if key in profile['execution']},
                         'preflight': profile.get('preflight', {})}
            resolved = resolve_execution_profile(projected, config)
            differences = _differences(resolved, profile, 'profile')
            if differences:
                raise ValueError('Resolved profile differs from configuration: ' + ', '.join(differences))
            return resolved
    return resolve_execution_profile(profile, config)


def _source_hashes(names):
    result = {}
    for name in names:
        module = importlib.import_module(name)
        path = getattr(module, '__file__', None)
        if not path or not Path(path).is_file():
            raise ValueError(f'Preflight source is not inspectable: {name}')
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def _runtime_worker(rank, world_size, config, profile, directory):
    import warnings
    import torch
    from .checkpoints import capture_rng, restore_rng
    from .training import ReferenceTrainer, _implementation, _recovery_contract, runtime_info

    config = resolve_config(config)
    requested = profile['execution']
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if requested['name'] == 'cpu-single':
            trainer = ReferenceTrainer(config)
        else:
            from .distributed_training import ReplicatedTrainer
            trainer = ReplicatedTrainer(config, world_size=world_size,
                                           accumulation_steps=requested['accumulation_steps'])
        if trainer.step != 0:
            raise ValueError('Preflight constructed a trainer with a nonzero update counter')
        if torch.get_num_threads() != 1:
            raise ValueError(f'Runtime threads differs from execution profile: expected 1, found {torch.get_num_threads()}')
        if torch.get_default_dtype() != torch.float32:
            raise ValueError(f'Runtime default_dtype differs from float32 profile: {torch.get_default_dtype()}')
        for owner in ('graph', 'prior', 'ema_graph', 'ema_prior'):
            module = getattr(trainer, owner)
            for kind, values in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
                for name, value in values:
                    field = f'{owner}.{kind}.{name}'
                    if value.device != trainer.device or value.layout != torch.strided:
                        raise ValueError(f'{field} must be a dense tensor on {trainer.device}')
                    if value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
                        raise ValueError(f'{field} must use float32')
                    if not torch.isfinite(value).all():
                        raise ValueError(f'{field} contains nonfinite values')
        if requested['name'] != 'cpu-single':
            observed = {key: trainer.strategy_info.get(key) for key in requested}
            differences = _differences(requested, observed, 'execution')
            if differences:
                raise ValueError('Constructed strategy differs from resolved profile: ' + ', '.join(differences))
        rng = capture_rng()
        try:
            contract, reasons = _recovery_contract(trainer)
            implementation = _implementation(trainer)
            if requested['name'] != 'cpu-single':
                implementation.update(_source_hashes(['hypergan.distributed_training', 'hypergan.distributed',
                                                       'hypergan.distributed_checkpoints', 'hypergan.distributed_commit',
                                                       'hypergan.replicated_execution', 'hypergan.replicated_worker',
                                                       'hypergan.cpu_worker_service', 'hypergan.preview_snapshot',
                                                       'hypergan.snapshot_renderer', 'hypergan.previews',
                                                       'hypergan.bounded_observer']))
            # Snapshot only numerical state: data is described by its contract;
            # sampling it or simulating checkpoint publication would exceed scope.
            from .distributed_checkpoints import _digest
            state = {}
            for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
                module = getattr(trainer, name)
                state[name] = {'state': module.state_dict(), 'buffers': dict(module.named_buffers()),
                               'modes': {key: child.training for key, child in module.named_modules()},
                               'trainable': {key: value.requires_grad for key, value in module.named_parameters()}}
            state['optimizers'] = [trainer.opt_g.state_dict(), trainer.opt_d.state_dict()]
            state['base_lrs'] = trainer.base_lrs
            from .recipes import move_tensors
            initial_state = _digest(move_tensors(state, 'cpu'))
        finally:
            restore_rng(rng)
        # Identity/state hooks are trusted Python too; reject runtime mutations
        # after inspection rather than publishing a contradictory execution profile.
        if torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32:
            raise ValueError('Identity/state inspection changed ' + ('CPU ' if trainer.device.type == 'cpu' else 'CUDA ') + 'runtime threads or default_dtype')
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize(trainer.device)
        if requested['name'] != 'cpu-single':
            from .distributed_checkpoints import distributed_runtime_info
            runtime = distributed_runtime_info(trainer)
        else:
            runtime = runtime_info()
            runtime.update(world_size=world_size, threads=torch.get_num_threads(),
                           interop_threads=torch.get_num_interop_threads(), backend='none')
        identity = {'config_sha256': fingerprint(config), 'execution': requested,
                    'runtime': runtime, 'implementation': implementation, 'data_contract': contract,
                    'recovery': {'supported': not reasons, 'reasons': reasons},
                    'initial_state_sha256': initial_state,
                    'strategy': trainer.strategy_info if requested['name'] != 'cpu-single' else {
                        **requested, 'gradient_reduction': 'none', 'data': 'single-process-draw',
                        'buffers': 'native-module-state', 'qualification': 'unqualified'}}
        warning_messages = list(dict.fromkeys([*config['warnings'], *reasons,
                                               *[str(item.message)[:1000] for item in captured]]))[:32]
        result = {'rank': rank, 'status': 'passed', 'step': 0, 'identity': identity, 'warnings': warning_messages}
        Path(directory, f'rank-{rank}.json').write_bytes(_encode(result))


def _preflight_factory(rank, world_size, config, profile, directory):
    return rank, world_size, config, profile, directory


def _preflight_command(state, operation, payload):
    if operation != 'construct' or payload is not None:
        raise ValueError('Invalid runtime preflight operation')
    _runtime_worker(*state)
    return None


def preflight(config, profile, *, expected_identity=None):
    """Return a JSON report after every construction worker exits successfully.

    ``config`` may be a resolved or raw recipe dictionary. ``profile`` may be
    raw or resolved profile values. ``expected_identity`` compares the complete
    previous numerical identity strictly; timeout policy and checker source are
    reported separately. This API must be called from an importable Python file
    under a main guard, following the existing spawn supervisor contract.
    """
    if isinstance(config, dict) and {'warnings', 'qualification'} & config.keys():
        resolved = resolve_config({key: value for key, value in config.items() if key not in ('warnings', 'qualification')})
        differences = _differences(resolved, config, 'config')
        if differences:
            raise ValueError('Resolved configuration metadata differs: ' + ', '.join(differences))
        config = resolved
    else:
        config = resolve_config(config)
    profile = _resolve_profile(profile, config)
    if expected_identity is not None:
        if not isinstance(expected_identity, dict):
            raise ValueError('expected_identity must be a complete identity dictionary')
        expected_identity = json.loads(_encode(expected_identity))
    with tempfile.TemporaryDirectory(prefix='hypergan-preflight-') as directory:
        if profile['execution']['name'] == 'cuda-replicated-nccl':
            limits = profile['preflight']
            with CPUWorkerService(_preflight_factory, _preflight_command,
                    args=(config_values(config), profile, directory),
                    run_id='preflight', attempt_id='construction',
                    world_size=profile['execution']['world_size'], backend='nccl',
                    startup_timeout=limits['timeout'], command_timeout=limits['timeout'],
                    collective_timeout=limits['collective_timeout'], total_timeout=limits['timeout']) as service:
                service.command('construct')
        else:
            launch_cpu_workers(_runtime_worker, args=(config_values(config), profile, directory),
                               world_size=profile['execution']['world_size'],
                               timeout=profile['preflight']['timeout'],
                               collective_timeout=profile['preflight']['collective_timeout'],
                               stdout_to_stderr=True,
                               initialize_process_group=profile['execution']['name'] != 'cpu-single')
        workers = []
        for rank in range(profile['execution']['world_size']):
            path = Path(directory, f'rank-{rank}.json')
            if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError(f'Missing or oversized preflight result from rank {rank}')
            worker = json.loads(path.read_bytes())
            if not isinstance(worker, dict) or worker.get('rank') != rank or worker.get('status') != 'passed' or worker.get('step') != 0:
                raise ValueError(f'Invalid preflight result from rank {rank}')
            workers.append(worker)
    identity = workers[0]['identity']
    for worker in workers[1:]:
        differences = _differences(identity, worker['identity'])
        if differences:
            raise ValueError(f"Preflight rank {worker['rank']} identity differs from rank 0: " + ', '.join(differences))
    if expected_identity is not None:
        differences = _differences(expected_identity, identity)
        if differences:
            raise ValueError('Preflight expected identity differs: ' + ', '.join(differences))
    report = {'schema_version': 1, 'status': 'passed', 'stage': 'runtime', 'runtime_checked': True, 'scope': 'construction-only',
              'profile': profile, 'identity': identity,
              'checker': _source_hashes(['hypergan.execution_preflight', 'hypergan.execution_profiles', 'hypergan.cpu_workers'] +
                                        (['hypergan.cpu_worker_service'] if profile['execution']['name'] == 'cuda-replicated-nccl' else [])),
              'ranks': [{key: value for key, value in worker.items() if key != 'identity'} for worker in workers],
              'warnings': list(dict.fromkeys(message for worker in workers for message in worker['warnings'])),
              'not_validated': ['data batches and model forward I/O', 'optimizer updates and numerical parity',
                                'checkpoint publication or restore', 'complete GPU training' if profile['execution']['name'] == 'cuda-replicated-nccl' else 'GPU execution', 'cluster execution']}
    _encode(report)
    return report
