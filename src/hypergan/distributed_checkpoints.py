"""Complete fixed-topology Gloo/NCCL checkpoints with separate preparation.

All ranks call each API in the same order, on a finite-timeout default group.
New services hold the lock in the parent, prepare on workers and publish through
a parent-only authority. The compatibility save API retains rank-zero publication
under its caller-owned lock. This format is distinct from single-process checkpoints.
All rank state is staged before a final readiness collective. A disconnect after commit
can report failure while leaving a valid complete generation; no protocol can make
filesystem commit and continued worker liveness one atomic operation.
"""
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import types
import uuid

import torch
import torch.distributed as dist

from .checkpoints import (
    capture_rng,
    file_sha256,
    read_checkpoint,
    resolve_resume_checkpoint_dir,
    restore_rng,
    restore_trainer,
    restore_trainer_cross_topology,
    trainer_state,
)
from .checkpoint_compatibility import (
    CURRENT_VERSION,
    format_topology_change_warning,
    validate_checkpoint_compatibility,
    validate_implementation,
    validate_runtime,
)
from .config import config_values, fingerprint, numerical_values, resume_compatible
from .recipes import move_tensors
from .run_state import atomic_json, sync_directory
from .training import _implementation, _recovery_contract, runtime_info, source_info
from .distributed_commit import (MAX_RANK_BYTES, make_prepared_receipt, preparation_directory,
                                 validate_fence, _validate_prepared, _publish)

SCHEMA = 1
KIND = 'hypergan-distributed-training-checkpoint'
MAX_METADATA_BYTES = 16 * 1024 * 1024
_SHARED = ('graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
           'step', 'buffers', 'trainable', 'modes')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')


def _group():
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError('Initialize a default Gloo or NCCL group with a finite timeout before distributed checkpointing')
    if dist.get_backend() not in ('gloo', 'nccl') or dist.get_world_size() < 2:
        raise ValueError('Distributed checkpoints require a fixed Gloo or NCCL group with at least two ranks')
    return dist.get_rank(), dist.get_world_size()


def distributed_runtime_info(trainer):
    """Common local fixed-topology identity, without introducing collectives.

    Every rank sees the same ordered visible devices. The CUDA mapping is
    rank -> visible index; record the entire mapping rather than comparing
    different rank-local UUIDs as though they should be equal.
    """
    backend = dist.get_backend()
    device = getattr(trainer, 'device', torch.device('cpu'))
    if backend == 'nccl':
        ranks_per_node = getattr(trainer, 'ranks_per_node', getattr(trainer, 'strategy_info', {}).get('ranks_per_node', trainer.world_size))
        local_rank = trainer.rank % ranks_per_node
        if (device != torch.device('cuda', local_rank)
                or torch.cuda.current_device() != local_rank
                or torch.cuda.device_count() < ranks_per_node):
            raise ValueError('NCCL runtime requires rank-owned visible CUDA devices')
        runtime = runtime_info('cuda:0')
        runtime['device'] = 'cuda'
        cuda = runtime['cuda']
        is_rocm = getattr(torch.version, 'hip', None) is not None
        pop_keys = ('name', 'capability', 'uuid', 'gcnArchName') if is_rocm else ('name', 'capability', 'uuid')
        for key in pop_keys:
            cuda.pop(key, None)
        nodes = getattr(trainer, 'nodes', getattr(trainer, 'strategy_info', {}).get('nodes', 1))
        is_multinode = nodes > 1
        if is_multinode:
            cuda.pop('visible_devices', None)
        nccl_ver = getattr(getattr(torch, 'cuda', None), 'nccl', None)
        cuda['nccl'] = list(nccl_ver.version()) if nccl_ver and hasattr(nccl_ver, 'version') else []
        rank_devices = []
        for rank in range(ranks_per_node):
            prop = torch.cuda.get_device_properties(rank)
            uuid_str = str(getattr(prop, 'uuid', 'unavailable'))
            if is_multinode or (is_rocm and uuid_str in ('None', '')):
                uuid_str = 'unavailable'
            entry = {
                'rank': rank,
                'device': f'cuda:{rank}',
                'uuid': uuid_str,
                'name': prop.name,
                'capability': list(torch.cuda.get_device_capability(rank)),
            }
            if is_rocm:
                gcn = getattr(prop, 'gcnArchName', None)
                entry['gcnArchName'] = str(gcn) if gcn not in (None, '', 'None') else 'unavailable'
            rank_devices.append(entry)
        cuda['rank_devices'] = rank_devices
    else:
        if backend != 'gloo' or device.type != 'cpu':
            raise ValueError('Gloo runtime requires CPU training state')
        runtime = runtime_info()
    runtime.update(world_size=trainer.world_size, backend=backend,
                   threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    return runtime


def _rank_state(trainer, last_batch):
    # Keep wire payloads device-independent; CUDA tensors never travel through
    # pickle with a rank-local device ordinal. Host copies complete pending work.
    if trainer.device.type == 'cuda':
        ranks_per_node = getattr(trainer, 'ranks_per_node', getattr(trainer, 'strategy_info', {}).get('ranks_per_node', trainer.world_size))
        local_rank = trainer.rank % ranks_per_node
        if torch.cuda.current_device() != local_rank:
            raise ValueError('CUDA checkpoint requires the rank-owned current device')
        torch.cuda.synchronize(trainer.device)
    state = trainer_state(trainer, last_batch)
    strategy = getattr(trainer, 'aggregation_strategy', getattr(trainer, 'aggregation', None))
    if strategy is not None and hasattr(strategy, 'state_dict') and callable(strategy.state_dict):
        state['aggregation'] = move_tensors(strategy.state_dict(), 'cpu')
    return move_tensors(state, 'cpu')


def _validate_rank_rng(trainer, state):
    if trainer.device.type == 'cuda':
        rng = state.get('rng')
        ranks_per_node = getattr(trainer, 'ranks_per_node', getattr(trainer, 'strategy_info', {}).get('ranks_per_node', trainer.world_size))
        local_rank = trainer.rank % ranks_per_node
        if not isinstance(rng, dict) or type(rng.get('cuda_device')) is not int or rng['cuda_device'] != local_rank:
            raise ValueError('Checkpoint CUDA RNG current device differs from the owning rank')


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _same_json(left, right):
    # Python container equality aliases True/1 and 1/1.0. Recovery identity and
    # all-rank descriptors require exact JSON value types as well as values.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _local_device(trainer):
    """The rank-owned CUDA device: local rank on this node, never the global rank."""
    device = getattr(trainer, 'device', None)
    if isinstance(device, torch.device) and device.type == 'cuda' and device.index is not None:
        return device
    ranks_per_node = getattr(trainer, 'ranks_per_node', None) or getattr(trainer, 'strategy_info', {}).get('ranks_per_node')
    return torch.device('cuda', dist.get_rank() % (ranks_per_node or torch.cuda.device_count()))


def _agree(operation, value=None, error=None, *, trainer=None):
    # Custom hooks may change the ambient device before failing. Agree on that
    # failure using this rank's device, never another rank's NCCL communicator.
    if dist.get_backend() == 'nccl':
        torch.cuda.set_device(_local_device(trainer))
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, {'operation': operation, 'value': value, 'error': error})
    if any(peer['operation'] != operation for peer in peers):
        raise ValueError('Ranks entered different distributed checkpoint operations')
    errors = [f"rank {rank}: {peer['error']}" for rank, peer in enumerate(peers) if peer['error']]
    if errors:
        raise ValueError('Distributed checkpoint rejected: ' + '; '.join(errors))
    return [peer['value'] for peer in peers]


def _digest(value):
    """Stable value digest, independent of pickle storage IDs and tensor aliasing."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != 'cpu' or item.layout != torch.strided:
                raise ValueError('Checkpoint tensors must be dense CPU tensors')
            digest.update(b'tensor')
            visit(str(item.dtype))
            visit(list(item.shape))
            data = item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
        elif isinstance(item, dict):
            digest.update(b'dict')
            visit(len(item))
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b'tuple' if isinstance(item, tuple) else b'list')
            visit(len(item))
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, int, float, bool):
            data = json.dumps(item, allow_nan=False).encode()
            digest.update(type(item).__name__.encode() + b':' + str(len(data)).encode() + b':' + data)
        else:
            raise ValueError(f'Unsupported distributed checkpoint value: {type(item).__name__}')
    visit(value)
    return digest.hexdigest()


def _shared_digest(state):
    return _digest({key: state[key] for key in _SHARED})


def _distributed_checkpoint_identity(trainer):
    """Record compatibility and build provenance on this rank."""
    rank, world_size = _group()
    if (getattr(trainer, 'rank', None) != rank or getattr(trainer, 'world_size', None) != world_size
            or not isinstance(getattr(trainer, 'strategy_info', None), dict)):
        raise ValueError('Trainer rank/topology does not match the active process group')
    strategy = _json(trainer.strategy_info)
    if strategy.get('world_size') != world_size:
        raise ValueError('Trainer strategy world_size does not match the process group')
    if strategy.get('global_batch_size') != trainer.config['training']['batch_size']:
        raise ValueError('Trainer strategy global batch differs from configuration')
    if strategy.get('local_batch_size') * world_size != strategy.get('global_batch_size'):
        raise ValueError('Trainer strategy requires equal rank batch sizes')
    contract, reasons = _recovery_contract(trainer)
    if reasons:
        raise ValueError('Recovery unsupported: ' + '; '.join(reasons))
    cached_provenance = getattr(trainer, '_checkpoint_provenance', None)
    if cached_provenance is None:
        implementation = _implementation(trainer)
        for module in (inspect.getmodule(type(trainer)), inspect.getmodule(distributed_checkpoint_identity)):
            path = getattr(module, '__file__', None)
            if not path or not Path(path).is_file():
                raise ValueError('Distributed trainer/checkpoint implementation needs an inspectable source file')
            implementation[module.__name__] = file_sha256(path)
        import hypergan.distributed
        implementation['hypergan.distributed'] = file_sha256(hypergan.distributed.__file__)
        import hypergan.distributed_commit
        implementation['hypergan.distributed_commit'] = file_sha256(hypergan.distributed_commit.__file__)
        # Worker execution and parent publication policy affect continuation too.
        # Hash files without importing the numerical worker in a parent process.
        for name in ('replicated_execution', 'replicated_worker', 'cpu_worker_service',
                     'preview_snapshot', 'snapshot_renderer', 'previews', 'bounded_observer'):
            implementation['hypergan.' + name] = file_sha256(Path(__file__).with_name(name + '.py'))
        source = source_info(trainer.config)
        try:
            trainer._checkpoint_provenance = (copy.deepcopy(implementation), copy.deepcopy(source))
        except (AttributeError, TypeError):
            pass
    else:
        implementation, source = copy.deepcopy(cached_provenance)
    runtime = distributed_runtime_info(trainer)
    topology = {k: v for k, v in strategy.items() if k not in ('node_rank', 'rendezvous')}
    return _json({'hypergan_checkpoint_version': CURRENT_VERSION,
                  'config': numerical_values(trainer.config), 'config_sha256': fingerprint(trainer.config),
                  'runtime': runtime, 'implementation': implementation, 'data_contract': contract,
                  'topology': topology, 'source': source})


def distributed_checkpoint_identity(trainer):
    """Read compatibility without consuming a custom identity hook's global RNG."""
    rng = capture_rng()
    try:
        return _distributed_checkpoint_identity(trainer)
    finally:
        restore_rng(rng)


def _metadata(value, restore=False):
    required = {'run_id'} if restore else {'run_id', 'attempt_id'}
    allowed = required if restore else required | {'next_sample_sequence', 'request_ids', 'event_boundary',
                                                 'source', 'initial_source'}
    if not isinstance(value, dict) or not required <= set(value) <= allowed:
        raise ValueError('Invalid distributed checkpoint lineage metadata fields')
    for key in required:
        if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
            raise ValueError(f'Distributed checkpoint {key} must be a safe nonempty identifier')
    result = _json(value)
    if not restore:
        result.setdefault('next_sample_sequence', 1)
        if type(result['next_sample_sequence']) is not int or result['next_sample_sequence'] < 1:
            raise ValueError('next_sample_sequence must be a positive integer')
        ids = result.get('request_ids', [])
        if not isinstance(ids, list) or len(ids) > 256 or any(not isinstance(i, str) or not _ID.fullmatch(i) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('request_ids must contain at most 256 distinct safe IDs')
    return result


def _serialize(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    if stream.tell() > MAX_RANK_BYTES:
        raise ValueError(f'Checkpoint rank payload exceeds {MAX_RANK_BYTES} bytes')
    return stream.getvalue()


def _deserialize(payload):
    try:
        state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or any(key not in state for key in _SHARED):
            raise ValueError('Incomplete distributed rank state')
        return state
    except (pickle.UnpicklingError, RuntimeError, EOFError, KeyError, TypeError) as exc:
        raise ValueError('Invalid distributed checkpoint payload; safe tensor loading failed') from exc


def _root(run_dir):
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise ValueError(f'Run directory must already exist: {run}')
    root = run / 'distributed-checkpoints'
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError('Distributed checkpoint root must be an ordinary directory')
    return root


def save_distributed_checkpoint(run_dir, trainer, last_batch, metadata):
    """Compatibility API: prepare on every rank, then publish from rank zero.

    This standalone path still requires the legacy rank-zero run lock. New parent
    services must call prepare_distributed_checkpoint and use their parent-owned
    CheckpointCommitAuthority instead; this API is not a worker service command.
    """
    rank, _ = _group()
    controller = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(controller, src=0)
    receipt = prepare_distributed_checkpoint(run_dir, trainer, last_batch, metadata,
                                            command_sequence=1, controller_id=controller[0])
    result = [None]
    if rank == 0:
        try:
            staging, target, info = _validate_prepared(run_dir, receipt,
                run_id=metadata['run_id'], attempt_id=metadata['attempt_id'],
                controller_id=controller[0], command_sequence=1,
                identity=distributed_checkpoint_identity(trainer))
            path = _publish(staging, target, info)
            result[0] = {'path': str(path), 'error': None}
        except Exception as exc:
            result[0] = {'path': None, 'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(result, src=0)
    if result[0]['error']:
        raise ValueError('Distributed checkpoint publication failed: ' + result[0]['error'])
    return Path(result[0]['path'])


def prepare_distributed_checkpoint(run_dir, trainer, last_batch, metadata, *, command_sequence, controller_id):
    """Stage every complete rank state without publishing a generation or latest pointer.

    Each payload is capped at MAX_RANK_BYTES (1 GiB); rank zero holds O(world_size * rank_bytes)
    memory. Trusted-worker object collectives carry CPU-serialized state (NCCL
    stages these bytes through the rank GPU). This is not a scalable/sharded
    checkpoint transport. The parent owns run_lock.
    Return only a bounded receipt; its presence is not proof that a supervised
    command completed on every worker. A parent authority publishes separately.
    """
    rank, world_size = _group()
    # Experimental local-SGD replicas differ between synchronizations; one complete
    # state needs them averaged first. Collective, and a no-op at sync boundaries.
    synchronize = getattr(trainer, 'synchronize_replicas', None)
    if callable(synchronize):
        synchronize()
    rng = capture_rng()
    temporary = None
    try:
        error, descriptor, payload = None, None, None
        try:
            root = _root(run_dir) if rank == 0 else None
            sync_packet = [controller_id, command_sequence, metadata] if rank == 0 else [None, None, None]
            dist.broadcast_object_list(sync_packet, src=0)
            controller_id, command_sequence, metadata = sync_packet
            validate_fence(controller_id, command_sequence)
            if world_size > 64:
                raise ValueError('Prepared checkpoints support at most 64 ranks')
            if getattr(trainer, 'checkpoint_ready', False) is not True:
                raise ValueError('Trainer is not at a complete update boundary; half/failed updates cannot be checkpointed')
            lineage = _metadata(metadata)
            identity = distributed_checkpoint_identity(trainer)
            if type(trainer.step) is not int or not 0 <= trainer.step <= trainer.config['training']['steps']:
                raise ValueError('Checkpoint step is outside the configured schedule')
            state = _rank_state(trainer, last_batch)
            _digest(state)  # All rank-local tensors must follow the CPU state contract too.
            payload = _serialize(state)
            descriptor = {'lineage': lineage, 'identity': identity,
                          'controller_id': controller_id, 'command_sequence': command_sequence,
                          'step': trainer.step, 'replicated_sha256': _shared_digest(state)}
            if len(json.dumps(descriptor, allow_nan=False).encode()) > MAX_METADATA_BYTES:
                raise ValueError('Distributed checkpoint metadata exceeds its byte bound')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        peers = _agree('prepare:collect', descriptor, error, trainer=trainer)
        if any(not _same_json(peer, peers[0]) for peer in peers):
            raise ValueError('Ranks disagree on checkpoint step, lineage, config/runtime/source/data/topology or replicated numerical state')
        payloads = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, object_gather_list=payloads, dst=0)
        error, receipt = None, None
        if rank == 0:
            try:
                nonce = uuid.uuid4().hex[:12]
                temporary = preparation_directory(run_dir, lineage['attempt_id'], controller_id,
                                                  command_sequence, nonce, create=True)
                records = []
                for index, rank_payload in enumerate(payloads):
                    if not isinstance(rank_payload, bytes) or len(rank_payload) > MAX_RANK_BYTES:
                        raise ValueError('Invalid/oversized gathered rank payload')
                    state = _deserialize(rank_payload)
                    if state['step'] != trainer.step or _shared_digest(state) != descriptor['replicated_sha256']:
                        raise ValueError('Gathered rank payload differs from complete replica agreement')
                    filename = f'rank-{index:05d}.pt'
                    with (temporary / filename).open('wb') as output:
                        output.write(rank_payload)
                        output.flush()
                        os.fsync(output.fileno())
                    records.append({'rank': index, 'file': filename, 'bytes': len(rank_payload),
                                    'sha256': hashlib.sha256(rank_payload).hexdigest()})
                strategy = getattr(trainer, 'aggregation_strategy', getattr(trainer, 'aggregation', None))
                agg_meta = strategy.manifest_metadata() if hasattr(strategy, 'manifest_metadata') else (
                    dict(strategy) if isinstance(strategy, dict) else None
                )
                info = dict(lineage, schema_version=SCHEMA, kind=KIND, step=trainer.step,
                            identity=identity, replicated_sha256=descriptor['replicated_sha256'], ranks=records)
                if agg_meta is not None:
                    info['aggregation'] = agg_meta
                if len((json.dumps(info, indent=2, allow_nan=False) + '\n').encode()) > MAX_METADATA_BYTES:
                    raise ValueError('Distributed checkpoint metadata exceeds its byte bound')
                atomic_json(temporary / 'manifest.json', info)
                sync_directory(temporary)
                receipt = make_prepared_receipt(run_dir, temporary, info, controller_id=controller_id,
                                                command_sequence=command_sequence, nonce=nonce)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        # Missing ranks must prevent the receipt from becoming a successful
        # all-rank command result. No worker in this API publishes canonical state.
        _agree('prepare:staged', error=error, trainer=trainer)
        result = [receipt]
        dist.broadcast_object_list(result, src=0)
        temporary = None  # Owned managed staging remains for parent validation.
        return result[0]
    finally:
        restore_rng(rng)
        if rank == 0 and temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _read_json(path):
    if path.is_symlink():
        raise ValueError('Distributed checkpoint metadata must not be a symlink')
    with path.open('rb') as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError('Distributed checkpoint metadata exceeds its byte bound')
    return json.loads(content)


def _validate_restore_identity(saved, current, *, warn=None):
    """Accept compatible HyperGAN builds without weakening current-rank agreement."""
    if not isinstance(saved, dict):
        raise ValueError('Distributed checkpoint identity must be a dictionary')
    validate_checkpoint_compatibility(saved)
    rt_warnings = validate_runtime(saved.get('runtime'), current['runtime'],
                                   warn=warn,
                                   saved_implementation=saved.get('implementation'),
                                   current_implementation=current['implementation'])
    validate_implementation(saved.get('implementation'), current['implementation'])
    for identity in (saved, current):
        if (not isinstance(identity.get('config'), dict)
                or fingerprint(identity['config']) != identity.get('config_sha256')):
            raise ValueError('Distributed checkpoint configuration fingerprint differs')
    if not resume_compatible(current['config'], saved['config']):
        raise ValueError('Distributed checkpoint config/data/topology identity differs')
    # The helpers checked the runtime, external implementations and contract.
    # Source provenance is recorded but does not decide restore compatibility.
    excluded = {'hypergan_checkpoint_version', 'runtime', 'implementation', 'source',
                'config', 'config_sha256'}
    saved_cmp = {key: value for key, value in saved.items() if key not in excluded}
    current_cmp = {key: value for key, value in current.items() if key not in excluded}
    if isinstance(saved_cmp.get('topology'), dict):
        saved_cmp['topology'] = {k: v for k, v in saved_cmp['topology'].items() if k not in ('node_rank', 'rendezvous')}
    if isinstance(current_cmp.get('topology'), dict):
        current_cmp['topology'] = {k: v for k, v in current_cmp['topology'].items() if k not in ('node_rank', 'rendezvous')}
    if not _same_json(saved_cmp, current_cmp):
        raise ValueError('Distributed checkpoint config/data/topology identity differs')
    return rt_warnings


def read_distributed_checkpoint_for_elastic(run_dir, target, expected_run):
    """Read and integrity-verify a distributed checkpoint of arbitrary world_size M >= 1.

    Verifies manifest schema, event boundary, every saved rank file's byte count,
    SHA-256 digest, step, and shared numerical state digest (`replicated_sha256`),
    then returns `(target, info, identity, rank0_state)`.
    """
    run_dir = Path(run_dir).resolve()
    root = (run_dir / 'distributed-checkpoints').resolve()
    target = Path(target)
    if not target.is_absolute():
        target = root / target
    if target.is_symlink():
        raise ValueError('Distributed checkpoint generation must not be a symlink')
    target = target.resolve()
    if target.parent != root or not target.is_dir() or target.name.startswith('.'):
        raise ValueError('Select a completed distributed checkpoint directory inside this run')
    info = _read_json(target / 'manifest.json')
    required = {'schema_version', 'kind', 'run_id', 'attempt_id', 'step', 'identity', 'replicated_sha256', 'ranks', 'next_sample_sequence'}
    allowed = required | {'request_ids', 'event_boundary', 'source', 'initial_source', 'aggregation'}
    if not isinstance(info, dict) or not required <= set(info) <= allowed or type(info['schema_version']) is not int or info['schema_version'] != SCHEMA or info['kind'] != KIND:
        raise ValueError('Invalid distributed checkpoint metadata/schema')
    from .run_state import validate_event_boundary
    validate_event_boundary(run_dir, info)
    if info['run_id'] != expected_run:
        raise ValueError('Distributed checkpoint run identity differs')
    identity = info.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('Distributed checkpoint identity must be a dictionary')
    validate_checkpoint_compatibility(identity)
    lineage = {key: info[key] for key in ('run_id', 'attempt_id', 'next_sample_sequence')}
    if 'request_ids' in info:
        lineage['request_ids'] = info['request_ids']
    _metadata(lineage)
    if type(info['step']) is not int or info['step'] < 0:
        raise ValueError('Distributed checkpoint step is invalid')
    if not isinstance(info['ranks'], list) or len(info['ranks']) < 1:
        raise ValueError('Distributed checkpoint must contain at least one saved rank')
    rank0_state = None
    for rank_idx, record in enumerate(info['ranks']):
        if (
            not isinstance(record, dict)
            or set(record) != {'rank', 'file', 'bytes', 'sha256'}
            or type(record['rank']) is not int
            or record['rank'] != rank_idx
            or record['file'] != f'rank-{rank_idx:05d}.pt'
        ):
            raise ValueError('Invalid distributed checkpoint rank inventory')
        if type(record['bytes']) is not int or not 1 <= record['bytes'] <= MAX_RANK_BYTES:
            raise ValueError('Invalid distributed checkpoint payload size')
        path = target / record['file']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Distributed rank payload size/type mismatch')
        with path.open('rb') as source:
            payload = source.read(record['bytes'] + 1)
        if len(payload) != record['bytes']:
            raise ValueError('Distributed rank payload changed size while reading')
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Distributed rank payload digest mismatch')
        state = _deserialize(payload)
        if state['step'] != info['step'] or _shared_digest(state) != info['replicated_sha256']:
            raise ValueError('Distributed rank numerical state differs from committed agreement')
        if rank_idx == 0:
            rank0_state = state
    return target, info, identity, rank0_state


def _is_same_distributed_topology(saved_identity, current_identity, saved_ranks_len, world_size):
    if not isinstance(saved_identity, dict) or saved_ranks_len != world_size:
        return False
    saved_cfg = saved_identity.get('config')
    current_cfg = current_identity.get('config')
    if not isinstance(saved_cfg, dict) or not isinstance(current_cfg, dict):
        return False
    saved_tr = saved_cfg.get('training', {})
    current_tr = current_cfg.get('training', {})
    if saved_tr.get('device') != current_tr.get('device') or saved_tr.get('batch_size') != current_tr.get('batch_size'):
        return False
    saved_topo = saved_identity.get('topology')
    current_topo = current_identity.get('topology')
    if not isinstance(saved_topo, dict) or not isinstance(current_topo, dict):
        return False
    s_clean = {k: v for k, v in saved_topo.items() if k not in ('node_rank', 'rendezvous')}
    c_clean = {k: v for k, v in current_topo.items() if k not in ('node_rank', 'rendezvous')}
    return _same_json(s_clean, c_clean)


def _read(root, checkpoint, expected_identity, expected_run, *, warn=None):
    pointer = None
    if checkpoint is None:
        pointer = _read_json(root / 'latest.json')
        if not isinstance(pointer, dict) or set(pointer) != {'schema_version', 'kind', 'checkpoint', 'step'} or type(pointer['schema_version']) is not int or pointer['schema_version'] != SCHEMA or pointer['kind'] != KIND:
            raise ValueError('Invalid distributed checkpoint pointer')
        if type(pointer['step']) is not int or not 0 <= pointer['step'] <= expected_identity['config']['training']['steps']:
            raise ValueError('Invalid distributed checkpoint pointer step')
        if not isinstance(pointer['checkpoint'], str) or Path(pointer['checkpoint']).name != pointer['checkpoint']:
            raise ValueError('Invalid distributed checkpoint pointer path')
        checkpoint = pointer['checkpoint']
    target = Path(checkpoint)
    if not target.is_absolute():
        target = root / target
    if target.is_symlink():
        raise ValueError('Distributed checkpoint generation must not be a symlink')
    target = target.resolve()
    if target.parent != root or not target.is_dir() or target.name.startswith('.'):
        raise ValueError('Select a completed distributed checkpoint directory inside this run')
    info = _read_json(target / 'manifest.json')
    required = {'schema_version', 'kind', 'run_id', 'attempt_id', 'step', 'identity', 'replicated_sha256', 'ranks', 'next_sample_sequence'}
    allowed = required | {'request_ids', 'event_boundary', 'source', 'initial_source', 'aggregation'}
    if not isinstance(info, dict) or not required <= set(info) <= allowed or type(info['schema_version']) is not int or info['schema_version'] != SCHEMA or info['kind'] != KIND:
        raise ValueError('Invalid distributed checkpoint metadata/schema')
    from .run_state import validate_event_boundary
    validate_event_boundary(root.parent, info)
    if info['run_id'] != expected_run:
        raise ValueError('Distributed checkpoint run identity differs')
    rt_warnings = _validate_restore_identity(info['identity'], expected_identity, warn=warn)
    lineage = {key: info[key] for key in ('run_id', 'attempt_id', 'next_sample_sequence')}
    if 'request_ids' in info:
        lineage['request_ids'] = info['request_ids']
    _metadata(lineage)
    if type(info['step']) is not int or not 0 <= info['step'] <= expected_identity['config']['training']['steps']:
        raise ValueError('Distributed checkpoint step is outside the original schedule')
    if pointer is not None and pointer['step'] != info['step']:
        raise ValueError('Distributed checkpoint pointer step differs from generation')
    if not isinstance(info['ranks'], list) or len(info['ranks']) != dist.get_world_size():
        raise ValueError('Distributed checkpoint must contain exactly every fixed rank')
    payloads = []
    for rank, record in enumerate(info['ranks']):
        if not isinstance(record, dict) or set(record) != {'rank', 'file', 'bytes', 'sha256'} or type(record['rank']) is not int or record['rank'] != rank or record['file'] != f'rank-{rank:05d}.pt':
            raise ValueError('Invalid distributed checkpoint rank inventory')
        if type(record['bytes']) is not int or not 1 <= record['bytes'] <= MAX_RANK_BYTES:
            raise ValueError('Invalid distributed checkpoint payload size')
        path = target / record['file']
        if path.is_symlink() or path.stat().st_size != record['bytes']:
            raise ValueError('Distributed rank payload size/type mismatch')
        with path.open('rb') as source:
            payload = source.read(record['bytes'] + 1)
        if len(payload) != record['bytes']:
            raise ValueError('Distributed rank payload changed size while reading')
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Distributed rank payload digest mismatch')
        state = _deserialize(payload)
        if state['step'] != info['step'] or _shared_digest(state) != info['replicated_sha256']:
            raise ValueError('Distributed rank numerical state differs from committed agreement')
        payloads.append(payload)
    if warn is not None:
        return target, info, payloads, rt_warnings
    return target, info, payloads


def _read_elastic(run_dir, checkpoint, expected_identity, expected_run, world_size):
    """Resolve across checkpoints/ and distributed-checkpoints/; choose exact or elastic restore."""
    run_manifest_path = Path(run_dir).resolve() / 'manifest.json'
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.is_file() else None
    target, pointer = resolve_resume_checkpoint_dir(run_dir, checkpoint, run_manifest=run_manifest)
    info = _read_json(target / 'manifest.json')
    if not isinstance(info, dict):
        raise ValueError('Invalid training checkpoint metadata')
    kind = info.get('kind')
    if kind == KIND:
        ranks_list = info.get('ranks')
        ranks_len = len(ranks_list) if isinstance(ranks_list, list) else -1
        if _is_same_distributed_topology(info.get('identity'), expected_identity, ranks_len, world_size):
            collected_warnings = []
            target, info, payloads, _ = _read(
                target.parent, target, expected_identity, expected_run, warn=collected_warnings.append
            )
            if pointer is not None and pointer.get('step') != info['step']:
                raise ValueError('Distributed checkpoint pointer step differs from generation')
            return 'exact', target, info, payloads, collected_warnings

        target, info, saved_identity, shared_state = read_distributed_checkpoint_for_elastic(
            run_dir, target, expected_run
        )
        saved_topology = saved_identity.get('topology')
    elif kind == 'hypergan-training-checkpoint':
        target, info, shared_state = read_checkpoint(run_dir, target)
        if info['run_id'] != expected_run:
            raise ValueError('Training checkpoint run identity differs')
        saved_identity = info
        saved_topology = None
    else:
        raise ValueError(f'Unsupported checkpoint kind for restore: {kind!r}')

    if pointer is not None and pointer.get('step') != info['step']:
        raise ValueError('Checkpoint pointer step differs from generation')
    if type(info.get('step')) is not int or not 0 <= info['step'] <= expected_identity['config']['training']['steps']:
        raise ValueError('Checkpoint step is outside the configured schedule')

    validate_checkpoint_compatibility(saved_identity)
    collected_warnings = []
    validate_runtime(
        saved_identity.get('runtime'),
        expected_identity['runtime'],
        warn=collected_warnings.append,
        saved_implementation=saved_identity.get('implementation'),
        current_implementation=expected_identity['implementation'],
        elastic_topology=True,
    )
    validate_implementation(saved_identity.get('implementation'), expected_identity['implementation'])
    saved_cfg = saved_identity.get('config')
    if not isinstance(saved_cfg, dict) or fingerprint(saved_cfg) != saved_identity.get('config_sha256'):
        raise ValueError('Checkpoint configuration fingerprint differs')
    if not resume_compatible(
        expected_identity['config'], saved_cfg, allow_device_change=True, allow_batch_size_change=True
    ):
        raise ValueError('Checkpoint configuration is incompatible with resume configuration')
    if not _same_json(saved_identity.get('data_contract'), expected_identity.get('data_contract')):
        raise ValueError('Checkpoint data identity or state protocol differs from trainer')

    topo_warning = format_topology_change_warning(
        saved_topology,
        expected_identity.get('topology'),
        saved_device=saved_cfg.get('training', {}).get('device'),
        current_device=expected_identity['config'].get('training', {}).get('device'),
        step=info['step'],
        emit=False,
    )
    collected_warnings.append(topo_warning)
    shared_payload = _serialize(move_tensors(shared_state, 'cpu'))
    return 'elastic', target, info, shared_payload, collected_warnings


def restore_distributed_checkpoint(run_dir, trainer, expected_metadata, checkpoint=None, *,
                                   allow_elastic=False, return_warnings=False):
    """Validate all rank files/identities before restoring each rank's own state.

    When ``allow_elastic`` is False (default), the same world size and rank ownership
    are mandatory. When ``allow_elastic`` is True, matching topologies still take the
    exact bitwise per-rank scatter path, while cross-topology checkpoints (`1 -> N`,
    `M -> N`, strategy/backend/device changes) broadcast verified shared numerical
    state from rank 0, re-seed per-rank streams deterministically, and assert 100%
    all-rank `_shared_digest` replica agreement.
    """
    rank, world_size = _group()
    initial_rng = capture_rng()
    error, descriptor = None, None
    try:
        root = _root(run_dir) if (rank == 0 and not allow_elastic) else None
        if rank == 0 and allow_elastic:
            run_path = Path(run_dir).resolve()
            if not run_path.is_dir():
                raise ValueError(f'Run directory must already exist: {run_path}')
        if getattr(trainer, '_poisoned', False):
            raise ValueError('Restore requires a fresh trainer after a failed update; restart the whole worker group')
        if getattr(trainer, 'checkpoint_ready', False) is not True:
            raise ValueError('Restore requires a fresh or complete-boundary trainer, not an active/incomplete update')
        expected = _metadata(expected_metadata, restore=True)
        identity = distributed_checkpoint_identity(trainer)
        descriptor = {'run_id': expected['run_id'], 'identity': identity,
                      'checkpoint': str(checkpoint) if checkpoint is not None else None}
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    restore_rng(initial_rng)
    peers = _agree('restore:prepare', descriptor, error, trainer=trainer)
    if any(not _same_json(peer, peers[0]) for peer in peers):
        raise ValueError('Ranks disagree on distributed checkpoint selection or expected identity')
    payloads, shared_payload, response = None, None, [None]
    if rank == 0:
        try:
            if allow_elastic:
                mode, target, info, raw_payload, warnings_out = _read_elastic(
                    run_dir, checkpoint, identity, expected['run_id'], world_size
                )
                if mode == 'exact':
                    payloads = raw_payload
                else:
                    shared_payload = raw_payload
                response[0] = {
                    'path': str(target),
                    'info': info,
                    'mode': mode,
                    'warnings': warnings_out,
                    'error': None,
                }
            else:
                target, info, payloads = _read(root, checkpoint, identity, expected['run_id'])
                response[0] = {'path': str(target), 'info': info, 'mode': 'exact', 'warnings': [], 'error': None}
        except Exception as exc:
            response[0] = {'path': None, 'info': None, 'mode': None, 'warnings': [], 'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(response, src=0)
    if response[0]['error']:
        raise ValueError('Invalid distributed checkpoint: ' + response[0]['error'])

    if response[0]['mode'] == 'elastic':
        shared_box = [shared_payload if rank == 0 else None]
        dist.broadcast_object_list(shared_box, src=0)
        error, candidate_shared_digest = None, None
        try:
            state = _deserialize(shared_box[0])
            memo = {id(module): module for module in list(sys.modules.values()) if isinstance(module, types.ModuleType)}
            candidate = copy.deepcopy(trainer, memo)
            candidate_last_batch = restore_trainer_cross_topology(
                candidate, copy.deepcopy(state), rank=rank, world_size=world_size
            )
            candidate_shared_digest = _shared_digest(_rank_state(candidate, candidate_last_batch))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            restore_rng(initial_rng)
        validated_digests = _agree('restore:validated', candidate_shared_digest, error=error, trainer=trainer)
        if any(item != validated_digests[0] for item in validated_digests):
            raise ValueError('Ranks disagree on cross-topology candidate shared numerical state')
        trainer.checkpoint_ready = False
        try:
            error, last_batch, live_shared_digest = None, None, None
            try:
                last_batch = restore_trainer_cross_topology(
                    trainer, _deserialize(shared_box[0]), rank=rank, world_size=world_size
                )
                live_shared_digest = _shared_digest(_rank_state(trainer, last_batch))
                if live_shared_digest != candidate_shared_digest:
                    raise ValueError('Live cross-topology rank state differs from validated candidate')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            loaded_digests = _agree('restore:loaded', live_shared_digest, error=error, trainer=trainer)
            if any(item != loaded_digests[0] for item in loaded_digests):
                raise ValueError('Ranks disagree on restored cross-topology shared numerical state')
        except BaseException:
            trainer.checkpoint_ready = False
            trainer._poisoned = True
            raise
        trainer.checkpoint_ready = True
        if return_warnings:
            return Path(response[0]['path']), response[0]['info'], last_batch, tuple(response[0].get('warnings') or ())
        return Path(response[0]['path']), response[0]['info'], last_batch

    local = [None]
    dist.scatter_object_list(local, scatter_object_input_list=payloads, src=0)
    error, state = None, None
    try:
        state = _deserialize(local[0])
        _validate_rank_rng(trainer, state)
        expected_digest = _digest(state)
        # Imported Python modules are singleton dependencies, not mutable owned
        # trainer state (ImageFolder keeps Pillow modules for lazy decoding).
        memo = {id(module): module for module in list(sys.modules.values()) if isinstance(module, types.ModuleType)}
        candidate = copy.deepcopy(trainer, memo)
        candidate_last_batch = restore_trainer(candidate, copy.deepcopy(state))
        if _digest(_rank_state(candidate, candidate_last_batch)) != expected_digest:
            raise ValueError('Restored candidate rank state differs from checkpoint')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        restore_rng(initial_rng)
    _agree('restore:validated', error=error, trainer=trainer)
    trainer.checkpoint_ready = False
    try:
        error, last_batch = None, None
        try:
            # Load hooks may mutate their input. Never pass the canonical state or
            # compare against an expected digest computed after invoking a hook.
            last_batch = restore_trainer(trainer, _deserialize(local[0]))
            if _digest(_rank_state(trainer, last_batch)) != expected_digest:
                raise ValueError('Live restored rank state differs from checkpoint')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        _agree('restore:loaded', error=error, trainer=trainer)
    except BaseException:
        trainer.checkpoint_ready = False
        trainer._poisoned = True
        raise
    trainer.checkpoint_ready = True
    if return_warnings:
        return Path(response[0]['path']), response[0]['info'], last_batch, tuple(response[0].get('warnings') or ())
    return Path(response[0]['path']), response[0]['info'], last_batch
