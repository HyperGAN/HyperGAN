"""Bounded fixed-world-size CPU/CUDA GAN updates with explicit gradient reduction.

This is an internal replicated strategy, not DDP, a launcher or a run service.
The caller owns the default Gloo or NCCL group, finite timeouts and whole-job lifecycle.
"""
import contextlib
import copy
import hashlib
import inspect
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist

from .config import config_values, fingerprint, resolve_config
from .distributed import Collectives
from .recipes import detach, move_tensors
from .training import NOISE_SEED_OFFSET, ReferenceTrainer, ScoredCritic, noise_levels, schedule_learning_rates, update_ema, with_noise
from .objective_program import (
    _bound_scores,
    _clear_discrete_prior_step,
    _prepare_discrete_prior_step,
    _rhbi_generator_context,
    _rhbi_simplex_loss,
    _sum_tensors,
    make_scored_critic,
    score_candidate,
)

DEFAULT_BUCKET_CAP_BYTES = 25 * 1024 * 1024  # 25 MB default bucket size


class _BroadcastProbeProxy:
    """Evaluate a post-step secant probe on rank 0 and broadcast the result bitwise to all ranks.

    Prevents ~1e-6 cuDNN/TF32 nondeterminism on identical probe inputs from causing
    divergent ``alpha_gain`` / ``_ema_func_gain`` updates across CUDA replicas.
    """

    __slots__ = ("_module", "_ref_shapes", "_call_idx", "_device")

    def __init__(self, module, ref_tensor):
        self._module = module
        if isinstance(ref_tensor, (list, tuple)):
            if not ref_tensor:
                raise ValueError("_BroadcastProbeProxy requires at least one reference tensor")
            self._ref_shapes = [tuple(t.shape) for t in ref_tensor]
            self._device = ref_tensor[0].device
        else:
            self._ref_shapes = [tuple(ref_tensor.shape)]
            self._device = ref_tensor.device
        self._call_idx = 0

    def modules(self):
        return self._module.modules()

    def eval(self):
        return self._module.eval()

    def parameters(self, recurse=True):
        return self._module.parameters(recurse=recurse)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __setattr__(self, name, value):
        if name in _BroadcastProbeProxy.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._module, name, value)

    def __delattr__(self, name):
        if name in _BroadcastProbeProxy.__slots__:
            object.__delattr__(self, name)
        else:
            delattr(self._module, name)

    def __call__(self, *args, **kwargs):
        shape = self._ref_shapes[min(self._call_idx, len(self._ref_shapes) - 1)]
        self._call_idx += 1
        if dist.get_rank() == 0:
            out = self._module(*args, **kwargs)
            target = out[0] if isinstance(out, (tuple, list)) and out else out
            buf = target.detach().to(device=self._device, dtype=torch.float32).contiguous()
        else:
            buf = torch.empty(shape, dtype=torch.float32, device=self._device)
        dist.broadcast(buf, src=0)
        return buf


def _bucket_parameters(parameters, bucket_cap_bytes=DEFAULT_BUCKET_CAP_BYTES):
    """Partition parameters into memory-contiguous buckets grouped by (device, dtype)."""
    buckets = []
    current_bucket = []
    current_bytes = 0
    current_key = None

    for p in parameters:
        key = (p.device, p.dtype)
        p_bytes = p.numel() * p.element_size()
        if current_bucket and (key != current_key or (current_bytes + p_bytes > bucket_cap_bytes and current_bytes > 0)):
            buckets.append(current_bucket)
            current_bucket = []
            current_bytes = 0
        current_key = key
        current_bucket.append(p)
        current_bytes += p_bytes

    if current_bucket:
        buckets.append(current_bucket)
    return buckets


class _Fold:
    """Rank-local phases, agreements and gathers that share one exchange."""

    def __init__(self, name):
        self.name, self.entries, self.failed, self.gathered, self.forced = name, [], False, {}, False

    def fail(self, name, exc):
        # The first rank-local error is the cause; later ones follow from it.
        if not self.failed:
            self.failed = True
            self.entries.append(('error', name, f'{type(exc).__name__}: {exc}'))


class ReplicatedTrainer(ReferenceTrainer):
    """One complete D/G/prior/auxiliary/EMA update per global batch.

    ``training.batch_size`` is global; explicit batches and latent draws are local
    shards of that batch. Accumulation partitions each local effective batch. Custom scalar
    objectives use the mean of rank-local objective values; global custom batch
    statistics require their own explicit collective implementation.
    """

    def __init__(self, config, *, world_size=2, accumulation_steps=1,
                 nodes=1, node_rank=0, ranks_per_node=None, rendezvous=None,
                 aggregation=None, execution=None, **kwargs):
        if execution is not None:
            world_size = execution.get('world_size', world_size)
            accumulation_steps = execution.get('accumulation_steps', accumulation_steps)
            nodes = execution.get('nodes', nodes)
            node_rank = execution.get('node_rank', node_rank)
            ranks_per_node = execution.get('ranks_per_node', ranks_per_node)
            rendezvous = execution.get('rendezvous', rendezvous)
            if 'aggregation' in execution:
                aggregation = execution['aggregation']
        if ranks_per_node is None:
            ranks_per_node = world_size // nodes
        if rendezvous is None:
            if nodes > 1:
                raise ValueError('Experimental multi-node execution requires an explicit rendezvous '
                                 '(tcp://<node-0-host>:<port>); loopback cannot reach other hosts')
            rendezvous = 'file://'
        if aggregation is None:
            aggregation = {'strategy': 'gradient_allreduce', 'sync_every_steps': 1,
                           'outer_lr': 1.0, 'outer_momentum': 0.0}
        else:
            aggregation = dict(aggregation)
        self.collectives = Collectives(world_size)
        self.device = self.collectives.device
        self.world_size, self.rank = world_size, dist.get_rank()
        self.nodes = nodes
        self.node_rank = node_rank
        self.ranks_per_node = ranks_per_node
        self.rendezvous = rendezvous
        self.aggregation = aggregation
        self.checkpoint_ready = False
        self._poisoned = False
        config = self._phase('configuration', lambda: resolve_config(config_values(config)))
        from .execution_profiles import validate_replicated_recipe
        self._phase('recipe policy', lambda: validate_replicated_recipe(config, accumulation_steps=accumulation_steps))
        self._agree('configuration', {'fingerprint': fingerprint(config), 'accumulation_steps': accumulation_steps})
        if world_size < 2:
            raise ValueError('Replicated training requires at least two ranks')
        def validate_accumulation_control():
            if type(accumulation_steps) is not int or accumulation_steps < 1:
                raise ValueError('accumulation_steps must be a positive integer')
        self._phase('accumulation control', validate_accumulation_control)
        def validate_device():
            expected = 'cuda' if self.device.type == 'cuda' else 'cpu'
            if config['training']['device'] != expected:
                raise ValueError(f"Replicated {self.collectives.backend} training requires training.device={expected!r}; CUDA indices are rank-owned")
        self._phase('execution device', validate_device)
        self.global_batch_size = config['training']['batch_size']
        if self.global_batch_size % world_size:
            raise ValueError('Global training.batch_size must divide evenly across world_size')
        self.local_batch_size = self.global_batch_size // world_size
        if self.local_batch_size % accumulation_steps:
            raise ValueError('accumulation_steps must divide local_batch_size evenly')
        self.accumulation_steps = accumulation_steps
        self.microbatch_size = self.local_batch_size // accumulation_steps
        self.strategy_info = {
            'name': 'cuda-replicated-nccl' if self.device.type == 'cuda' else 'cpu-replicated-gloo',
            'world_size': world_size,
            'nodes': nodes,
            'node_rank': node_rank,
            'ranks_per_node': ranks_per_node,
            'rendezvous': rendezvous,
            'accumulation_steps': accumulation_steps,
            'global_batch_size': self.global_batch_size,
            'local_batch_size': self.local_batch_size,
            'microbatch_size': self.microbatch_size,
            'accumulation_algorithm': ('detached-logit-vjp-replay-v1' if accumulation_steps > 1
                                       else 'retained-local-graph-v1'),
            'aggregation': aggregation,
            'gradient_reduction': 'post-backward-mean',
            'buffers': 'require-replica-equality',
            'data': 'replicated-global-draw-rank-slice',
            'qualification': 'unqualified',
        }
        torch.set_num_threads(1)
        def initialize():
            # Keep the shared recipe identity unchanged across ranks. Only native
            # construction receives the concrete rank-owned device selection.
            local_config = copy.deepcopy(config)
            local_config['training']['device'] = str(self.device)
            if 'device' in local_config['prior']['args']:
                local_config['prior']['args']['device'] = str(self.device)
            super(ReplicatedTrainer, self).__init__(local_config)
            self.config = config
        self._phase('initialization', initialize)
        from .aggregation import resolve_aggregation_strategy
        self.aggregation_strategy = resolve_aggregation_strategy(self.strategy_info['aggregation'], trainer=self)
        # Every rank must take the same replica-check collectives.
        self._agree('replica check interval', getattr(self.aggregation_strategy, 'replica_check_every', None))
        self._phase('module compatibility', self._check_modules)
        if accumulation_steps > 1:
            self._phase('accumulation compatibility', self._check_accumulation)
        self._assert_replicas('initial state')
        # Initialize identical models, then give each worker distinct reproducible
        # random streams. Controlled fixtures can supply matched explicit draws.
        rank_seed = (config['training']['seed'] + 104729 * self.rank) % (2 ** 63)
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        settings = config['training']
        data_offset = settings.get('data_seed_offset', 1)
        prior_offset = settings.get('prior_seed_offset', 2)
        self.streams['data'].manual_seed((config['training']['seed'] + data_offset) % (2 ** 63))
        self.streams['prior'].manual_seed((rank_seed + prior_offset) % (2 ** 63))
        self.streams['penalty'].manual_seed((rank_seed + 3) % (2 ** 63))
        self.streams['noise'].manual_seed((rank_seed + NOISE_SEED_OFFSET) % (2 ** 63))
        self._phase('initial CUDA completion', self._synchronize)
        self.checkpoint_ready = True
        self._handshake_complete = False
        self._bucket_cap_bytes = DEFAULT_BUCKET_CAP_BYTES

    @property
    def _in_hot_path(self) -> bool:
        if getattr(self, '_force_barriers', False):
            return False
        if os.environ.get('HYPERGAN_DISTRIBUTED_BARRIERS') == '1':
            return False
        return getattr(self, 'step', 0) >= 1 and getattr(self, '_handshake_complete', False)

    def _synchronize(self, force=False):
        if not force and self._in_hot_path:
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _exchange(self, operation, value):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        payload = {'operation': operation, 'value': value}
        peers = [None] * self.world_size
        dist.all_gather_object(peers, payload)
        operations = [peer.get('operation') if isinstance(peer, dict) else None for peer in peers]
        if any(item != operations[0] for item in operations):
            raise ValueError(f'Ranks entered different trainer operations: {operations}')
        return [peer['value'] for peer in peers]

    _fold = None

    @contextlib.contextmanager
    def _folded(self, name):
        """Agree on every phase, agreement and gather in the block with one exchange.

        Phases run rank-locally and the exchange at the end of the block raises
        the earliest failure identically on every rank. A rank that failed skips
        the block's remaining phases, and any other exception unwinds to the
        exchange, so no rank is left waiting. Only the block's first phase may
        run collectives: a rank that failed earlier would skip them.
        """
        fold = self._fold = _Fold(name)
        try:
            yield fold
        except BaseException as exc:
            fold.fail(name, exc)
        finally:
            self._fold = None
        self._settle(fold)

    def _hot_path_agree_ok(self, payload, *, has_error=False):
        """Check cross-rank agreement and error status via one int64 MAX all_reduce (zero all_gather_object)."""
        if getattr(self, 'device', None) is not None and self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        unhashable = False
        words = (0,) * 8
        if not has_error:
            try:
                raw = bytes.fromhex(self._digest(payload))
                words = tuple(int.from_bytes(raw[i:i + 4], 'big') for i in range(0, 32, 4))
            except ValueError:
                unhashable = True
        buf_vals = [int(bool(has_error or unhashable))]
        for w in words:
            buf_vals.append(w)
            buf_vals.append(-w)
        buf = torch.tensor(buf_vals, dtype=torch.int64, device=self.device)
        dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        out = buf.tolist()
        if out[0]:
            return False
        for i in range(8):
            if out[1 + 2 * i] + out[2 + 2 * i] != 0:
                return False
        return True

    def _settle(self, fold, force=False):
        if not force and not getattr(fold, 'forced', False) and getattr(self, '_in_hot_path', False):
            has_error = fold.failed or any(entry[0] == 'error' for entry in fold.entries)
            if self._hot_path_agree_ok((fold.name, fold.entries), has_error=has_error):
                return
        peers = self._exchange('fold:' + fold.name, fold.entries)
        for index in range(max(len(entries) for entries in peers)):
            rows = [entries[index] if index < len(entries) else None for entries in peers]
            errors = [(rank, row) for rank, row in enumerate(rows) if row is not None and row[0] == 'error']
            if errors:
                names = ' / '.join(dict.fromkeys(row[1] for _, row in errors))
                raise RuntimeError(f'{names} failed: ' + '; '.join(f'rank {rank}: {row[2]}' for rank, row in errors))
            if any(row is None or row[:2] != rows[0][:2] for row in rows):
                raise ValueError(f'Ranks entered different trainer operations: {[row and row[1] for row in rows]}')
            kind, name, value = rows[0]
            if kind == 'agreement' and any(row[2] != value for row in rows):
                raise ValueError(f'Ranks disagree on {name}')
            if kind == 'gather':
                fold.gathered[name] = [row[2] for row in rows]

    def _phase(self, name, function, force=False, *, gather=False):
        fold = self._fold
        if fold is not None:
            if force or gather:
                fold.forced = True
            if fold.failed:
                return None
            try:
                result = function()
            except BaseException as exc:
                fold.fail(name, exc)
                return None
            fold.entries.append(('gather', name, result) if gather else ('phase', name, None))
            return result
        result, error = None, None
        try:
            result = function()
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        if not force and not gather and getattr(self, '_in_hot_path', False):
            if getattr(self, 'device', None) is not None and self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            failed = torch.tensor(int(error is not None), dtype=torch.int32, device=self.device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if not failed.item():
                return result
        errors = self._exchange('phase:' + name, error)
        if any(errors):
            raise RuntimeError(f'{name} failed: ' + '; '.join(f'rank {rank}: {value}' for rank, value in enumerate(errors) if value))
        return result

    def _agree(self, name, value, force=False):
        fold = self._fold
        if fold is not None:
            if force:
                fold.forced = True
            if not fold.failed:
                fold.entries.append(('agreement', name, value))
            return
        if not force and getattr(self, '_in_hot_path', False):
            if self._hot_path_agree_ok(('agreement:' + name, value), has_error=False):
                return
        values = self._exchange('agreement:' + name, value)
        if any(item != values[0] for item in values):
            raise ValueError(f'Ranks disagree on {name}')

    def _check_modules(self):
        for name, module in self.graph.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                raise ValueError(f'Training-mode BatchNorm ({name}) has rank-local statistics; this global-batch strategy requires a qualified alternative or frozen evaluation mode')
        for module in (self.graph, self.prior, self.ema_graph, self.ema_prior):
            if any(value.device != self.device or value.layout != torch.strided for value in (*module.parameters(), *module.buffers())):
                raise ValueError('Replicated modules must own dense parameters and buffers on the rank device')
            if any(not torch.isfinite(value).all() for value in module.parameters()):
                raise ValueError('Replicated parameters must be finite')

    @staticmethod
    def _digest(value):
        digest = hashlib.sha256()
        def visit(item):
            if isinstance(item, torch.Tensor):
                digest.update(f'tensor:{item.dtype}:{tuple(item.shape)}'.encode())
                digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, dict):
                digest.update(b'dict')
                for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                    visit(key)
                    visit(item[key])
            elif isinstance(item, (tuple, list)):
                digest.update(type(item).__name__.encode())
                for child in item:
                    visit(child)
            elif item is None or type(item) in (bool, int, float, str):
                digest.update((type(item).__name__ + ':' + repr(item)).encode())
            else:
                raise ValueError('Replicated state must contain tensors and primitive containers')
            digest.update(b'\0')
        visit(value)
        return digest.hexdigest()

    def _replica_digest(self):
        from .checkpoints import capture_rng, restore_rng
        rng = capture_rng()
        try:
            state = {}
            for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
                module = getattr(self, name)
                state[name] = {'state': module.state_dict(), 'buffers': dict(module.named_buffers()),
                               'trainable': {key: p.requires_grad for key, p in module.named_parameters()},
                               'modes': {key: child.training for key, child in module.named_modules()}}
            state['optimizers'] = [optimizer.state_dict() for optimizer in self.optimizers]
            state['base_lrs'] = self.base_lrs
            strategy = getattr(self, 'aggregation_strategy', None)
            if strategy is not None and callable(getattr(strategy, 'state_dict', None)):
                state['aggregation'] = strategy.state_dict()
            prior_reg_ckpt = self.recipe.prior_regularizer_checkpoint(self.spread) if callable(getattr(self.recipe, 'prior_regularizer_checkpoint', None)) else {}
            if prior_reg_ckpt:
                state['prior_regularizer'] = prior_reg_ckpt['prior_regularizer']
            return self._digest(state)
        finally:
            restore_rng(rng)

    def _assert_replicas(self, name):
        self._agree(name, self._phase(name + ' validation', self._replica_digest, force=True), force=True)

    def synchronize_replicas(self):
        """Force one local-SGD synchronization so a checkpoint has one replicated state.

        Between synchronizations parameter_average/pseudo_gradient replicas differ.
        A checkpoint at such a step averages them first, which is one extra
        synchronization in the trajectory; checkpoint_every is validated against
        sync_every_steps so this only happens for stop and on-demand checkpoints.
        """
        strategy = self.aggregation_strategy
        if not callable(getattr(strategy, 'is_synchronized', None)) or not callable(getattr(strategy, 'synchronize', None)):
            return False
        def boundary():
            # A half/failed update is not synchronized here; the checkpoint itself rejects it.
            return bool(self._poisoned or not self.checkpoint_ready or strategy.is_synchronized(self.step))
        with self._folded('aggregation state'):
            synchronized = self._phase('aggregation state', boundary, force=True)
            self._agree('aggregation state', synchronized, force=True)
        if synchronized:
            return False
        self.checkpoint_ready = False
        try:
            with self._folded('aggregation synchronization'):
                # The synchronization's collectives come first, before any rank can have failed.
                self._phase('aggregation synchronization', strategy.synchronize)
                self._phase('synchronized CUDA completion', self._synchronize)
                self._assert_replicas('synchronized replicated state')
        except BaseException:
            self._poisoned = True
            raise
        self.checkpoint_ready = True
        return True

    def batch(self):
        def shard(value):
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    return value.clone()
                if len(value) != self.global_batch_size:
                    raise ValueError('Sampled data tensors must share the global batch dimension')
                start = self.rank * self.local_batch_size
                return value[start:start + self.local_batch_size].clone()
            if isinstance(value, dict):
                return {key: shard(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(shard(item) for item in value)
            return value

        def draw():
            if self._data_shards is None:
                self._data_shards = self._accepts_shard(self.data)
            if self._data_shards:
                return self.data(self.global_batch_size, generator=self.streams['data'], rank=self.rank, world_size=self.world_size)
            return self.data(self.global_batch_size, generator=self.streams['data'])

        data_draw = self._phase('data draw', draw)
        if isinstance(data_draw, dict) and isinstance(data_draw.get('real'), torch.Tensor):
            if len(data_draw['real']) == self.local_batch_size:
                batch = move_tensors(data_draw, self.device)
            elif len(data_draw['real']) == self.global_batch_size:
                batch = self._phase('global data slicing', lambda: move_tensors(shard(data_draw), self.device))
            else:
                raise ValueError(f"Data returned 'real' tensor with length {len(data_draw['real'])}; expected {self.local_batch_size} or {self.global_batch_size}")
        else:
            raise ValueError("Data must return a dictionary containing tensor 'real'")

        self._phase('local data validation', lambda: self._validate_batch(batch))
        return batch

    _data_shards = None

    @staticmethod
    def _accepts_shard(source):
        """Pass rank/world_size only to data sources that declare them."""
        try:
            parameters = inspect.signature(source).parameters.values()
        except (TypeError, ValueError):
            return False
        names = {parameter.name for parameter in parameters}
        return ({'rank', 'world_size'} <= names
                or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters))

    def _validate_batch(self, batch):
        if not isinstance(batch, dict) or not isinstance(batch.get('real'), torch.Tensor):
            raise ValueError("Data must return a dictionary containing tensor 'real'")
        real = batch['real']
        if real.device != self.device or not real.is_floating_point() or real.ndim < 1 or len(real) != self.local_batch_size:
            raise ValueError('Real data must be floating tensors on the rank device of local_batch_size')
        if not torch.isfinite(real).all():
            raise ValueError('Nonfinite real data')

    def _parameters(self, optimizer):
        return [parameter for group in optimizer.param_groups for parameter in group['params']]

    def _unscale(self, device):
        if not hasattr(self, '_unscale_scalars'):
            self._unscale_scalars = {}
        if device not in self._unscale_scalars:
            self._unscale_scalars[device] = torch.ones((), dtype=torch.float32, device=device)
        return self._unscale_scalars[device]

    def _gradient_flags(self, parameters):
        fused, flags = {}, []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_floating_point() and gradient.layout == torch.strided:
                fused.setdefault((gradient.device, gradient.dtype), []).append(gradient)
            else:
                flags.append(torch.isfinite(gradient).all().logical_not().to(device=self.device, dtype=torch.int32))
        for (device, _), gradients in fused.items():
            found = torch.zeros((), dtype=torch.float32, device=device)
            torch._amp_foreach_non_finite_check_and_unscale_(gradients, found, self._unscale(device))
            flags.append(found.to(device=self.device, dtype=torch.int32))
        return flags

    def _reduce_gradients(self, optimizer, name):
        self.aggregation_strategy.on_backward(optimizer, name)
        if not getattr(self.aggregation_strategy, 'reduces_gradients', False):
            return
        parameters = self._parameters(optimizer)
        present = [p.grad is not None for p in parameters]
        sparse = any(p.grad.layout != torch.strided for p in parameters if p.grad is not None)

        found_flags = self._gradient_flags(parameters)
        local_nonfinite = torch.tensor(1, dtype=torch.int32, device=self.device) if sparse else (
            torch.stack(found_flags).sum().clamp(max=1) if found_flags else torch.zeros((), dtype=torch.int32, device=self.device)
        )

        control = torch.empty(len(parameters) + 1, dtype=torch.int32, device=self.device)
        control[0] = local_nonfinite
        control[1:] = torch.tensor(present, dtype=torch.int32, device=self.device)
        dist.all_reduce(control, op=dist.ReduceOp.SUM)

        status = control.tolist()
        if status[0] > 0:
            for p in parameters:
                p.grad = None
            raise ValueError('Nonfinite or sparse gradient; optimizer step refused')

        active_params = []
        for count, parameter in zip(status[1:], parameters):
            if count:
                active_params.append(parameter)
            else:
                parameter.grad = None

        if active_params:
            bucket_cap = getattr(self, '_bucket_cap_bytes', DEFAULT_BUCKET_CAP_BYTES)
            buckets = _bucket_parameters(active_params, bucket_cap_bytes=bucket_cap)
            for bucket in buckets:
                total_numel = sum(p.numel() for p in bucket)
                dtype = bucket[0].dtype
                device = bucket[0].device
                bucket_buffer = torch.empty(total_numel, dtype=dtype, device=device)

                # Pack gradients into contiguous bucket
                offset = 0
                for p in bucket:
                    numel = p.numel()
                    dest_slice = bucket_buffer[offset : offset + numel].view(p.shape)
                    if p.grad is None:
                        dest_slice.zero_()
                    else:
                        dest_slice.copy_(p.grad)
                    offset += numel

                # Coalesced collective communication
                dist.all_reduce(bucket_buffer, op=dist.ReduceOp.SUM)
                bucket_buffer.div_(self.world_size)

                # Unpack reduced gradients back to parameters
                offset = 0
                for p in bucket:
                    numel = p.numel()
                    dest_slice = bucket_buffer[offset : offset + numel].view(p.shape)
                    if p.grad is None:
                        p.grad = dest_slice.clone()
                    else:
                        p.grad.copy_(dest_slice)
                    offset += numel

        # Finite local gradients can overflow in SUM before the world-size
        # division. Every replica receives the same reduction, so reject it
        # before Adam can mutate parameters or moments.
        reduced_flags = self._gradient_flags(parameters)
        if reduced_flags and torch.stack(reduced_flags).sum().item():
            for parameter in parameters:
                parameter.grad = None
            raise ValueError('Nonfinite reduced gradient; optimizer step refused')

    def _step_critics(self, terms):
        for name, optimizer in self.critic_optimizers.items():
            self._reduce_gradients(optimizer, name)
        critic_params = self._critic_checked_parameters(terms)
        critic_grad_norm_t = self._grad_norm_scalar(critic_params)
        explicit_guard = getattr(self.opt_d, 'd_guard_ratio', None)
        for name, optimizer in self.critic_optimizers.items():
            rec_d = getattr(optimizer, 'record', None)
            if rec_d is not None and getattr(self.aggregation_strategy, 'reduces_gradients', False):
                if hasattr(rec_d, 'proto_initialized'):
                    from .aggregation import sync_empf_record
                    sync_empf_record(rec_d, self.device, self.world_size)
                elif hasattr(rec_d, 't1c_ell_initialized'):
                    from .aggregation import sync_unified_v11_record
                    sync_unified_v11_record(rec_d, self.device, self.world_size)
                elif hasattr(rec_d, 'chart_state') and hasattr(rec_d, 'thermal_scale'):
                    from .aggregation import sync_axiom_record
                    sync_axiom_record(rec_d, self.device, self.world_size)
        thermal_coordinator = getattr(self, 'thermal_coordinator', None)
        if thermal_coordinator is not None:
            thermal_coordinator.synchronize()
        for name, optimizer in self.critic_optimizers.items():
            rec_d = getattr(optimizer, 'record', None)
            self._sync_unified_v11_d_probes(rec_d)
            probes_d = getattr(rec_d, '_d_probe_list', None)
            orig_critic = getattr(optimizer, 'critic', None)
            wrap_critic = (
                bool(probes_d)
                and orig_critic is not None
                and self.world_size > 1
                and getattr(self.aggregation_strategy, 'reduces_gradients', False)
                and all(isinstance(p[2], torch.Tensor) for p in probes_d)
            )
            if wrap_critic:
                optimizer.critic = _BroadcastProbeProxy(orig_critic, [p[2] for p in probes_d])
            try:
                self._optimizer_step(optimizer, name)
            finally:
                if wrap_critic:
                    optimizer.critic = orig_critic
            self._average_critic_statistics(rec_d)
        if explicit_guard is not None:
            d_guard_ratio_t = torch.tensor(float(explicit_guard or 1.0), dtype=torch.float64, device=self.device)
        elif getattr(self.opt_d, 'guard', None) is not None:
            post_critic_grad_norm_t = self._grad_norm_scalar(critic_params)
            d_guard_ratio_t = torch.where(
                critic_grad_norm_t > 0.0,
                post_critic_grad_norm_t / critic_grad_norm_t.clamp_min(1e-30),
                torch.ones_like(critic_grad_norm_t),
            )
        else:
            d_guard_ratio_t = torch.ones((), dtype=torch.float64, device=self.device)
        return critic_grad_norm_t, d_guard_ratio_t

    def _average_critic_statistics(self, rec):
        # KA2 controller statistics come from rank-local critic scores.
        # t1c / unified_v11 / axiom records expose ema_residual as a read-only property
        # and sync via coupled pairs or sync_unified_v11_record / sync_axiom_record.
        if (
            rec is None
            or hasattr(rec, 't1c_ell_initialized')
            or (hasattr(rec, 'chart_state') and hasattr(rec, 'thermal_scale'))
        ):
            return
        if hasattr(rec, 'ema_residual') and hasattr(rec, 'ema_data_scale'):
            stats = torch.tensor([rec.ema_residual, rec.ema_data_scale], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats)
            stats /= self.world_size
            rec.ema_residual = float(stats[0].item())
            rec.ema_data_scale = float(stats[1].item())

    def _finite_optimizer(self, optimizer):
        for parameter in self._parameters(optimizer):
            if not torch.isfinite(parameter).all():
                raise ValueError('Optimizer produced nonfinite parameters')
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError('Optimizer produced nonfinite state')

    def _optimizer_step(self, optimizer, name):
        with self._folded(name + ' optimizer'):
            self._phase(name + ' optimizer', optimizer.step)
            self._phase(name + ' optimizer state', lambda: self._finite_optimizer(optimizer))

    def _logits(self, real, fake):
        # Row-separable losses use rank-local logits; coupled Sinkhorn/RHBI losses
        # gather global pairs via _gather_rhbi_pair before loss evaluation.
        return real, fake

    def _all_gather_coupled(self, tensor):
        """Gather ``[B_local, ...]`` across ranks into ``[B_global, ...]``, preserving local autograd."""
        detached = tensor.detach().contiguous()
        shards = [torch.empty_like(detached) for _ in range(self.world_size)]
        dist.all_gather(shards, detached)
        if tensor.requires_grad:
            shards[self.rank] = tensor
        return torch.cat(shards, dim=0)

    def _scale_coupled_backward(self, loss):
        """Preserve global forward loss value while scaling local backward cotangent for mean all-reduce."""
        if loss.requires_grad and self.world_size > 1:
            return loss * self.world_size + loss.detach() * (1.0 - self.world_size)
        return loss

    @staticmethod
    def _term_real_sample(phase, context):
        path = phase.real.path
        if path == 'generated':
            return context.get('generated')
        if path == 'latent':
            return context.get('latent')
        if path.startswith('batch.'):
            batch = context.get('batch')
            if isinstance(batch, dict):
                return batch.get(path.split('.', 1)[1])
        components = context.get('components')
        if isinstance(components, dict):
            key = path.split('.', 1)[1] if path.startswith('components.') else path
            if key in components:
                return components[key]
        return None

    @staticmethod
    def _is_unified_v15_record(rec):
        return rec is not None and getattr(type(rec), '__module__', '').startswith('particlegan._unified_v15')

    @classmethod
    def _is_unified_v15_term(cls, term):
        gan = getattr(term, 'gan', None)
        if gan is None:
            return False
        return (
            cls._is_unified_v15_record(getattr(gan, 'record', None))
            or getattr(type(gan), '__module__', '').startswith('particlegan._unified_v15')
            or type(gan).__name__ == 'UnifiedV15GANLoss'
        )

    @classmethod
    def _is_highres_image_term(cls, term, sample):
        if not (isinstance(sample, torch.Tensor) and sample.ndim >= 2 and sample.shape[0] > 0 and len(term.routes) == 1):
            return False
        is_v15_or_axiom = hasattr(term.gan, 'axiom_record') or cls._is_unified_v15_term(term)
        limit = (4096 if sample.ndim >= 4 else 1024) if is_v15_or_axiom else 1024
        return int(sample[0].numel()) > limit

    @classmethod
    def _uses_coupled_critic_loss(cls, term, sample=None):
        if hasattr(term.gan, 'empf_record'):
            return True
        if hasattr(term.gan, 'rhbi') and (
            getattr(term.gan, 'critic_weight', 0.0) > 0.0
            or getattr(term.gan, 'fibre_rel_blend', 0.0) > 0.0
        ):
            return True
        if hasattr(term.gan, 't1c') and not cls._is_highres_image_term(term, sample):
            return bool(
                getattr(term.gan, 'critic_weight', 0.0) > 0.0
                or getattr(term.gan, 'fibre_rel_blend', 0.0) > 0.0
                or hasattr(getattr(term.gan, 'record', None), 'observe_margin_and_scale')
            )
        return False

    @classmethod
    def _uses_coupled_generator_loss(cls, term, sample=None):
        if hasattr(term.gan, 'empf_record'):
            return True
        if hasattr(term.gan, 'rhbi') and getattr(term.gan.rhbi, 'weight', 0.0) > 0.0:
            return True
        if hasattr(term.gan, 't1c') and getattr(term.gan.t1c, 'weight', 0.0) > 0.0 and not cls._is_highres_image_term(term, sample):
            return True
        return False

    def _reduce_protected(self, optimizer):
        protected = getattr(optimizer, '_protected', None)
        if protected is None or not getattr(self.aggregation_strategy, 'reduces_gradients', False):
            return
        if isinstance(protected, torch.Tensor):
            dist.all_reduce(protected)
            protected /= self.world_size
            return
        if isinstance(protected, (list, tuple)):
            for group in protected:
                if isinstance(group, torch.Tensor):
                    dist.all_reduce(group)
                    group /= self.world_size
                elif isinstance(group, (list, tuple)):
                    for tensor in group:
                        if isinstance(tensor, torch.Tensor):
                            dist.all_reduce(tensor)
                            tensor /= self.world_size

    def _gather_axiom_pair(self, fake_score, real_score, rec):
        """Gather critic logits and Axiom v6 CaptureEntry metadata across all ranks."""
        from particlegan._axiom.state import (
            CaptureEntry,
            lookup_axiom_capture,
            record_axiom_capture,
        )

        ef = lookup_axiom_capture(fake_score, rec)
        hint_rec = ef.record if (ef is not None and ef.record is not None) else rec
        er = lookup_axiom_capture(real_score, hint_rec)
        eff_rec = hint_rec if hint_rec is not None else (er.record if er is not None else None)
        global_f_sc = self._all_gather_coupled(fake_score)
        global_r_sc = self._all_gather_coupled(real_score)
        if ef is not None and er is not None:
            global_x_f = self._all_gather_coupled(ef.x_in)
            global_x_r = self._all_gather_coupled(er.x_in)
            global_cond_f = self._all_gather_coupled(ef.cond) if ef.cond is not None else None
            global_cond_r = self._all_gather_coupled(er.cond) if er.cond is not None else None
            global_parts_f = tuple(self._all_gather_coupled(p) for p in ef.parts) if ef.parts is not None else None
            global_parts_r = tuple(self._all_gather_coupled(p) for p in er.parts) if er.parts is not None else None
            record_axiom_capture(
                global_f_sc,
                CaptureEntry(
                    global_x_f,
                    global_cond_f,
                    eff_rec,
                    ef.has_cont_cond,
                    global_parts_f,
                    ef.thermally_scaled,
                ),
            )
            record_axiom_capture(
                global_r_sc,
                CaptureEntry(
                    global_x_r,
                    global_cond_r,
                    eff_rec,
                    er.has_cont_cond,
                    global_parts_r,
                    er.thermally_scaled,
                ),
            )
        return global_f_sc, global_r_sc

    def _gather_t1c_pair(self, fake_score, real_score, rec):
        """Gather critic logits and t1c/unified_v11 input/conditioning captures across all ranks."""
        from particlegan._t1_coupling.critic import _lookup_t1c_capture_parts

        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_t1c_capture_parts(fake_score, rec)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_t1c_capture_parts(
            real_score, rec_f if rec_f is not None else rec
        )
        eff_rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else rec)
        global_f_sc = self._all_gather_coupled(fake_score)
        global_r_sc = self._all_gather_coupled(real_score)
        if x_f is not None and x_r is not None:
            global_x_f = self._all_gather_coupled(x_f)
            global_x_r = self._all_gather_coupled(x_r)
            global_cond_f = self._all_gather_coupled(cond_f) if cond_f is not None else None
            global_cond_r = self._all_gather_coupled(cond_r) if cond_r is not None else None
            global_parts_f = tuple(self._all_gather_coupled(p) for p in parts_f) if parts_f is not None else None
            global_parts_r = tuple(self._all_gather_coupled(p) for p in parts_r) if parts_r is not None else None
            global_f_sc._t1c_input = global_x_f
            global_f_sc._t1c_cond = global_cond_f
            global_f_sc._t1c_cont_cond = cc_f
            global_f_sc._t1c_parts = global_parts_f
            global_f_sc._t1c_record = eff_rec
            if getattr(fake_score, '_t1c_thermally_scaled', False):
                global_f_sc._t1c_thermally_scaled = True
            global_r_sc._t1c_input = global_x_r
            global_r_sc._t1c_cond = global_cond_r
            global_r_sc._t1c_cont_cond = cc_r
            global_r_sc._t1c_parts = global_parts_r
            global_r_sc._t1c_record = eff_rec
        return global_f_sc, global_r_sc

    def _gather_empf_pair(self, fake_score, real_score, rec, *, fake_sample=None, real_sample=None):
        """Gather critic logits and EMPF spatial/conditioning captures across all ranks."""
        global_f_sc = self._all_gather_coupled(fake_score)
        global_r_sc = self._all_gather_coupled(real_score)
        eff_rec = (
            getattr(fake_score, '_empf_record', None)
            or getattr(real_score, '_empf_record', None)
            or rec
        )
        b_f_local = int(fake_score.shape[0]) if fake_score.ndim >= 1 else 0
        b_r_local = int(real_score.shape[0]) if real_score.ndim >= 1 else 0

        def _gather_cond_tuple(cond_tensors, b_local):
            if not cond_tensors:
                return ()
            out = []
            for c in cond_tensors:
                if isinstance(c, torch.Tensor) and c.ndim >= 1 and int(c.shape[0]) == b_local:
                    out.append(self._all_gather_coupled(c.detach()))
                else:
                    out.append(c)
            return tuple(out)

        x_f = getattr(fake_score, '_empf_input', None)
        x_r = getattr(real_score, '_empf_input', None)
        if x_f is None and isinstance(fake_sample, torch.Tensor) and x_r is not None and fake_sample.shape == x_r.shape:
            x_f = fake_sample.detach()
        if x_r is None and isinstance(real_sample, torch.Tensor) and x_f is not None and real_sample.shape == x_f.shape:
            x_r = real_sample.detach()
        if isinstance(x_f, torch.Tensor) and isinstance(x_r, torch.Tensor):
            global_f_sc._empf_input = self._all_gather_coupled(x_f.detach())
            global_r_sc._empf_input = self._all_gather_coupled(x_r.detach())
        if eff_rec is not None:
            global_f_sc._empf_record = eff_rec
            global_r_sc._empf_record = eff_rec
        if hasattr(fake_score, '_empf_conditioned'):
            global_f_sc._empf_conditioned = bool(fake_score._empf_conditioned)
        if hasattr(real_score, '_empf_conditioned'):
            global_r_sc._empf_conditioned = bool(real_score._empf_conditioned)
        c_f = getattr(fake_score, '_empf_cond_tensors', ())
        c_r = getattr(real_score, '_empf_cond_tensors', ())
        if c_f:
            global_f_sc._empf_cond_tensors = _gather_cond_tuple(c_f, b_f_local)
        if c_r:
            global_r_sc._empf_cond_tensors = _gather_cond_tuple(c_r, b_r_local)
        return global_f_sc, global_r_sc

    def _gather_coupled_pair(self, term, fake_score, real_score, *, fake_sample=None, real_sample=None):
        rec = getattr(term.gan, 'record', None)
        if hasattr(term.gan, 'empf_record'):
            return self._gather_empf_pair(
                fake_score,
                real_score,
                getattr(term.gan, 'empf_record', None) or rec,
                fake_sample=fake_sample,
                real_sample=real_sample,
            )
        if hasattr(term.gan, 'axiom_record'):
            return self._gather_axiom_pair(fake_score, real_score, rec)
        if hasattr(term.gan, 't1c'):
            return self._gather_t1c_pair(fake_score, real_score, rec)
        return self._gather_rhbi_pair(fake_score, real_score, rec)

    def _sync_coupled_generator_latents(self, terms):
        """Gather v15 generator latents across ranks so highd_basin_bank and sparse_obs_bank match global x_f."""
        if self.world_size <= 1 or not getattr(self.aggregation_strategy, 'reduces_gradients', False):
            return
        seen = set()
        for term in terms:
            rec = getattr(term.gan, 'record', None)
            if rec is None or id(rec) in seen:
                continue
            seen.add(id(rec))
            g_lat = getattr(rec, '_g_last_latent', None)
            if (
                isinstance(g_lat, tuple)
                and len(g_lat) == 2
                and isinstance(g_lat[1], torch.Tensor)
                and g_lat[1].ndim >= 1
                and int(g_lat[1].shape[0]) == self.local_batch_size
            ):
                rec._g_last_latent = (g_lat[0], self._all_gather_coupled(g_lat[1].detach()))
            g_lat_live = getattr(rec, '_g_last_latent_live', None)
            if (
                isinstance(g_lat_live, tuple)
                and len(g_lat_live) == 2
                and isinstance(g_lat_live[1], torch.Tensor)
                and g_lat_live[1].ndim >= 1
                and int(g_lat_live[1].shape[0]) == self.local_batch_size
            ):
                rec._g_last_latent_live = (g_lat_live[0], self._all_gather_coupled(g_lat_live[1]))

    def _probe_budget_for_record(self, rec, b_global, elem_dim):
        if hasattr(rec, 'chart_state') and hasattr(rec, 'thermal_scale'):
            return min(b_global, max(1, min(32, 1024 // max(1, elem_dim))))
        if self._is_unified_v15_record(rec):
            from particlegan._unified_v15.critic import _probe_batch_count
            return _probe_batch_count(b_global, elem_dim)
        from particlegan._unified_v11.critic import _PROBE_SLICE
        return min(b_global, max(2, min(_PROBE_SLICE, 8192 // max(1, elem_dim))))

    def _gather_probe_value(self, value, b_local, b_probe, p_idx=None):
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and int(value.shape[0]) == b_local:
                gathered = self._all_gather_coupled(value.detach())
                if p_idx is not None:
                    return gathered.index_select(0, p_idx.to(device=gathered.device))
                return gathered[:b_probe]
            return value.detach()
        if isinstance(value, dict):
            return {k: self._gather_probe_value(v, b_local, b_probe, p_idx=p_idx) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._gather_probe_value(v, b_local, b_probe, p_idx=p_idx) for v in value)
        if isinstance(value, list):
            return [self._gather_probe_value(v, b_local, b_probe, p_idx=p_idx) for v in value]
        return value

    def _sync_unified_v11_d_probes(self, rec):
        """Gather critic secant probe slices across ranks so all ranks compute identical alpha_gain."""
        probes = getattr(rec, '_d_probe_list', None)
        if not probes or self.world_size <= 1 or not getattr(self.aggregation_strategy, 'reduces_gradients', False):
            return
        is_v15 = self._is_unified_v15_record(rec)
        select_v15 = None
        if is_v15:
            from particlegan._unified_v15.critic import _select_probe_indices as select_v15
        synced = []
        for p_args, p_kwargs, d_pre in probes:
            if isinstance(d_pre, torch.Tensor) and d_pre.ndim >= 1 and d_pre.shape[0] > 0:
                b_local = int(d_pre.shape[0])
                b_gathered = b_local * self.world_size
                first_t = next((a for a in p_args if isinstance(a, torch.Tensor) and a.ndim >= 1 and int(a.shape[0]) == b_local), None)
                in_dim = max(1, int(first_t[0].numel())) if first_t is not None else max(1, int(d_pre[0].numel()))
                b_p = self._probe_budget_for_record(rec, b_gathered, in_dim)
                if len(p_args) == 1 and not p_kwargs and isinstance(p_args[0], torch.Tensor) and int(p_args[0].shape[0]) == b_local:
                    gx_all = self._all_gather_coupled(p_args[0].detach())
                    gd_all = self._all_gather_coupled(d_pre.detach())
                    p_idx = select_v15(gx_all, int(gx_all.shape[0]), b_p) if (select_v15 is not None and b_p < int(gx_all.shape[0])) else None
                    if p_idx is not None:
                        gx = gx_all.index_select(0, p_idx.to(device=gx_all.device))
                        gd = gd_all.index_select(0, p_idx.to(device=gd_all.device))
                    else:
                        gx = gx_all[:b_p]
                        gd = gd_all[:b_p]
                    synced.append(((gx,), {}, gd))
                else:
                    gd_all = self._all_gather_coupled(d_pre.detach())
                    ref_all = self._all_gather_coupled(first_t.detach()) if first_t is not None else gd_all
                    p_idx = select_v15(ref_all, int(ref_all.shape[0]), b_p) if (select_v15 is not None and b_p < int(ref_all.shape[0])) else None
                    g_args = self._gather_probe_value(p_args, b_local, b_p, p_idx=p_idx)
                    g_kwargs = self._gather_probe_value(p_kwargs, b_local, b_p, p_idx=p_idx)
                    gd = gd_all.index_select(0, p_idx.to(device=gd_all.device)) if p_idx is not None else gd_all[:b_p]
                    synced.append((g_args, g_kwargs, gd))
            else:
                synced.append((p_args, p_kwargs, d_pre))
        rec._d_probe_list = synced

    def _sync_unified_v11_g_probe(self):
        """Gather element-budgeted generator secant probe slice across ranks so all ranks compute identical alpha_gain."""
        rec = getattr(self.opt_d, 'record', None)
        cache = getattr(rec, '_g_probe_cache', None)
        if cache is None or self.world_size <= 1 or not getattr(self.aggregation_strategy, 'reduces_gradients', False):
            return
        g_mod, p_args, p_kwargs, y_pre = cache
        b_local = int(y_pre.shape[0])
        b_gathered = b_local * self.world_size
        out_dim = max(1, int(y_pre[0].numel()))
        b_p = self._probe_budget_for_record(rec, b_gathered, out_dim)
        gy_all = self._all_gather_coupled(y_pre.detach())
        p_idx = None
        if self._is_unified_v15_record(rec) and b_p < int(gy_all.shape[0]):
            from particlegan._unified_v15.critic import _select_probe_indices
            p_idx = _select_probe_indices(gy_all, int(gy_all.shape[0]), b_p)
        g_args = self._gather_probe_value(p_args, b_local, b_p, p_idx=p_idx)
        g_kwargs = self._gather_probe_value(p_kwargs, b_local, b_p, p_idx=p_idx)
        gy = gy_all.index_select(0, p_idx.to(device=gy_all.device)) if p_idx is not None else gy_all[:b_p]
        rec._g_probe_cache = (_BroadcastProbeProxy(g_mod, gy), g_args, g_kwargs, gy)

    def _gather_rhbi_pair(self, fake_score, real_score, rec):
        """Gather critic logits and RHBI input/conditioning captures across all ranks."""
        from particlegan._rhbi.critic import _lookup_rhbi_capture_parts

        x_f, cond_f, rec_f, cc_f, parts_f = _lookup_rhbi_capture_parts(fake_score, rec)
        x_r, cond_r, rec_r, cc_r, parts_r = _lookup_rhbi_capture_parts(
            real_score, rec_f if rec_f is not None else rec
        )
        eff_rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else rec)
        global_f_sc = self._all_gather_coupled(fake_score)
        global_r_sc = self._all_gather_coupled(real_score)
        if x_f is not None and x_r is not None:
            global_x_f = self._all_gather_coupled(x_f)
            global_x_r = self._all_gather_coupled(x_r)
            global_cond_f = self._all_gather_coupled(cond_f) if cond_f is not None else None
            global_cond_r = self._all_gather_coupled(cond_r) if cond_r is not None else None
            global_parts_f = tuple(self._all_gather_coupled(p) for p in parts_f) if parts_f is not None else None
            global_parts_r = tuple(self._all_gather_coupled(p) for p in parts_r) if parts_r is not None else None
            global_f_sc._rhbi_input = global_x_f
            global_f_sc._rhbi_cond = global_cond_f
            global_f_sc._rhbi_cont_cond = cc_f
            global_f_sc._rhbi_parts = global_parts_f
            global_f_sc._rhbi_record = eff_rec
            if getattr(fake_score, '_rhbi_thermally_scaled', False):
                global_f_sc._rhbi_thermally_scaled = True
            global_r_sc._rhbi_input = global_x_r
            global_r_sc._rhbi_cond = global_cond_r
            global_r_sc._rhbi_cont_cond = cc_r
            global_r_sc._rhbi_parts = global_parts_r
            global_r_sc._rhbi_record = eff_rec
        return global_f_sc, global_r_sc

    @staticmethod
    def _critic_checked_parameters(terms):
        if len(terms) == 1:
            return list(terms[0].module.parameters())
        checked = []
        seen = set()
        for term in terms:
            for parameter in term.module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                checked.append(parameter)
        return checked

    def _grad_norm_scalar(self, parameters):
        with torch.no_grad():
            grads = [p.grad.detach() for p in parameters if p.grad is not None]
            if not grads:
                return torch.zeros((), dtype=torch.float64, device=self.device)
            sq_sum = sum(g.double().norm(2) ** 2 for g in grads)
            return torch.sqrt(sq_sum).reshape(())

    def _particle_rms_scalar(self):
        with torch.no_grad():
            if hasattr(self, 'prior') and hasattr(self.prior, 'z') and isinstance(self.prior.z, torch.Tensor):
                return torch.sqrt(torch.mean(self.prior.z.detach().double() ** 2)).reshape(())
            return torch.zeros((), dtype=torch.float64, device=self.device)

    def _critic_score_scalars(self, real_score, fake_score):
        with torch.no_grad():
            if isinstance(real_score, torch.Tensor) and isinstance(fake_score, torch.Tensor):
                rs = real_score.detach()
                fs = fake_score.detach()
                real_mean = rs.double().mean().reshape(())
                fake_mean = fs.double().mean().reshape(())
                try:
                    acc = (rs > fs).double().mean().reshape(())
                except RuntimeError:
                    acc = torch.full((), 0.5, dtype=torch.float64, device=self.device)
                return real_mean, fake_mean, acc
            zero = torch.zeros((), dtype=torch.float64, device=self.device)
            half = torch.full((), 0.5, dtype=torch.float64, device=self.device)
            return zero, zero, half

    def _sample_diversity_stats(self, fake_samples, real_samples):
        """Capture per-feature mean and cancellation-free centered sum of squares for global inter-sample spread."""
        with torch.no_grad():
            zero = torch.zeros((), dtype=torch.float64, device=self.device)

            def stats_for(sample):
                if isinstance(sample, torch.Tensor) and sample.ndim >= 1 and sample.shape[0] > 0 and sample.numel() > 0:
                    b = int(sample.shape[0])
                    flat = sample.detach().reshape(b, -1).double()
                    mean_d = flat.mean(dim=0)
                    centered_ss = (flat - mean_d.unsqueeze(0)).square().sum().reshape(())
                    return mean_d, centered_ss, b, int(flat.shape[1])
                return None, zero, 0, 0

            return stats_for(fake_samples), stats_for(real_samples)

    def _combine_diversity_stats(self, stats_list):
        """Combine microbatch (mean_d, centered_ss, b, feat_dim) tuples via Chan's parallel variance identity."""
        with torch.no_grad():
            zero = torch.zeros((), dtype=torch.float64, device=self.device)
            valid = [s for s in stats_list if s[0] is not None and s[2] > 0 and s[3] > 0]
            if not valid:
                return None, zero, 0, 0
            total_b = sum(s[2] for s in valid)
            feat_dim = valid[0][3]
            mean_d = sum(s[0] * float(s[2]) for s in valid) / float(total_b)
            centered_ss = sum(
                s[1] + float(s[2]) * (s[0] - mean_d).square().sum()
                for s in valid
            ).reshape(())
            return mean_d, centered_ss, total_b, feat_dim

    def _global_diversity_var_scalars(self, fake_stats, real_stats):
        """Reduce per-feature means once to obtain exact unbiased global inter-sample feature variances."""
        with torch.no_grad():
            zero = torch.zeros((), dtype=torch.float64, device=self.device)
            f_mean, f_ss, f_b, f_d = fake_stats
            r_mean, r_ss, r_b, r_d = real_stats
            global_f_b = f_b * self.world_size
            global_r_b = r_b * self.world_size
            if f_mean is None or r_mean is None or global_f_b < 2 or global_r_b < 2 or f_d < 1 or r_d < 1:
                return zero, zero, global_f_b, global_r_b
            means_cat = torch.cat((f_mean, r_mean), dim=0).contiguous()
            dist.all_reduce(means_cat, op=dist.ReduceOp.SUM)
            means_cat.div_(self.world_size)
            global_f_mean = means_cat[:f_d]
            global_r_mean = means_cat[f_d:]
            rank_f_ss = f_ss + float(f_b) * (f_mean - global_f_mean).square().sum()
            rank_r_ss = r_ss + float(r_b) * (r_mean - global_r_mean).square().sum()
            # _complete averages scalars across ranks (dividing by world_size), so scaling by
            # world_size here yields the exact global sum / (D * (B_global - 1)) after _complete.
            fake_var_contrib = (rank_f_ss * (float(self.world_size) / float(f_d * (global_f_b - 1)))).reshape(())
            real_var_contrib = (rank_r_ss * (float(self.world_size) / float(r_d * (global_r_b - 1)))).reshape(())
            return fake_var_contrib, real_var_contrib, global_f_b, global_r_b

    @staticmethod
    def _populate_telemetry(row, tail, global_fake_batch, global_real_batch):
        (
            raw_real_mean,
            raw_fake_mean,
            raw_acc,
            raw_gn_critic,
            raw_gn_gen,
            raw_gn_prior,
            raw_particle_rms,
            fake_var,
            real_var,
            raw_d_guard,
        ) = tail
        critic_real_mean = float(raw_real_mean) if math.isfinite(raw_real_mean) else 0.0
        critic_fake_mean = float(raw_fake_mean) if math.isfinite(raw_fake_mean) else 0.0
        critic_margin = critic_real_mean - critic_fake_mean
        if not math.isfinite(critic_margin):
            critic_margin = 0.0
        critic_accuracy = float(raw_acc) if math.isfinite(raw_acc) else 0.5
        critic_grad_norm = float(raw_gn_critic) if math.isfinite(raw_gn_critic) else 0.0
        gen_grad_norm = float(raw_gn_gen) if math.isfinite(raw_gn_gen) else 0.0
        prior_grad_norm = float(raw_gn_prior) if math.isfinite(raw_gn_prior) else 0.0
        particle_rms = float(raw_particle_rms) if math.isfinite(raw_particle_rms) else 0.0
        d_guard_ratio = float(raw_d_guard) if math.isfinite(raw_d_guard) else 1.0

        if (
            global_fake_batch >= 2
            and global_real_batch >= 2
            and math.isfinite(fake_var)
            and math.isfinite(real_var)
        ):
            fake_spread = math.sqrt(max(0.0, float(fake_var)))
            real_spread = math.sqrt(max(0.0, float(real_var)))
            if real_spread > 0.0:
                cand_ratio = fake_spread / (real_spread + 1e-8)
                diversity_ratio = float(cand_ratio) if math.isfinite(cand_ratio) else 1.0
            else:
                diversity_ratio = 1.0
        else:
            diversity_ratio = 1.0

        row.update(
            critic_real_mean=critic_real_mean,
            critic_fake_mean=critic_fake_mean,
            critic_margin=critic_margin,
            critic_accuracy=critic_accuracy,
            grad_norm_critic=critic_grad_norm,
            grad_norm_generator=gen_grad_norm,
            grad_norm_prior=prior_grad_norm,
            d_guard_ratio=d_guard_ratio,
            particle_rms=particle_rms,
            diversity_ratio=diversity_ratio,
        )
        row["critic/real_mean"] = critic_real_mean
        row["critic/fake_mean"] = critic_fake_mean
        row["critic/margin"] = critic_margin
        row["critic/accuracy"] = critic_accuracy
        row["grad_norm/critic"] = critic_grad_norm
        row["grad_norm/generator"] = gen_grad_norm
        row["grad_norm/prior"] = prior_grad_norm
        row["optimizer/d_guard_ratio"] = d_guard_ratio
        row["prior/particle_rms"] = particle_rms
        row["diversity/ratio"] = diversity_ratio

    @staticmethod
    def _populate_per_critic_metrics(row, terms, per_term_values, *, thermal_coordinator=None):
        row["critic/thermal_scale"] = (
            thermal_coordinator.effective_scale() if thermal_coordinator is not None else 1.0
        )
        for idx, term in enumerate(terms):
            offset = idx * 5
            d_adv_w_v, d_pen_v, d_tot_v, r_mean_v, f_mean_v = per_term_values[offset:offset + 5]
            r_m = float(r_mean_v) if math.isfinite(r_mean_v) else 0.0
            f_m = float(f_mean_v) if math.isfinite(f_mean_v) else 0.0
            margin_v = r_m - f_m
            if not math.isfinite(margin_v):
                margin_v = 0.0
            eff_th, raw_th = (
                thermal_coordinator.term_scales(term) if thermal_coordinator is not None else (1.0, 1.0)
            )
            tid = term.id
            row[f"loss/critics/{tid}/d_adversarial"] = float(d_adv_w_v)
            row[f"loss/critics/{tid}/gradient_penalty"] = float(d_pen_v)
            row[f"loss/critics/{tid}/d_total"] = float(d_tot_v)
            row[f"critic/{tid}/real_mean"] = r_m
            row[f"critic/{tid}/fake_mean"] = f_m
            row[f"critic/{tid}/margin"] = margin_v
            row[f"critic/{tid}/thermal_scale"] = eff_th
            row[f"critic/{tid}/raw_thermal_scale"] = raw_th

    def update(self, batch=None, latent_draw=None):
        try:
            self._agree('update boundary', {'step': self.step, 'ready': self.checkpoint_ready, 'poisoned': self._poisoned})
        except BaseException:
            self.checkpoint_ready = False
            self._poisoned = True
            raise
        if self._poisoned or not self.checkpoint_ready:
            raise RuntimeError('Trainer is not at a complete boundary; restart the whole worker group from a coordinated checkpoint')
        if self.step >= self.config['training']['steps']:
            raise ValueError('Configured total update schedule is complete')
        self.checkpoint_ready = False
        try:
            with self._training_step_precision():
                return self._update(batch, latent_draw)
        except BaseException:
            self._poisoned = True
            raise

    # An update folds its rank-local phases into one error agreement between
    # consecutive collectives (review 2026-10-25 M5: it used to exchange once per
    # phase, about 23 pickled object collectives per update).

    def _inputs(self, batch, latent_draw):
        batch, latent_draw = self._phase('input device transfer', lambda: (move_tensors(batch, self.device), move_tensors(latent_draw, self.device)))
        self._agree('input source', {'sample_data': batch is None, 'sample_prior': latent_draw is None})
        return self.batch() if batch is None else batch, latent_draw

    def _complete(self, step, metrics):
        """Finish an update; return the rank-mean metrics once every rank completed it."""
        with self._folded('update completion'):
            # The strategy hook may synchronize replicas, so its collectives come first.
            self._phase('aggregation step complete', lambda: self.aggregation_strategy.on_step_complete(step))
            self._phase('EMA', lambda: (update_ema(self.ema_graph, self.graph, self.config['training']['ema']),
                                        update_ema(self.ema_prior, self.prior, self.config['training']['ema'])))
            self._phase('module compatibility', self._check_modules)
            if self.aggregation_strategy.should_assert_replicas(step):
                self._assert_replicas('complete replicated state')
            self._phase('complete CUDA update', self._synchronize)
        values = torch.tensor(metrics, dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        values /= self.world_size
        return values.tolist()

    def _update(self, batch, latent_draw):
        if self.accumulation_steps > 1:
            return self._accumulated_update(batch, latent_draw)
        cfg, step = self.config, self.step + 1
        settings = cfg['training']
        independent = settings.get('phase_draws') == 'independent'
        scale = schedule_learning_rates(self, step - 1)
        input_std, output_std = noise_levels(self, step - 1)
        noise_stream = self.streams['noise']

        def prepare(b, l_draw):
            local_batch = b
            self._validate_batch(local_batch)
            z, ids = self.prior.sample(self.local_batch_size, generator=self.streams['prior']) if l_draw is None else l_draw
            if not isinstance(z, torch.Tensor) or z.device != self.device or z.ndim < 1 or len(z) != self.local_batch_size or not torch.isfinite(z).all():
                raise ValueError('Latent draw must contain finite local-batch values on the rank device')
            if ids is not None and (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 1 or len(ids) != self.local_batch_size):
                raise ValueError('Prior IDs must be an int64 vector of local_batch_size')
            context = self.graph.generate(z, local_batch, prior=self.prior)
            raw_fake, real = context['generated'], local_batch['real']
            if not self._valid_generated_shape(raw_fake, real) or not torch.isfinite(raw_fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            return local_batch, context, raw_fake, real, ids

        def prepare_for_d(b, l_draw):
            if independent:
                with torch.no_grad():
                    return prepare(b, l_draw)
            return prepare(b, l_draw)

        with self._folded('batch and generator forward'):
            batch, latent_draw = self._inputs(batch, latent_draw)
            batch, context, fake, real, ids = self._phase('batch and generator forward', lambda: prepare_for_d(batch, latent_draw))
            self._agree('real data shape', (tuple(real.shape), str(real.dtype)))
            self._agree('prior kind', ids is None)
        selected = None
        if ids is not None:
            selected = self.collectives.unique_indices(ids, num_rows=len(self.prior.z))
        if hasattr(self.opt_d, 'controller') and self.opt_d.controller is not None and getattr(self.opt_d, 'record', None) is not None:
            controller, record = self.opt_d.controller, self.opt_d.record
            for key in getattr(controller, '_TENSORS', ()):
                val = getattr(controller, key, None)
                if isinstance(val, torch.Tensor) and val.device != self.device:
                    setattr(controller, key, val.to(self.device))
            gathered = [torch.empty_like(real) for _ in range(self.world_size)]
            dist.all_gather(gathered, real.contiguous())
            controller.begin_critic_step(torch.cat(gathered, dim=0), record)

        terms = self.program.adversarial_terms
        noise = ((output_std, noise_stream), (input_std, noise_stream))

        from .particlegan_runtime import invoke_critic_penalty
        thermal_coordinator = getattr(self, 'thermal_coordinator', None)
        if thermal_coordinator is not None:
            thermal_coordinator.synchronize()

        def penalty_fn(term, real_sample, fake_sample):
            critic_fn = make_scored_critic(term, context, self.graph, noise=(input_std, noise_stream))
            return invoke_critic_penalty(term.penalty_fn, critic_fn, real_sample, fake_sample)

        def discriminator_forward():
            unweighted = []
            weighted = []
            penalties = []
            per_term_tensors = []
            first_fake = None
            first_real = None
            first_r_sc = None
            first_f_sc = None
            for index, term in enumerate(terms):
                r_s, f_s, r_sc, f_sc = _bound_scores(
                    term, context, self.graph, "critic", term.critic_phase, first="real", noise=noise)
                r_sc, f_sc = self._logits(r_sc, f_sc)
                loss = term.gan.d_loss(r_sc, f_sc)
                weighted_t = term.critic_weight * loss
                unweighted.append(loss)
                weighted.append(weighted_t)
                if term.penalty_fn is not None:
                    pen_t = penalty_fn(term, r_s, f_s)
                    penalties.append(pen_t)
                else:
                    pen_t = f_s.new_zeros(())
                tot_t = weighted_t + pen_t
                r_mean_t, f_mean_t, _ = self._critic_score_scalars(r_sc, f_sc)
                per_term_tensors.extend((
                    weighted_t.detach().reshape(()),
                    pen_t.detach().reshape(()),
                    tot_t.detach().reshape(()),
                    r_mean_t,
                    f_mean_t,
                ))
                if index == 0:
                    first_real = r_s
                    first_fake = f_s
                    first_r_sc = r_sc
                    first_f_sc = f_sc
            d_adv = _sum_tensors(unweighted)
            d_adv_weighted = _sum_tensors(weighted)
            d_pen = _sum_tensors(penalties) if penalties else (first_fake.new_zeros(()) if first_fake is not None else real.new_zeros(()))
            d_tot = d_adv_weighted + d_pen
            score_stats = self._critic_score_scalars(first_r_sc, first_f_sc)
            diversity_stats = self._sample_diversity_stats(first_fake, first_real)
            return d_adv, d_adv_weighted, d_pen, d_tot, score_stats, diversity_stats, per_term_tensors

        for optimizer in self.critic_optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        coupled_d = bool(
            getattr(self.aggregation_strategy, 'reduces_gradients', False)
            and any(self._uses_coupled_critic_loss(term, self._term_real_sample(term.critic_phase, context)) for term in terms)
        )
        if coupled_d:
            def discriminator_scores_only():
                scored = []
                for term in terms:
                    scored.append(_bound_scores(
                        term, context, self.graph, "critic", term.critic_phase, first="real", noise=noise))
                return scored

            with self._folded('discriminator scores'):
                d_scored = self._phase('discriminator scores', discriminator_scores_only)
            d_global_pairs = [
                self._gather_coupled_pair(term, f_sc, r_sc, fake_sample=f_s, real_sample=r_s)
                if self._uses_coupled_critic_loss(term, r_s) else None
                for term, (r_s, f_s, r_sc, f_sc) in zip(terms, d_scored)
            ]

            def discriminator_coupled_loss():
                unweighted = []
                weighted = []
                penalties = []
                per_term_tensors = []
                first_fake = None
                first_real = None
                first_r_sc = None
                first_f_sc = None
                for index, (term, (r_s, f_s, r_sc, f_sc), g_pair) in enumerate(zip(terms, d_scored, d_global_pairs)):
                    if g_pair is not None:
                        gf_sc, gr_sc = g_pair
                        loss = self._scale_coupled_backward(term.gan.d_loss(gr_sc, gf_sc))
                    else:
                        r_sc_l, f_sc_l = self._logits(r_sc, f_sc)
                        loss = term.gan.d_loss(r_sc_l, f_sc_l)
                    weighted_t = term.critic_weight * loss
                    unweighted.append(loss)
                    weighted.append(weighted_t)
                    if term.penalty_fn is not None:
                        pen_t = penalty_fn(term, r_s, f_s)
                        penalties.append(pen_t)
                    else:
                        pen_t = f_s.new_zeros(())
                    tot_t = weighted_t + pen_t
                    r_mean_t, f_mean_t, _ = self._critic_score_scalars(r_sc, f_sc)
                    per_term_tensors.extend((
                        weighted_t.detach().reshape(()),
                        pen_t.detach().reshape(()),
                        tot_t.detach().reshape(()),
                        r_mean_t,
                        f_mean_t,
                    ))
                    if index == 0:
                        first_real = r_s
                        first_fake = f_s
                        first_r_sc = r_sc
                        first_f_sc = f_sc
                d_adv = _sum_tensors(unweighted)
                d_adv_weighted = _sum_tensors(weighted)
                d_pen = _sum_tensors(penalties) if penalties else (first_fake.new_zeros(()) if first_fake is not None else real.new_zeros(()))
                d_tot = d_adv_weighted + d_pen
                score_stats = self._critic_score_scalars(first_r_sc, first_f_sc)
                diversity_stats = self._sample_diversity_stats(first_fake, first_real)
                return d_adv, d_adv_weighted, d_pen, d_tot, score_stats, diversity_stats, per_term_tensors

            with self._folded('discriminator backward'):
                d_adversarial, d_adversarial_weighted, d_penalty, d_loss, d_score_stats, d_diversity_stats, per_term_tensors = self._phase(
                    'discriminator forward and penalty', discriminator_coupled_loss)
                critic_real_mean_t, critic_fake_mean_t, critic_acc_t = d_score_stats
                self._phase('discriminator loss', lambda: self._finite_loss(d_loss))
                self._phase('discriminator backward', d_loss.backward)
        else:
            with self._folded('discriminator backward'):
                d_adversarial, d_adversarial_weighted, d_penalty, d_loss, d_score_stats, d_diversity_stats, per_term_tensors = self._phase(
                    'discriminator forward and penalty', discriminator_forward)
                critic_real_mean_t, critic_fake_mean_t, critic_acc_t = d_score_stats
                self._phase('discriminator loss', lambda: self._finite_loss(d_loss))
                self._phase('discriminator backward', d_loss.backward)
        critic_grad_norm_t, d_guard_ratio_t = self._step_critics(terms)

        saved_flags = []
        saved_bn_tracking = []
        seen_modules = set()
        for term in terms:
            if id(term.module) in seen_modules:
                continue
            seen_modules.add(id(term.module))
            flags = [p.requires_grad for p in term.module.parameters()]
            term.module.requires_grad_(False)
            saved_flags.append((term.module, flags))
            for sub in term.module.modules():
                if (
                    isinstance(sub, torch.nn.modules.batchnorm._BatchNorm)
                    and sub.training
                    and sub.track_running_stats
                ):
                    sub.track_running_stats = False
                    saved_bn_tracking.append(sub)
        rhbi_real = self._term_real_sample(terms[0].critic_phase, context) if len(terms) == 1 else real
        rhbi_mod, is_discrete_prior, use_rhbi_simplex = _rhbi_generator_context(self, terms, rhbi_real)
        thermal_scope = thermal_coordinator.step_scope() if thermal_coordinator is not None else contextlib.nullcontext()
        try:
            with thermal_scope:
                if independent:
                    with self._folded('generator batch and forward'):
                        gen_batch = self.batch()
                        gen_batch, gen_context, gen_fake, gen_real, gen_ids = self._phase(
                            'generator batch and forward', lambda: prepare(gen_batch, None))
                    gen_selected = None if gen_ids is None else self.collectives.unique_indices(gen_ids, num_rows=len(self.prior.z))
                else:
                    gen_batch, gen_context, gen_fake, gen_real, gen_selected, gen_ids = batch, context, fake, real, selected, ids

                self.opt_g.zero_grad(set_to_none=True)
                global_ids_for_axiom = (
                    self._all_gather_coupled(gen_ids.detach())
                    if (
                        gen_ids is not None
                        and getattr(self.recipe, 'effective_critic_formulation', None) == 'axiom'
                        and getattr(self.aggregation_strategy, 'reduces_gradients', False)
                    )
                    else gen_ids
                )
                global_real_for_axiom = (
                    self._all_gather_coupled(gen_real.detach())
                    if (
                        use_rhbi_simplex
                        and getattr(self.recipe, 'effective_critic_formulation', None) == 'axiom'
                        and getattr(self.aggregation_strategy, 'reduces_gradients', False)
                    )
                    else gen_real
                )
                effective_simplex = _prepare_discrete_prior_step(
                    self, rhbi_mod, is_discrete_prior, use_rhbi_simplex, global_ids_for_axiom, global_real_for_axiom
                )

                def generator_adversarial_forward():
                    unweighted = []
                    weighted = []
                    for term in terms:
                        r_s, f_s, r_sc, f_sc = _bound_scores(
                            term, gen_context, self.graph, "generator", term.generator_phase, first="fake", noise=noise)
                        r_sc, f_sc = self._logits(r_sc, f_sc)
                        loss = term.gan.g_loss(f_sc, r_sc)
                        unweighted.append(loss)
                        weighted.append(term.generator_weight * loss)
                    return _sum_tensors(unweighted), _sum_tensors(weighted)

                def remaining_losses(simplex_real=None):
                    rows = None if gen_selected is None else self.prior.z if cfg['prior_regularizer']['rows'] == 'full' else self.prior.z[gen_selected]
                    prior_loss = self.recipe.prior_regularization(rows, regularizer=self.spread, prior=self.prior) if rows is not None else gen_fake.new_zeros(())
                    simplex_loss = (
                        _rhbi_simplex_loss(self, rhbi_mod, gen_selected, simplex_real)
                        if (effective_simplex and rhbi_mod is not None and simplex_real is not None)
                        else None
                    )
                    objective_losses = []
                    for term_spec, objective in zip(cfg['objectives'], self.objectives):
                        inputs = {arg: self.graph.resolve(path, gen_context) for arg, path in term_spec['inputs'].items()}
                        for arg in term_spec['detach']:
                            inputs[arg] = detach(inputs[arg])
                        value = objective(**inputs)
                        if not isinstance(value, torch.Tensor) or value.numel() != 1:
                            raise ValueError('Each objective must return one scalar tensor')
                        objective_losses.append(term_spec['weight'] * value)
                    return prior_loss, simplex_loss, objective_losses

                from particlegan.optim.constraint_geometry import constraint_geometry_backward
                coupled_g = bool(
                    getattr(self.aggregation_strategy, 'reduces_gradients', False)
                    and any(self._uses_coupled_generator_loss(term, self._term_real_sample(term.generator_phase, gen_context)) for term in terms)
                )
                if coupled_g:
                    def generator_scores_only():
                        return [
                            _bound_scores(term, gen_context, self.graph, "generator", term.generator_phase, first="fake", noise=noise)
                            for term in terms
                        ]

                    with self._folded('generator critic scores'):
                        g_scored = self._phase('generator critic scores', generator_scores_only)
                    g_global_pairs = [
                        self._gather_coupled_pair(term, f_sc, r_sc, fake_sample=f_s, real_sample=r_s)
                        if self._uses_coupled_generator_loss(term, r_s) else None
                        for term, (r_s, f_s, r_sc, f_sc) in zip(terms, g_scored)
                    ]
                    self._sync_coupled_generator_latents(terms)
                    global_simplex_real = (
                        self._all_gather_coupled(gen_real.detach())
                        if (effective_simplex and rhbi_mod is not None)
                        else None
                    )

                    def generator_coupled_adversarial():
                        unweighted = []
                        weighted = []
                        for term, (_, _, r_sc, f_sc), g_pair in zip(terms, g_scored, g_global_pairs):
                            if g_pair is not None:
                                gf_sc, gr_sc = g_pair
                                loss = self._scale_coupled_backward(term.gan.g_loss(gf_sc, gr_sc))
                            else:
                                r_sc_l, f_sc_l = self._logits(r_sc, f_sc)
                                loss = term.gan.g_loss(f_sc_l, r_sc_l)
                            unweighted.append(loss)
                            weighted.append(term.generator_weight * loss)
                        return _sum_tensors(unweighted), _sum_tensors(weighted)

                    with self._folded('generator backward'):
                        g_adversarial, g_adversarial_weighted = self._phase(
                            'generator critic forward', generator_coupled_adversarial)
                        prior_loss, simplex_loss, objective_losses = self._phase(
                            'generator regularizers and objectives', lambda: remaining_losses(global_simplex_real))
                        g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
                        if simplex_loss is not None:
                            g_loss = g_loss + simplex_loss
                        set_rows = getattr(self.opt_g, 'set_sampled_rows', None)
                        if set_rows is not None and gen_selected is not None and self.program.prior_parameters:
                            set_rows(self.prior.z, gen_selected)
                        self._phase('generator loss', lambda: self._finite_loss(g_loss))
                        self._phase('generator backward', lambda: constraint_geometry_backward(g_loss, self.opt_g, (g_adversarial_weighted,)))
                else:
                    simplex_real_local = gen_real if (effective_simplex and rhbi_mod is not None) else None
                    with self._folded('generator backward'):
                        g_adversarial, g_adversarial_weighted = self._phase(
                            'generator critic forward', generator_adversarial_forward)
                        prior_loss, simplex_loss, objective_losses = self._phase(
                            'generator regularizers and objectives', lambda: remaining_losses(simplex_real_local))
                        g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
                        if simplex_loss is not None:
                            g_loss = g_loss + simplex_loss
                        set_rows = getattr(self.opt_g, 'set_sampled_rows', None)
                        if set_rows is not None and gen_selected is not None and self.program.prior_parameters:
                            set_rows(self.prior.z, gen_selected)
                        self._phase('generator loss', lambda: self._finite_loss(g_loss))
                        self._phase('generator backward', lambda: constraint_geometry_backward(g_loss, self.opt_g, (g_adversarial_weighted,)))

                self._reduce_protected(self.opt_g)

                self._reduce_gradients(self.opt_g, 'generator/prior/auxiliary')
                gen_grad_norm_t = self._grad_norm_scalar(self.program.generator_parameters)
                prior_grad_norm_t = (
                    self._grad_norm_scalar(self.program.prior_checked)
                    if self.program.prior_checked
                    else torch.zeros((), dtype=torch.float64, device=self.device)
                )
                self._sync_unified_v11_g_probe()
                self._optimizer_step(self.opt_g, 'generator')
        finally:
            _clear_discrete_prior_step(self, rhbi_mod)
            for sub in saved_bn_tracking:
                sub.track_running_stats = True
            for module, flags in saved_flags:
                for parameter, flag in zip(module.parameters(), flags):
                    parameter.requires_grad_(flag)
        fake_var_t, real_var_t, global_fake_batch, global_real_batch = self._global_diversity_var_scalars(
            d_diversity_stats[0], d_diversity_stats[1]
        )
        particle_rms_t = self._particle_rms_scalar()
        n_objectives = len(objective_losses)
        transferred = self._metric_transfer([
            d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial, d_adversarial_weighted, g_adversarial_weighted,
            *objective_losses,
            critic_real_mean_t, critic_fake_mean_t, critic_acc_t,
            critic_grad_norm_t, gen_grad_norm_t, prior_grad_norm_t,
            particle_rms_t,
            fake_var_t, real_var_t,
            d_guard_ratio_t,
            *per_term_tensors,
        ])
        values = self._complete(step, transferred)
        self.step = step
        self.checkpoint_ready = True
        self._handshake_complete = True
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:8 + n_objectives], lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size, world_size=self.world_size)
        telemetry_start = 8 + n_objectives
        self._populate_telemetry(
            row,
            values[telemetry_start:telemetry_start + 10],
            global_fake_batch,
            global_real_batch,
        )
        self._populate_per_critic_metrics(
            row, terms, values[telemetry_start + 10:], thermal_coordinator=thermal_coordinator
        )
        return row, detach(batch)

    def _scored_critic(self, context, input_noise=(0, None)):
        std, stream = input_noise
        disc = self.graph.models['discriminator']
        use_bf16 = bool(self.graph._training_bf16_active and self.graph.uses_bf16(disc))
        critic = ScoredCritic(disc, None, bf16=use_bf16)
        def score(module, value):
            source = value
            if std and stream is not None:
                value = with_noise(value, std, stream)
            fixed = detach(context)
            fixed['candidate'] = value
            kwargs = {}
            for arg, path in self.graph.specs['discriminator']['inputs'].items():
                resolved = self.graph.resolve(path, fixed)
                kwargs[arg] = resolved if path == 'candidate' else detach(resolved)
            return self.graph.call_critic(module, kwargs, source_candidate=source, bf16=critic.bf16)
        critic.score = score
        return critic

    def _check_accumulation(self):
        # Arbitrary Python forwards cannot be proved sample independent. Keep
        # them configurable, with an explicit contract rather than silent claims.
        custom = [name for name, spec in self.config['components'].items() if ':' in spec['factory']]
        if custom:
            warnings.warn('Accumulation requires sample-independent, replayable custom components: '
                          + ', '.join(custom) + '. Batch-coupled operations or unregistered mutable state are not qualified.',
                          RuntimeWarning, stacklevel=2)
        for name, module in self.graph.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and (module.training or not module.track_running_stats):
                raise ValueError(f'Accumulation cannot preserve batch-dependent BatchNorm in {name}')
        self._objective_reductions = []
        for objective in self.objectives:
            reduction = objective.reduction if type(objective) in (torch.nn.MSELoss, torch.nn.L1Loss) else getattr(objective, 'accumulation_reduction', None)
            if reduction not in ('mean', 'sum'):
                raise ValueError("Accumulated objectives must declare accumulation_reduction='mean' or 'sum' for separable sample terms; builtin MSE/L1 use reduction")
            self._objective_reductions.append(reduction)

    def _accumulation_rng(self):
        from .checkpoints import capture_rng
        return capture_rng(), {name: stream.get_state() for name, stream in self.streams.items()}

    def _restore_accumulation_rng(self, state):
        from .checkpoints import restore_rng
        restore_rng(state[0])
        for name, value in state[1].items():
            self.streams[name].set_state(value)

    def _isolated_replay(self, state, function):
        current = self._accumulation_rng()
        try:
            self._restore_accumulation_rng(state)
            return function()
        finally:
            self._restore_accumulation_rng(current)

    def _forward_state_digest(self):
        # Covers nonpersistent buffers and portable extra state as well as
        # parameters/modes/flags. Hashing costs time, but retains no activations.
        rng = self._accumulation_rng()
        try:
            modules = [self.graph, *[x for x in self.objectives if isinstance(x, torch.nn.Module)]]
            return self._digest([{'state': module.state_dict(), 'buffers': dict(module.named_buffers()),
                                  'modes': {name: child.training for name, child in module.named_modules()},
                                  'flags': {name: p.requires_grad for name, p in module.named_parameters()}}
                                 for module in modules])
        finally:
            self._restore_accumulation_rng(rng)

    def _immutable_forward(self, function):
        before = self._forward_state_digest()
        result = function()
        if self._forward_state_digest() != before:
            raise ValueError('Accumulation forward mutated registered parameters, buffers, extra state, modes or trainability; replay is unsupported')
        return result

    def _microbatch(self, value, start):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if len(value) != self.local_batch_size:
                raise ValueError('Accumulation data tensors must share local_batch_size or be scalar')
            return value[start:start + self.microbatch_size]
        if isinstance(value, dict):
            return {key: self._microbatch(item, start) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._microbatch(item, start) for item in value)
        return value

    def _micro_logits(self, *values):
        for value in values:
            if (not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != self.microbatch_size
                    or not value.is_floating_point() or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError('Accumulation critic logits must be finite floating tensors on the rank device with a microbatch leading dimension')

    def _accumulated_update(self, batch, latent_draw):
        """Replay one micro graph at a time using a full-logit loss cotangent.

        Only inputs, the prior/latent graph, detached logits/cotangents and RNG
        metadata scale with the effective batch. Hidden network activations and
        exact penalty double-backward graphs are bounded by one microbatch.
        """
        cfg, step = self.config, self.step + 1
        scale = schedule_learning_rates(self, step - 1)

        def prepare():
            z, ids = self.prior.sample(self.local_batch_size, generator=self.streams['prior']) if latent_draw is None else latent_draw
            if not isinstance(z, torch.Tensor) or z.device != self.device or z.ndim < 1 or len(z) != self.local_batch_size or not torch.isfinite(z).all():
                raise ValueError('Latent draw must contain finite local-batch values on the rank device')
            if ids is not None and (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 1 or len(ids) != self.local_batch_size):
                raise ValueError('Prior IDs must be an int64 vector of local_batch_size')
            # Validate nested batch slicing before any model state changes.
            for start in range(0, self.local_batch_size, self.microbatch_size):
                self._microbatch(batch, start)
            return z, ids

        with self._folded('accumulation inputs'):
            batch, latent_draw = self._inputs(batch, latent_draw)
            self._phase('accumulation batch', lambda: self._validate_batch(batch))
            z, ids = self._phase('accumulation latent', prepare)
            self._agree('real data shape', (tuple(batch['real'].shape), str(batch['real'].dtype)))
            self._agree('prior kind', ids is None)
        selected = None if ids is None else self.collectives.unique_indices(ids, num_rows=len(self.prior.z))
        if hasattr(self.opt_d, 'controller') and self.opt_d.controller is not None and getattr(self.opt_d, 'record', None) is not None:
            controller = self.opt_d.controller
            for key in getattr(controller, '_TENSORS', ()):
                val = getattr(controller, key, None)
                if isinstance(val, torch.Tensor) and val.device != self.device:
                    setattr(controller, key, val.to(self.device))
            gathered = [torch.empty_like(batch['real']) for _ in range(self.world_size)]
            dist.all_gather(gathered, batch['real'].contiguous())
            controller.begin_critic_step(torch.cat(gathered, dim=0), self.opt_d.record)
        starts = range(0, self.local_batch_size, self.microbatch_size)
        generation_rng, d_records = [], []
        # The penalty record counts calls. Discovery and replay evaluate one
        # logical penalty per microbatch; keep the count of the plain update.
        records = [
            rec
            for optimizer in self.critic_optimizers.values()
            if (rec := getattr(optimizer, 'record', None)) is not None and hasattr(rec, 'calls')
        ]
        logical_calls = None

        def generate(local_z, local_batch):
            context = self.graph.generate(local_z, local_batch, prior=self.prior)
            fake = context['generated']
            if not self._valid_generated_shape(fake, local_batch['real']) or not torch.isfinite(fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            return context

        terms = self.program.adversarial_terms
        acc_noise = ((0, None), (0, None))

        from .particlegan_runtime import invoke_critic_penalty
        thermal_coordinator = getattr(self, 'thermal_coordinator', None)
        if thermal_coordinator is not None:
            thermal_coordinator.synchronize()

        def penalty_fn_acc(term, real_sample, fake_sample, micro_context):
            critic_fn = make_scored_critic(term, micro_context, self.graph, noise=(0, None))
            return invoke_critic_penalty(term.penalty_fn, critic_fn, real_sample, fake_sample)

        def discriminator_micro(index, start, *, capture_moments=False):
            local_batch = self._microbatch(batch, start)
            local_z = z.detach()[start:start + self.microbatch_size].requires_grad_(z.requires_grad)
            context = detach(generate(local_z, local_batch))
            term_records = []
            first_real = None
            first_fake = None
            for t_idx, term in enumerate(terms):
                r_s, f_s, r_sc, f_sc = _bound_scores(
                    term, context, self.graph, "critic", term.critic_phase, first="real", noise=acc_noise)
                self._micro_logits(r_sc, f_sc)
                if t_idx == 0:
                    first_real = r_s
                    first_fake = f_s
                if term.penalty_fn is not None:
                    pen = penalty_fn_acc(term, r_s, f_s, context)
                else:
                    pen = f_s.new_zeros(())
                term_records.append((r_sc, f_sc, pen))
            if capture_moments:
                return term_records, self._digest(context['generated']), self._sample_diversity_stats(first_fake, first_real)
            return term_records, self._digest(context['generated'])

        with self._folded('discriminator backward'):
            # Prepasses use ordinary grad mode and immediately discard each graph,
            # so custom grad-mode behavior is the same on the checked replay.
            for index, start in enumerate(starts):
                rng = self._accumulation_rng()
                generation_rng.append(rng)
                def prepass():
                    term_records, fake_digest, sample_moments = self._immutable_forward(
                        lambda: discriminator_micro(index, start, capture_moments=True))
                    for _, _, pen in term_records:
                        self._finite_loss(pen)
                    return [(r_sc.detach(), f_sc.detach(), pen.detach()) for r_sc, f_sc, pen in term_records], fake_digest, sample_moments
                d_records.append((rng, self._phase(f'D micro {index} discovery', prepass)))
                if records and logical_calls is None:
                    logical_calls = [record.calls for record in records]
            all_dr, all_df = [], []
            unweighted, weighted = [], []
            for t, term in enumerate(terms):
                dr_t = torch.cat([rec[1][0][t][0] for rec in d_records]).requires_grad_(True)
                df_t = torch.cat([rec[1][0][t][1] for rec in d_records]).requires_grad_(True)
                global_dr_t, global_df_t = self._logits(dr_t, df_t)
                loss_raw = term.gan.d_loss(global_dr_t, global_df_t)
                unweighted.append(loss_raw)
                weighted.append(term.critic_weight * loss_raw)
                all_dr.append(dr_t)
                all_df.append(df_t)
            d_adversarial_raw = _sum_tensors(unweighted)
            d_adversarial = _sum_tensors(weighted)
            self._phase('accumulated D objective', lambda: self._finite_loss(d_adversarial))
            all_targets = tuple(all_dr + all_df)
            all_cotangents = self._phase('accumulated D logit derivatives', lambda: torch.autograd.grad(d_adversarial, all_targets))
            dr_cotangents = all_cotangents[:len(terms)]
            df_cotangents = all_cotangents[len(terms):]
            d_penalty = sum(rec[1][0][t][2] / self.accumulation_steps for rec in d_records for t in range(len(terms)))
            d_loss = d_adversarial.detach() + d_penalty
            self._phase('accumulated D total loss', lambda: self._finite_loss(d_loss))
            for optimizer in self.critic_optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for index, start in enumerate(starts):
                rng, expected = d_records[index]
                def replay():
                    current = self._immutable_forward(lambda: discriminator_micro(index, start))
                    current_terms, current_digest = current
                    if current_digest != expected[1]:
                        raise ValueError('Discriminator microbatch replay differs from discovery outputs')
                    step_loss = 0
                    for t, (dr_m, df_m, pen_m) in enumerate(current_terms):
                        dr_slice = dr_cotangents[t][start:start + self.microbatch_size]
                        df_slice = df_cotangents[t][start:start + self.microbatch_size]
                        term_loss = (dr_m * dr_slice).sum() + (df_m * df_slice).sum() + pen_m / self.accumulation_steps
                        step_loss = step_loss + term_loss
                    step_loss.backward()
                self._phase(f'D micro {index} replay', lambda: self._isolated_replay(rng, replay))
            if logical_calls is not None:
                for record, calls in zip(records, logical_calls):
                    record.calls = calls
        critic_grad_norm_t, d_guard_ratio_t = self._step_critics(terms)
        saved_flags = []
        saved_bn_tracking = []
        seen_modules = set()
        for term in terms:
            if id(term.module) in seen_modules:
                continue
            seen_modules.add(id(term.module))
            flags = [p.requires_grad for p in term.module.parameters()]
            term.module.requires_grad_(False)
            saved_flags.append((term.module, flags))
            for sub in term.module.modules():
                if (
                    isinstance(sub, torch.nn.modules.batchnorm._BatchNorm)
                    and sub.training
                    and sub.track_running_stats
                ):
                    sub.track_running_stats = False
                    saved_bn_tracking.append(sub)
        thermal_scope_acc = thermal_coordinator.step_scope() if thermal_coordinator is not None else contextlib.nullcontext()
        try:
            with thermal_scope_acc:
                def generator_micro(index, start):
                    local_batch = self._microbatch(batch, start)
                    local_z = z.detach()[start:start + self.microbatch_size].requires_grad_(z.requires_grad)
                    # Recreate precisely the fake used during D, without consuming a
                    # second generator RNG trajectory in the logical update.
                    context = self._isolated_replay(generation_rng[index], lambda: generate(local_z, local_batch))
                    fake = context['generated']
                    if self._digest(fake) != d_records[index][1][1]:
                        raise ValueError('Generator replay did not reproduce the fake used by the discriminator')
                    term_records = []
                    for term in terms:
                        r_s, f_s, r_sc, f_sc = _bound_scores(
                            term, context, self.graph, "generator", term.generator_phase, first="fake", noise=acc_noise)
                        self._micro_logits(f_sc, r_sc)
                        term_records.append((f_sc, r_sc.detach()))
                    objectives = []
                    for term_spec, objective in zip(cfg['objectives'], self.objectives):
                        inputs = {arg: self.graph.resolve(path, context) for arg, path in term_spec['inputs'].items()}
                        for arg in term_spec['detach']:
                            inputs[arg] = detach(inputs[arg])
                        value = objective(**inputs)
                        if not isinstance(value, torch.Tensor) or value.numel() != 1:
                            raise ValueError('Each objective must return one scalar tensor')
                        objectives.append(term_spec['weight'] * value)
                    return term_records, objectives, local_z, self._digest(fake)

                with self._folded('generator backward'):
                    g_records = []
                    for index, start in enumerate(starts):
                        rng = self._accumulation_rng()
                        def prepass():
                            term_records, terms_obj, _, fake_digest = self._immutable_forward(lambda: generator_micro(index, start))
                            for value in terms_obj:
                                self._finite_loss(value)
                            return (
                                [(gf.detach(), gr.detach()) for gf, gr in term_records],
                                detach(terms_obj),
                                fake_digest
                            )
                        g_records.append((rng, self._phase(f'G micro {index} discovery', prepass)))
                    all_gf = []
                    g_unweighted, g_weighted = [], []
                    for t, term in enumerate(terms):
                        gf_t = torch.cat([rec[1][0][t][0] for rec in g_records]).requires_grad_(True)
                        gr_t = torch.cat([rec[1][0][t][1] for rec in g_records])
                        global_gr_t, global_gf_t = self._logits(gr_t, gf_t)
                        loss_raw = term.gan.g_loss(global_gf_t, global_gr_t)
                        g_unweighted.append(loss_raw)
                        g_weighted.append(term.generator_weight * loss_raw)
                        all_gf.append(gf_t)
                    g_adversarial = _sum_tensors(g_unweighted)
                    weighted_adversarial = _sum_tensors(g_weighted)
                    self._phase('accumulated G objective', lambda: self._finite_loss(weighted_adversarial))
                    g_cotangents = self._phase('accumulated G logit derivatives', lambda: torch.autograd.grad(weighted_adversarial, tuple(all_gf)))
                    objective_factors = [1 / self.accumulation_steps if reduction == 'mean' else 1 for reduction in self._objective_reductions]
                    objective_losses = [sum(record[1][1][i] * factor for record in g_records) for i, factor in enumerate(objective_factors)]
                    self.opt_g.zero_grad(set_to_none=True)
                    z_cotangent = torch.zeros_like(z) if z.requires_grad else None
                    any_z_gradient = False
                    needs_protected = callable(getattr(self.opt_g, 'bind_protected_losses', None))
                    opt_g_params = self._parameters(self.opt_g) if needs_protected else []
                    trainable_opt_g = [p for p in opt_g_params if p.requires_grad] if needs_protected else []
                    protected_grads = {id(p): torch.zeros_like(p) for p in opt_g_params} if needs_protected else None
                    adv_z_cotangent = torch.zeros_like(z) if (needs_protected and z.requires_grad) else None
                    any_adv_z_gradient = False
                    for index, start in enumerate(starts):
                        rng, expected = g_records[index]
                        def replay():
                            nonlocal any_z_gradient, any_adv_z_gradient
                            current = self._immutable_forward(lambda: generator_micro(index, start))
                            term_records, objectives, local_z, fake_digest = current
                            if fake_digest != expected[2]:
                                raise ValueError('Generator microbatch replay differs from discovery outputs')
                            adv_loss = 0
                            for t, (gf_m, _) in enumerate(term_records):
                                gf_slice = g_cotangents[t][start:start + self.microbatch_size]
                                adv_loss = adv_loss + (gf_m * gf_slice).sum()
                            if needs_protected and isinstance(adv_loss, torch.Tensor) and adv_loss.requires_grad:
                                targets_p = list(trainable_opt_g) + ([local_z] if adv_z_cotangent is not None else [])
                                if targets_p:
                                    computed_p = torch.autograd.grad(adv_loss, targets_p, retain_graph=True, allow_unused=True)
                                    for p_obj, g_val in zip(trainable_opt_g, computed_p[:len(trainable_opt_g)]):
                                        if g_val is not None:
                                            protected_grads[id(p_obj)].add_(g_val.detach())
                                    if adv_z_cotangent is not None:
                                        gz_adv = computed_p[-1]
                                        if gz_adv is not None:
                                            any_adv_z_gradient = True
                                            adv_z_cotangent[start:start + self.microbatch_size].copy_(gz_adv.detach())
                            step_loss = adv_loss + sum(value * factor for value, factor in zip(objectives, objective_factors))
                            step_loss.backward()
                            if z_cotangent is not None and local_z.grad is not None:
                                any_z_gradient = True
                                z_cotangent[start:start + self.microbatch_size].copy_(local_z.grad)
                        self._phase(f'G micro {index} replay', lambda: self._isolated_replay(rng, replay))
                    def prior_backward():
                        rows = None if selected is None else self.prior.z if cfg['prior_regularizer']['rows'] == 'full' else self.prior.z[selected]
                        prior_loss = self.recipe.prior_regularization(rows, regularizer=self.spread, prior=self.prior) if rows is not None else z.new_zeros(())
                        self._finite_loss(prior_loss)
                        self._finite_loss(weighted_adversarial.detach() + prior_loss.detach() + sum(objective_losses))
                        set_rows = getattr(self.opt_g, 'set_sampled_rows', None)
                        if set_rows is not None and selected is not None and self.program.prior_parameters:
                            set_rows(self.prior.z, selected)
                        if needs_protected:
                            if any_adv_z_gradient and adv_z_cotangent is not None:
                                prior_trainable = [p for p in self.program.prior_parameters if p.requires_grad]
                                if prior_trainable:
                                    adv_prior_grads = torch.autograd.grad(
                                        z, prior_trainable, grad_outputs=adv_z_cotangent, retain_graph=True, allow_unused=True
                                    )
                                    for p_obj, g_val in zip(prior_trainable, adv_prior_grads):
                                        if g_val is not None and id(p_obj) in protected_grads:
                                            protected_grads[id(p_obj)].add_(g_val.detach())
                            pieces = []
                            for p_obj in opt_g_params:
                                piece = protected_grads[id(p_obj)].detach().clone()
                                grp = next(g for g in self.opt_g.param_groups if any(p is p_obj for p in g['params']))
                                if grp.get('algorithm') == 'rownorm' and grp.get('sampled_rows_required'):
                                    sampled = self.opt_g.sampled_rows_for(p_obj)
                                    if sampled is None:
                                        raise ValueError('constraint_geometry requires explicit sampled rows')
                                    masked = torch.zeros_like(piece)
                                    masked[sampled] = piece[sampled]
                                    piece = masked
                                pieces.append(piece.flatten())
                            protected_tensor = torch.stack([torch.cat(pieces)]) if pieces else None
                            if protected_tensor is not None:
                                if not bool(torch.isfinite(protected_tensor).all()):
                                    self.opt_g._protected = None
                                    raise ValueError('nonfinite constraint_geometry protected gradient')
                                self.opt_g._protected = protected_tensor
                        outputs, gradients = [], []
                        if any_z_gradient:
                            outputs.append(z)
                            gradients.append(z_cotangent)
                        if prior_loss.requires_grad:
                            outputs.append(prior_loss)
                            gradients.append(torch.ones_like(prior_loss))
                        if outputs:
                            torch.autograd.backward(outputs, gradients)
                        return prior_loss.detach()
                    prior_loss = self._phase('global prior regularizer and latent backward', prior_backward)
                self._reduce_protected(self.opt_g)
                self._reduce_gradients(self.opt_g, 'generator/prior/auxiliary')
                gen_grad_norm_t = self._grad_norm_scalar(self.program.generator_parameters)
                prior_grad_norm_t = (
                    self._grad_norm_scalar(self.program.prior_checked)
                    if self.program.prior_checked
                    else torch.zeros((), dtype=torch.float64, device=self.device)
                )
                self._sync_unified_v11_g_probe()
                self._optimizer_step(self.opt_g, 'generator')
        finally:
            for sub in saved_bn_tracking:
                sub.track_running_stats = True
            for module, flags in saved_flags:
                for parameter, flag in zip(module.parameters(), flags):
                    parameter.requires_grad_(flag)
        g_loss = weighted_adversarial.detach() + prior_loss + sum(objective_losses)
        critic_real_mean_t, critic_fake_mean_t, critic_acc_t = self._critic_score_scalars(
            all_dr[0] if all_dr else None, all_df[0] if all_df else None)
        acc_per_term_tensors = []
        for t, term in enumerate(terms):
            d_adv_w_t = weighted[t].detach().reshape(())
            d_pen_t = sum(rec[1][0][t][2] / self.accumulation_steps for rec in d_records).detach().reshape(())
            d_tot_t = (d_adv_w_t + d_pen_t).reshape(())
            r_mean_t, f_mean_t, _ = self._critic_score_scalars(all_dr[t], all_df[t])
            acc_per_term_tensors.extend((d_adv_w_t, d_pen_t, d_tot_t, r_mean_t, f_mean_t))
        fake_div_stats = self._combine_diversity_stats([rec[1][2][0] for rec in d_records])
        real_div_stats = self._combine_diversity_stats([rec[1][2][1] for rec in d_records])
        fake_var_t, real_var_t, global_fake_batch, global_real_batch = self._global_diversity_var_scalars(
            fake_div_stats, real_div_stats
        )
        particle_rms_t = self._particle_rms_scalar()
        n_objectives = len(objective_losses)
        transferred = self._metric_transfer([
            d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial_raw, d_adversarial, weighted_adversarial,
            *objective_losses,
            critic_real_mean_t, critic_fake_mean_t, critic_acc_t,
            critic_grad_norm_t, gen_grad_norm_t, prior_grad_norm_t,
            particle_rms_t,
            fake_var_t, real_var_t,
            d_guard_ratio_t,
            *acc_per_term_tensors,
        ])
        values = self._complete(step, transferred)
        self.step, self.checkpoint_ready = step, True
        self._handshake_complete = True
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:8 + n_objectives], lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size,
                   world_size=self.world_size, accumulation_steps=self.accumulation_steps, microbatch_size=self.microbatch_size)
        telemetry_start = 8 + n_objectives
        self._populate_telemetry(
            row,
            values[telemetry_start:telemetry_start + 10],
            global_fake_batch,
            global_real_batch,
        )
        self._populate_per_critic_metrics(
            row, terms, values[telemetry_start + 10:], thermal_coordinator=thermal_coordinator
        )
        return row, detach(batch)

    @staticmethod
    def _finite_loss(value):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite loss; optimizer step refused')


class ReplicatedCPUTrainer(ReplicatedTrainer):
    """Existing CPU/Gloo entry point; CUDA callers use ReplicatedTrainer."""

    def __init__(self, config, *, world_size=2, accumulation_steps=1, **kwargs):
        if dist.is_initialized() and dist.get_backend() != 'gloo':
            raise ValueError('ReplicatedCPUTrainer requires the default Gloo group')
        super().__init__(config, world_size=world_size, accumulation_steps=accumulation_steps, **kwargs)
