"""Bounded fixed-world-size CPU/CUDA GAN updates with explicit gradient reduction.

This is an internal replicated strategy, not DDP, a launcher or a run service.
The caller owns the default Gloo or NCCL group, finite timeouts and whole-job lifecycle.
"""
import copy
import hashlib
import inspect
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
from .objective_program import _bound_scores, _sum_tensors, score_candidate

DEFAULT_BUCKET_CAP_BYTES = 25 * 1024 * 1024  # 25 MB default bucket size


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

    def _phase(self, name, function, force=False):
        if not force and self._in_hot_path:
            return function()
        result, error = None, None
        try:
            result = function()
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self._exchange('phase:' + name, error)
        if any(errors):
            raise RuntimeError(f'{name} failed: ' + '; '.join(f'rank {rank}: {value}' for rank, value in enumerate(errors) if value))
        return result

    def _agree(self, name, value, force=False):
        if not force and self._in_hot_path:
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
            state['optimizers'] = [self.opt_g.state_dict(), self.opt_d.state_dict()]
            state['base_lrs'] = self.base_lrs
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
        synchronized = self._phase('aggregation state', boundary, force=True)
        self._agree('aggregation state', synchronized, force=True)
        if synchronized:
            return False
        self.checkpoint_ready = False
        try:
            self._phase('aggregation synchronization', strategy.synchronize, force=True)
            self._phase('synchronized CUDA completion', lambda: self._synchronize(force=True), force=True)
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

    def _average_critic_statistics(self):
        # KA2 controller statistics come from rank-local critic scores.
        rec = getattr(self.opt_d, 'record', None)
        if rec is not None and hasattr(rec, 'ema_residual') and hasattr(rec, 'ema_data_scale'):
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

    def _logits(self, real, fake):
        # The supported loss pairs real and fake rows, so rank-local
        # logits suffice; the parameter-gradient mean forms the global loss.
        return real, fake

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
            batch, latent_draw = self._phase('input device transfer', lambda: (move_tensors(batch, self.device), move_tensors(latent_draw, self.device)))
            return self._update(batch, latent_draw)
        except BaseException:
            self._poisoned = True
            raise

    def _update(self, batch, latent_draw):
        if self.accumulation_steps > 1:
            return self._accumulated_update(batch, latent_draw)
        cfg, step = self.config, self.step + 1
        settings = cfg['training']
        independent = settings.get('phase_draws') == 'independent'
        scale = schedule_learning_rates(self, step - 1)
        input_std, output_std = noise_levels(self, step - 1)
        noise_stream = self.streams['noise']

        self._agree('input source', {'sample_data': batch is None, 'sample_prior': latent_draw is None})
        if batch is None:
            batch = self.batch()

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
            if not isinstance(raw_fake, torch.Tensor) or raw_fake.shape != real.shape or not torch.isfinite(raw_fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            fake = with_noise(raw_fake, output_std, noise_stream)
            context['generated'] = fake
            return local_batch, context, fake, real, ids

        batch, context, fake, real, ids = self._phase('batch and generator forward', lambda: prepare(batch, latent_draw))
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

        def penalty_fn(term, real_sample, fake_sample):
            critic_fn = ScoredCritic(term.module, lambda module, value: score_candidate(
                term, value, context, self.graph, "critic", module=module, noise=(input_std, noise_stream)))
            return term.penalty_fn(critic_fn, real_sample, fake_sample)

        def discriminator_forward():
            unweighted = []
            weighted = []
            penalties = []
            last_fake = None
            for term in terms:
                r_s, f_s, r_sc, f_sc = _bound_scores(
                    term, context, self.graph, "critic", term.critic_phase, first="real", noise=noise)
                r_sc, f_sc = self._logits(r_sc, f_sc)
                loss = term.gan.d_loss(r_sc, f_sc)
                unweighted.append(loss)
                weighted.append(term.weight * loss)
                last_fake = f_s
                if term.penalty_fn is not None:
                    penalties.append(penalty_fn(term, r_s, f_s))
            d_adv = _sum_tensors(unweighted)
            d_adv_weighted = _sum_tensors(weighted)
            d_pen = _sum_tensors(penalties) if penalties else (last_fake.new_zeros(()) if last_fake is not None else real.new_zeros(()))
            d_tot = d_adv_weighted + d_pen
            return d_adv, d_adv_weighted, d_pen, d_tot

        self.opt_d.zero_grad(set_to_none=True)
        d_adversarial, d_adversarial_weighted, d_penalty, d_loss = self._phase(
            'discriminator forward and penalty', discriminator_forward)
        self._phase('discriminator loss', lambda: self._finite_loss(d_loss))
        self._phase('discriminator backward', d_loss.backward)
        self._reduce_gradients(self.opt_d, 'discriminator')
        self._phase('discriminator optimizer', self.opt_d.step)
        self._phase('discriminator optimizer state', lambda: self._finite_optimizer(self.opt_d))
        self._average_critic_statistics()

        saved_flags = []
        seen_modules = set()
        for term in terms:
            if id(term.module) in seen_modules:
                continue
            seen_modules.add(id(term.module))
            flags = [p.requires_grad for p in term.module.parameters()]
            term.module.requires_grad_(False)
            saved_flags.append((term.module, flags))
        try:
            if independent:
                gen_batch = self.batch()
                gen_batch, gen_context, gen_fake, gen_real, gen_ids = self._phase(
                    'generator batch and forward', lambda: prepare(gen_batch, None))
                gen_selected = None if gen_ids is None else self.collectives.unique_indices(gen_ids, num_rows=len(self.prior.z))
            else:
                gen_batch, gen_context, gen_fake, gen_real, gen_selected, gen_ids = batch, context, fake, real, selected, ids

            self.opt_g.zero_grad(set_to_none=True)
            def generator_adversarial_forward():
                unweighted = []
                weighted = []
                for term in terms:
                    r_s, f_s, r_sc, f_sc = _bound_scores(
                        term, gen_context, self.graph, "generator", term.generator_phase, first="fake", noise=noise)
                    r_sc, f_sc = self._logits(r_sc, f_sc)
                    loss = term.gan.g_loss(f_sc, r_sc)
                    unweighted.append(loss)
                    weighted.append(term.weight * loss)
                return _sum_tensors(unweighted), _sum_tensors(weighted)

            g_adversarial, g_adversarial_weighted = self._phase(
                'generator critic forward', generator_adversarial_forward)

            def remaining_losses():
                rows = None if gen_selected is None else self.prior.z if cfg['prior_regularizer']['rows'] == 'full' else self.prior.z[gen_selected]
                prior_loss = self.recipe.prior_regularization(rows, regularizer=self.spread, prior=self.prior) if rows is not None else gen_fake.new_zeros(())
                objective_losses = []
                for term_spec, objective in zip(cfg['objectives'], self.objectives):
                    inputs = {arg: self.graph.resolve(path, gen_context) for arg, path in term_spec['inputs'].items()}
                    for arg in term_spec['detach']:
                        inputs[arg] = detach(inputs[arg])
                    value = objective(**inputs)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise ValueError('Each objective must return one scalar tensor')
                    objective_losses.append(term_spec['weight'] * value)
                return prior_loss, objective_losses

            prior_loss, objective_losses = self._phase('generator regularizers and objectives', remaining_losses)
            g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)

            from particlegan.optim.constraint_geometry import constraint_geometry_backward
            set_rows = getattr(self.opt_g, 'set_sampled_rows', None)
            if set_rows is not None and gen_selected is not None and self.program.prior_parameters:
                set_rows(self.prior.z, gen_selected)

            self._phase('generator loss', lambda: self._finite_loss(g_loss))
            self._phase('generator backward', lambda: constraint_geometry_backward(g_loss, self.opt_g, (g_adversarial_weighted,)))

            if getattr(self.opt_g, '_protected', None) is not None:
                if getattr(self.aggregation_strategy, 'reduces_gradients', False):
                    dist.all_reduce(self.opt_g._protected)
                    self.opt_g._protected /= self.world_size

            self._reduce_gradients(self.opt_g, 'generator/prior/auxiliary')
            self._phase('generator optimizer', self.opt_g.step)
            self._phase('generator optimizer state', lambda: self._finite_optimizer(self.opt_g))
            self._phase('aggregation step complete', lambda: self.aggregation_strategy.on_step_complete(step))
        finally:
            for module, flags in saved_flags:
                for parameter, flag in zip(module.parameters(), flags):
                    parameter.requires_grad_(flag)
        self._phase('EMA', lambda: (update_ema(self.ema_graph, self.graph, settings['ema']), update_ema(self.ema_prior, self.prior, settings['ema'])))
        self._phase('module compatibility', self._check_modules)
        if self.aggregation_strategy.should_assert_replicas(step):
            self._assert_replicas('complete replicated state')
        values = torch.tensor(self._metric_transfer([d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial, d_adversarial_weighted, g_adversarial_weighted,
            *objective_losses]), dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        values /= self.world_size
        self._phase('complete CUDA update', self._synchronize)
        self.step = step
        self.checkpoint_ready = True
        self._handshake_complete = True
        values = values.tolist()
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:], lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size, world_size=self.world_size)
        return row, detach(batch)

    def _scored_critic(self, context, input_noise=(0, None)):
        std, stream = input_noise
        def score(module, value):
            if std and stream is not None:
                value = with_noise(value, std, stream)
            return self.graph.critic(value, context, module=module)
        return ScoredCritic(self.graph.models['discriminator'], score)

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
        settings = cfg['training']
        scale = schedule_learning_rates(self, step - 1)
        self._agree('input source', {'sample_data': batch is None, 'sample_prior': latent_draw is None})
        if batch is None:
            batch = self.batch()
        self._phase('accumulation batch', lambda: self._validate_batch(batch))
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
        record, logical_calls = self.opt_d.record, None
        calls_before = record.calls

        def generate(local_z, local_batch):
            context = self.graph.generate(local_z, local_batch, prior=self.prior)
            fake = context['generated']
            if not isinstance(fake, torch.Tensor) or fake.shape != local_batch['real'].shape or not torch.isfinite(fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            return context

        terms = self.program.adversarial_terms
        acc_noise = ((0, None), (0, None))

        def penalty_fn_acc(term, real_sample, fake_sample, micro_context):
            critic_fn = ScoredCritic(term.module, lambda module, value: score_candidate(
                term, value, micro_context, self.graph, "critic", module=module, noise=(0, None)))
            return term.penalty_fn(critic_fn, real_sample, fake_sample)

        def discriminator_micro(index, start):
            local_batch = self._microbatch(batch, start)
            local_z = z.detach()[start:start + self.microbatch_size].requires_grad_(z.requires_grad)
            context = detach(generate(local_z, local_batch))
            term_records = []
            for term in terms:
                r_s, f_s, r_sc, f_sc = _bound_scores(
                    term, context, self.graph, "critic", term.critic_phase, first="real", noise=acc_noise)
                self._micro_logits(r_sc, f_sc)
                if term.penalty_fn is not None:
                    pen = penalty_fn_acc(term, r_s, f_s, context)
                else:
                    pen = f_s.new_zeros(())
                term_records.append((r_sc, f_sc, pen))
            return term_records, self._digest(context['generated'])

        # Prepasses use ordinary grad mode and immediately discard each graph,
        # so custom grad-mode behavior is the same on the checked replay.
        for index, start in enumerate(starts):
            rng = self._accumulation_rng()
            generation_rng.append(rng)
            def prepass():
                term_records, fake_digest = self._immutable_forward(lambda: discriminator_micro(index, start))
                for _, _, pen in term_records:
                    self._finite_loss(pen)
                return [(r_sc.detach(), f_sc.detach(), pen.detach()) for r_sc, f_sc, pen in term_records], fake_digest
            d_records.append((rng, self._phase(f'D micro {index} discovery', prepass)))
            if logical_calls is None:
                logical_calls = record.calls
        all_dr, all_df = [], []
        unweighted, weighted = [], []
        for t, term in enumerate(terms):
            dr_t = torch.cat([rec[1][0][t][0] for rec in d_records]).requires_grad_(True)
            df_t = torch.cat([rec[1][0][t][1] for rec in d_records]).requires_grad_(True)
            global_dr_t, global_df_t = self._logits(dr_t, df_t)
            loss_raw = term.gan.d_loss(global_dr_t, global_df_t)
            unweighted.append(loss_raw)
            weighted.append(term.weight * loss_raw)
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
        self.opt_d.zero_grad(set_to_none=True)
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
        record.calls = calls_before if logical_calls is None else logical_calls
        self._reduce_gradients(self.opt_d, 'discriminator')
        self._phase('discriminator optimizer', self.opt_d.step)
        self._phase('discriminator optimizer state', lambda: self._finite_optimizer(self.opt_d))
        self._average_critic_statistics()
        saved_flags = []
        seen_modules = set()
        for term in terms:
            if id(term.module) in seen_modules:
                continue
            seen_modules.add(id(term.module))
            flags = [p.requires_grad for p in term.module.parameters()]
            term.module.requires_grad_(False)
            saved_flags.append((term.module, flags))
        try:
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
                g_weighted.append(term.weight * loss_raw)
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
            for index, start in enumerate(starts):
                rng, expected = g_records[index]
                def replay():
                    nonlocal any_z_gradient
                    current = self._immutable_forward(lambda: generator_micro(index, start))
                    term_records, objectives, local_z, fake_digest = current
                    if fake_digest != expected[2]:
                        raise ValueError('Generator microbatch replay differs from discovery outputs')
                    step_loss = 0
                    for t, (gf_m, _) in enumerate(term_records):
                        gf_slice = g_cotangents[t][start:start + self.microbatch_size]
                        step_loss = step_loss + (gf_m * gf_slice).sum()
                    step_loss = step_loss + sum(value * factor for value, factor in zip(objectives, objective_factors))
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
            self._reduce_gradients(self.opt_g, 'generator/prior/auxiliary')
            self._phase('generator optimizer', self.opt_g.step)
            self._phase('generator optimizer state', lambda: self._finite_optimizer(self.opt_g))
            self._phase('aggregation step complete', lambda: self.aggregation_strategy.on_step_complete(step))
        finally:
            for module, flags in saved_flags:
                for parameter, flag in zip(module.parameters(), flags):
                    parameter.requires_grad_(flag)
        g_loss = weighted_adversarial.detach() + prior_loss + sum(objective_losses)
        self._phase('EMA', lambda: (update_ema(self.ema_graph, self.graph, settings['ema']), update_ema(self.ema_prior, self.prior, settings['ema'])))
        self._phase('module compatibility', self._check_modules)
        if self.aggregation_strategy.should_assert_replicas(step):
            self._assert_replicas('complete replicated state')
        values = torch.tensor(self._metric_transfer([d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial_raw, d_adversarial, weighted_adversarial,
            *objective_losses]), dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        values /= self.world_size
        self._phase('complete CUDA update', self._synchronize)
        self.step, self.checkpoint_ready = step, True
        self._handshake_complete = True
        values = values.tolist()
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:], lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size,
                   world_size=self.world_size, accumulation_steps=self.accumulation_steps, microbatch_size=self.microbatch_size)
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
