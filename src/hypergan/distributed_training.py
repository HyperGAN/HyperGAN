"""Bounded fixed-world-size CPU/CUDA GAN updates with explicit gradient reduction.

This is an internal replicated strategy, not DDP, a launcher or a run service.
The caller owns the default Gloo or NCCL group, finite timeouts and whole-job lifecycle.
"""
import copy
import hashlib
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist

from .config import config_values, fingerprint, resolve_config
from .distributed import Collectives
from .recipes import detach, move_tensors
from .training import NOISE_SEED_OFFSET, ReferenceTrainer, ScoredCritic, schedule_learning_rates, update_ema


class ReplicatedTrainer(ReferenceTrainer):
    """One complete D/G/prior/auxiliary/EMA update per global batch.

    ``training.batch_size`` is global; explicit batches and latent draws are local
    shards of that batch. Accumulation partitions each local effective batch. Custom scalar
    objectives use the mean of rank-local objective values; global custom batch
    statistics require their own explicit collective implementation.
    """

    def __init__(self, config, *, world_size=2, accumulation_steps=1):
        self.collectives = Collectives(world_size)
        self.device = self.collectives.device
        self.world_size, self.rank = world_size, dist.get_rank()
        self.checkpoint_ready = False
        self._poisoned = False
        config = self._phase('configuration', lambda: resolve_config(config_values(config)))
        from .execution_profiles import validate_replicated_recipe
        self._phase('recipe policy', lambda: validate_replicated_recipe(config))
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
        self.strategy_info = {'name': 'cuda-replicated-nccl' if self.device.type == 'cuda' else 'cpu-replicated-gloo', 'world_size': world_size,
                              'global_batch_size': self.global_batch_size, 'local_batch_size': self.local_batch_size,
                              'accumulation_steps': accumulation_steps, 'microbatch_size': self.microbatch_size,
                              'accumulation_algorithm': 'detached-logit-vjp-replay-v1' if accumulation_steps > 1 else 'retained-local-graph-v1',
                              'gradient_reduction': 'post-backward-mean',
                              'buffers': 'require-replica-equality', 'data': 'replicated-global-draw-rank-slice', 'qualification': 'unqualified'}
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
        self._phase('module compatibility', self._check_modules)
        if accumulation_steps > 1:
            self._phase('accumulation compatibility', self._check_accumulation)
        self._assert_replicas('initial state')
        # Initialize identical models, then give each worker distinct reproducible
        # random streams. Controlled fixtures can supply matched explicit draws.
        rank_seed = (config['training']['seed'] + 104729 * self.rank) % (2 ** 63)
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2 ** 32))
        for offset, name in enumerate(('data', 'prior', 'penalty'), 1):
            self.streams[name].manual_seed(((config['training']['seed'] if name == 'data' else rank_seed) + offset) % (2 ** 63))
        # Replicated recipes have no noise schedule; the stream is still rank-owned.
        self.streams['noise'].manual_seed((rank_seed + NOISE_SEED_OFFSET) % (2 ** 63))
        self._phase('initial CUDA completion', self._synchronize)
        self.checkpoint_ready = True

    def _synchronize(self):
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

    def _phase(self, name, function):
        result, error = None, None
        try:
            result = function()
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self._exchange('phase:' + name, error)
        if any(errors):
            raise RuntimeError(f'{name} failed: ' + '; '.join(f'rank {rank}: {value}' for rank, value in enumerate(errors) if value))
        return result

    def _agree(self, name, value):
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
        self._agree(name, self._phase(name + ' validation', self._replica_digest))

    def batch(self):
        # Duplicate decoding is intentional in this correctness-first strategy:
        # all workers advance ONE identical global sampler before rank slicing.
        full = self._phase('global data draw', lambda: self.data(self.global_batch_size, generator=self.streams['data']))
        self._agree('global data draw', self._phase('global data digest', lambda: self._digest(full)))
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
        batch = self._phase('global data slicing', lambda: move_tensors(shard(full), self.device))
        self._phase('local data validation', lambda: self._validate_batch(batch))
        return batch

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

    def _reduce_gradients(self, optimizer, name):
        parameters = self._parameters(optimizer)
        self._agree(name + ' parameter inventory', [(tuple(p.shape), str(p.dtype), p.requires_grad) for p in parameters])
        def validate():
            for parameter in parameters:
                if parameter.grad is not None and (parameter.grad.layout != torch.strided or not torch.isfinite(parameter.grad).all()):
                    raise ValueError('Nonfinite or sparse gradient; optimizer step refused')
        self._phase(name + ' gradients', validate)
        presence = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int64, device=self.device)
        dist.all_reduce(presence)
        for count, parameter in zip(presence.tolist(), parameters):
            if count:
                gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().contiguous()
                dist.all_reduce(gradient)
                parameter.grad = gradient / self.world_size
            else:
                parameter.grad = None
        self._phase(name + ' reduced gradients', validate)

    def _finite_optimizer(self, optimizer):
        for parameter in self._parameters(optimizer):
            if not torch.isfinite(parameter).all():
                raise ValueError('Optimizer produced nonfinite parameters')
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError('Optimizer produced nonfinite state')

    def _logits(self, real, fake):
        # RpGAN pairs each real with the fake of its own row, so rank-local
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
        scale = schedule_learning_rates(self, step - 1)

        self._agree('input source', {'sample_data': batch is None, 'sample_prior': latent_draw is None})
        if batch is None:
            batch = self.batch()

        def prepare():
            local_batch = batch
            self._validate_batch(local_batch)
            z, ids = self.prior.sample(self.local_batch_size, generator=self.streams['prior']) if latent_draw is None else latent_draw
            if not isinstance(z, torch.Tensor) or z.device != self.device or z.ndim < 1 or len(z) != self.local_batch_size or not torch.isfinite(z).all():
                raise ValueError('Latent draw must contain finite local-batch values on the rank device')
            if ids is not None and (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 1 or len(ids) != self.local_batch_size):
                raise ValueError('Prior IDs must be an int64 vector of local_batch_size')
            context = self.graph.generate(z, local_batch)
            fake, real = context['generated'], local_batch['real']
            if not isinstance(fake, torch.Tensor) or fake.shape != real.shape or not torch.isfinite(fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            return local_batch, context, fake, real, ids
        batch, context, fake, real, ids = self._phase('batch and generator forward', prepare)
        self._agree('real data shape', (tuple(real.shape), str(real.dtype)))
        self._agree('prior kind', ids is None)
        selected = None
        if ids is not None:
            selected = self.collectives.unique_indices(ids, num_rows=len(self.prior.z))
        critic = lambda value: self.graph.critic(value, context)
        self.opt_d.zero_grad(set_to_none=True)
        dr, df, d_penalty = self._phase('discriminator forward and penalty', lambda: (
            critic(real), critic(fake.detach()), self.penalty(self._scored_critic(context), real, fake.detach())))
        dr, df = self._logits(dr, df)
        d_adversarial = self.gan.d_loss(dr, df)
        d_adversarial_weighted = cfg['adversarial']['weight'] * d_adversarial
        d_loss = d_adversarial_weighted + d_penalty
        self._phase('discriminator loss', lambda: self._finite_loss(d_loss))
        self._phase('discriminator backward', d_loss.backward)
        self._reduce_gradients(self.opt_d, 'discriminator')
        self._phase('discriminator optimizer', self.opt_d.step)
        self._phase('discriminator optimizer state', lambda: self._finite_optimizer(self.opt_d))
        discriminator = self.graph.models['discriminator']
        flags = [p.requires_grad for p in discriminator.parameters()]
        discriminator.requires_grad_(False)
        try:
            self.opt_g.zero_grad(set_to_none=True)
            gf, gr = self._phase('generator critic forward', lambda: (critic(fake), critic(real).detach()))
            gr, gf = self._logits(gr, gf)
            g_adversarial = self.gan.g_loss(gf, gr)
            def remaining_losses():
                rows = None if selected is None else self.prior.z if cfg['prior_regularizer']['rows'] == 'full' else self.prior.z[selected]
                prior_loss = self.spread(rows) if rows is not None else fake.new_zeros(())
                objective_losses = []
                for term, objective in zip(cfg['objectives'], self.objectives):
                    inputs = {arg: self.graph.resolve(path, context) for arg, path in term['inputs'].items()}
                    for arg in term['detach']:
                        inputs[arg] = detach(inputs[arg])
                    value = objective(**inputs)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise ValueError('Each objective must return one scalar tensor')
                    objective_losses.append(term['weight'] * value)
                return prior_loss, objective_losses
            prior_loss, objective_losses = self._phase('generator regularizers and objectives', remaining_losses)
            g_adversarial_weighted = cfg['adversarial']['weight'] * g_adversarial
            g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
            self._phase('generator loss', lambda: self._finite_loss(g_loss))
            self._phase('generator backward', g_loss.backward)
            self._reduce_gradients(self.opt_g, 'generator/prior/auxiliary')
            self._phase('generator optimizer', self.opt_g.step)
            self._phase('generator optimizer state', lambda: self._finite_optimizer(self.opt_g))
        finally:
            for parameter, flag in zip(discriminator.parameters(), flags):
                parameter.requires_grad_(flag)
        self._phase('EMA', lambda: (update_ema(self.ema_graph, self.graph, settings['ema']), update_ema(self.ema_prior, self.prior, settings['ema'])))
        self._phase('module compatibility', self._check_modules)
        self._assert_replicas('complete replicated state')
        values = torch.tensor(self._metric_transfer([d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial, d_adversarial_weighted, g_adversarial_weighted,
            *objective_losses]), dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        values /= self.world_size
        self._phase('complete CUDA update', self._synchronize)
        self.step = step
        self.checkpoint_ready = True
        values = values.tolist()
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:], lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size, world_size=self.world_size)
        return row, detach(batch)

    def _scored_critic(self, context):
        return ScoredCritic(self.graph.models['discriminator'],
                            lambda module, value: self.graph.critic(value, context, module=module))

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
        starts = range(0, self.local_batch_size, self.microbatch_size)
        generation_rng, d_records = [], []
        # The penalty record counts calls. Discovery and replay evaluate one
        # logical penalty per microbatch; keep the count of the plain update.
        record, logical_calls = self.opt_d.record, None
        calls_before = record.calls

        def generate(local_z, local_batch):
            context = self.graph.generate(local_z, local_batch)
            fake = context['generated']
            if not isinstance(fake, torch.Tensor) or fake.shape != local_batch['real'].shape or not torch.isfinite(fake).all():
                raise ValueError('Generator output must be finite and match local real data shape')
            return context

        def discriminator_micro(index, start):
            local_batch = self._microbatch(batch, start)
            local_z = z.detach()[start:start + self.microbatch_size].requires_grad_(z.requires_grad)
            context = detach(generate(local_z, local_batch))
            critic = lambda value: self.graph.critic(value, context)
            real, fake = local_batch['real'], context['generated']
            dr, df = critic(real), critic(fake)
            self._micro_logits(dr, df)
            penalty = self.penalty(self._scored_critic(context), real, fake)
            return dr, df, penalty, self._digest(fake)

        # Prepasses use ordinary grad mode and immediately discard each graph,
        # so custom grad-mode behavior is the same on the checked replay.
        for index, start in enumerate(starts):
            rng = self._accumulation_rng()
            generation_rng.append(rng)
            def prepass():
                dr, df, penalty, fake_digest = self._immutable_forward(lambda: discriminator_micro(index, start))
                self._finite_loss(penalty)
                return dr.detach(), df.detach(), penalty.detach(), fake_digest
            d_records.append((rng, self._phase(f'D micro {index} discovery', prepass)))
            if logical_calls is None:
                logical_calls = record.calls
        dr = torch.cat([record[1][0] for record in d_records]).requires_grad_(True)
        df = torch.cat([record[1][1] for record in d_records]).requires_grad_(True)
        global_dr, global_df = self._logits(dr, df)
        d_adversarial_raw = self.gan.d_loss(global_dr, global_df)
        d_adversarial = cfg['adversarial']['weight'] * d_adversarial_raw
        self._phase('accumulated D objective', lambda: self._finite_loss(d_adversarial))
        d_cotangents = self._phase('accumulated D logit derivatives', lambda: torch.autograd.grad(d_adversarial, (dr, df)))
        d_penalty = sum(record[1][2] / self.accumulation_steps for record in d_records)
        d_loss = d_adversarial.detach() + d_penalty
        self._phase('accumulated D total loss', lambda: self._finite_loss(d_loss))
        self.opt_d.zero_grad(set_to_none=True)
        for index, start in enumerate(starts):
            rng, expected = d_records[index]
            def replay():
                current = self._immutable_forward(lambda: discriminator_micro(index, start))
                if self._digest(detach(current)) != self._digest(expected):
                    raise ValueError('Discriminator microbatch replay differs from discovery outputs')
                dr_micro, df_micro, penalty, _ = current
                terms = (dr_micro * d_cotangents[0][start:start + self.microbatch_size]).sum()
                terms = terms + (df_micro * d_cotangents[1][start:start + self.microbatch_size]).sum() + penalty / self.accumulation_steps
                terms.backward()
            self._phase(f'D micro {index} replay', lambda: self._isolated_replay(rng, replay))
        record.calls = calls_before if logical_calls is None else logical_calls
        self._reduce_gradients(self.opt_d, 'discriminator')
        self._phase('discriminator optimizer', self.opt_d.step)
        self._phase('discriminator optimizer state', lambda: self._finite_optimizer(self.opt_d))
        discriminator = self.graph.models['discriminator']
        flags = [p.requires_grad for p in discriminator.parameters()]
        discriminator.requires_grad_(False)
        try:
            def generator_micro(index, start):
                local_batch = self._microbatch(batch, start)
                local_z = z.detach()[start:start + self.microbatch_size].requires_grad_(z.requires_grad)
                # Recreate precisely the fake used during D, without consuming a
                # second generator RNG trajectory in the logical update.
                context = self._isolated_replay(generation_rng[index], lambda: generate(local_z, local_batch))
                fake, real = context['generated'], local_batch['real']
                if self._digest(fake) != d_records[index][1][3]:
                    raise ValueError('Generator replay did not reproduce the fake used by the discriminator')
                gf, gr = self.graph.critic(fake, context), self.graph.critic(real, context).detach()
                self._micro_logits(gf, gr)
                objectives = []
                for term, objective in zip(cfg['objectives'], self.objectives):
                    inputs = {arg: self.graph.resolve(path, context) for arg, path in term['inputs'].items()}
                    for arg in term['detach']:
                        inputs[arg] = detach(inputs[arg])
                    value = objective(**inputs)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise ValueError('Each objective must return one scalar tensor')
                    objectives.append(term['weight'] * value)
                return gf, gr, objectives, local_z, self._digest(fake)

            g_records = []
            for index, start in enumerate(starts):
                rng = self._accumulation_rng()
                def prepass():
                    gf, gr, terms, _, fake_digest = self._immutable_forward(lambda: generator_micro(index, start))
                    for value in terms:
                        self._finite_loss(value)
                    return gf.detach(), gr.detach(), detach(terms), fake_digest
                g_records.append((rng, self._phase(f'G micro {index} discovery', prepass)))
            gf = torch.cat([record[1][0] for record in g_records]).requires_grad_(True)
            gr = torch.cat([record[1][1] for record in g_records])
            global_gr, global_gf = self._logits(gr, gf)
            g_adversarial = self.gan.g_loss(global_gf, global_gr)
            weighted_adversarial = cfg['adversarial']['weight'] * g_adversarial
            self._phase('accumulated G objective', lambda: self._finite_loss(weighted_adversarial))
            g_cotangent, = self._phase('accumulated G logit derivatives', lambda: torch.autograd.grad(weighted_adversarial, (gf,)))
            objective_factors = [1 / self.accumulation_steps if reduction == 'mean' else 1 for reduction in self._objective_reductions]
            objective_losses = [sum(record[1][2][i] * factor for record in g_records) for i, factor in enumerate(objective_factors)]
            self.opt_g.zero_grad(set_to_none=True)
            z_cotangent = torch.zeros_like(z) if z.requires_grad else None
            any_z_gradient = False
            for index, start in enumerate(starts):
                rng, expected = g_records[index]
                def replay():
                    nonlocal any_z_gradient
                    current = self._immutable_forward(lambda: generator_micro(index, start))
                    gf_micro, gr_micro, objectives, local_z, fake_digest = current
                    if self._digest(detach((gf_micro, gr_micro, objectives, fake_digest))) != self._digest(expected):
                        raise ValueError('Generator microbatch replay differs from discovery outputs')
                    loss = (gf_micro * g_cotangent[start:start + self.microbatch_size]).sum()
                    loss = loss + sum(value * factor for value, factor in zip(objectives, objective_factors))
                    loss.backward()
                    if z_cotangent is not None and local_z.grad is not None:
                        any_z_gradient = True
                        z_cotangent[start:start + self.microbatch_size].copy_(local_z.grad)
                self._phase(f'G micro {index} replay', lambda: self._isolated_replay(rng, replay))
            def prior_backward():
                rows = None if selected is None else self.prior.z if cfg['prior_regularizer']['rows'] == 'full' else self.prior.z[selected]
                prior_loss = self.spread(rows) if rows is not None else z.new_zeros(())
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
        finally:
            for parameter, flag in zip(discriminator.parameters(), flags):
                parameter.requires_grad_(flag)
        g_loss = weighted_adversarial.detach() + prior_loss + sum(objective_losses)
        self._phase('EMA', lambda: (update_ema(self.ema_graph, self.graph, settings['ema']), update_ema(self.ema_prior, self.prior, settings['ema'])))
        self._phase('module compatibility', self._check_modules)
        self._assert_replicas('complete replicated state')
        values = torch.tensor(self._metric_transfer([d_loss, g_loss, g_adversarial, prior_loss,
            d_penalty, d_adversarial_raw, d_adversarial, weighted_adversarial,
            *objective_losses]), dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        values /= self.world_size
        self._phase('complete CUDA update', self._synchronize)
        self.step, self.checkpoint_ready = step, True
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

    def __init__(self, config, *, world_size=2, accumulation_steps=1):
        if dist.is_initialized() and dist.get_backend() != 'gloo':
            raise ValueError('ReplicatedCPUTrainer requires the default Gloo group')
        super().__init__(config, world_size=world_size, accumulation_steps=accumulation_steps)
