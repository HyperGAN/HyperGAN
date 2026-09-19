"""Bounded fixed-world-size CPU GAN updates with explicit gradient reduction.

This is an internal replicated strategy, not DDP, a launcher or a run service.
The caller owns the default Gloo group, finite timeouts and whole-job lifecycle.
"""
import hashlib
import random

import numpy as np
import torch
import torch.distributed as dist
from particlegan import learning_rate_scale

from .config import config_values, fingerprint, resolve_config
from .distributed import GlooCollectives
from .recipes import detach
from .training import ReferenceTrainer, update_ema


class ReplicatedCPUTrainer(ReferenceTrainer):
    """One complete D/G/prior/auxiliary/EMA update per global batch.

    ``training.batch_size`` is global; explicit batches and latent draws are local
    shards of that batch. Accumulation is currently exactly one. Custom scalar
    objectives use the mean of rank-local objective values; global custom batch
    statistics require their own explicit collective implementation.
    """

    def __init__(self, config, *, world_size=2, accumulation_steps=1):
        self.collectives = GlooCollectives(world_size)
        self.world_size, self.rank = world_size, dist.get_rank()
        self.checkpoint_ready = False
        self._poisoned = False
        config = self._phase('configuration', lambda: resolve_config(config_values(config)))
        self._agree('configuration', {'fingerprint': fingerprint(config), 'accumulation_steps': accumulation_steps})
        if world_size < 2:
            raise ValueError('Replicated CPU training requires at least two ranks')
        if type(accumulation_steps) is not int or accumulation_steps != 1:
            raise ValueError('This strategy supports accumulation_steps=1 only; nonlinear global objectives cannot be averaged across independent microbatches')
        self.global_batch_size = config['training']['batch_size']
        if self.global_batch_size % world_size:
            raise ValueError('Global training.batch_size must divide evenly across world_size')
        self.local_batch_size = self.global_batch_size // world_size
        self.strategy_info = {'name': 'cpu-replicated-gloo', 'world_size': world_size,
                              'global_batch_size': self.global_batch_size, 'local_batch_size': self.local_batch_size,
                              'accumulation_steps': 1, 'gradient_reduction': 'post-backward-mean',
                              'buffers': 'require-replica-equality', 'data': 'replicated-global-draw-rank-slice', 'qualification': 'unqualified'}
        torch.set_num_threads(1)
        self._phase('initialization', lambda: super(ReplicatedCPUTrainer, self).__init__(config))
        self._phase('module compatibility', self._check_modules)
        self._assert_replicas('initial state')
        # Initialize identical models, then give each worker distinct reproducible
        # random streams. Controlled fixtures can supply matched explicit draws.
        rank_seed = (config['training']['seed'] + 104729 * self.rank) % (2 ** 63)
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2 ** 32))
        for offset, name in enumerate(('data', 'prior', 'penalty'), 1):
            self.streams[name].manual_seed(((config['training']['seed'] if name == 'data' else rank_seed) + offset) % (2 ** 63))
        self.checkpoint_ready = True

    def _exchange(self, operation, value):
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
            if any(value.device.type != 'cpu' or value.layout != torch.strided for value in (*module.parameters(), *module.buffers())):
                raise ValueError('Replicated CPU modules must own dense CPU parameters and buffers')
            if any(not torch.isfinite(value).all() for value in module.parameters()):
                raise ValueError('Replicated CPU parameters must be finite')

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
        batch = self._phase('global data slicing', lambda: shard(full))
        self._phase('local data validation', lambda: self._validate_batch(batch))
        return batch

    def _validate_batch(self, batch):
        if not isinstance(batch, dict) or not isinstance(batch.get('real'), torch.Tensor):
            raise ValueError("Data must return a dictionary containing tensor 'real'")
        real = batch['real']
        if real.device.type != 'cpu' or not real.is_floating_point() or real.ndim < 1 or len(real) != self.local_batch_size:
            raise ValueError('Real data must be floating CPU tensors of local_batch_size')
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
        presence = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int64)
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
        if self.config['adversarial']['mode'] == 'ra':
            # Run the exact public upstream kernel over global logits. Its means
            # are differentiable; gather backward sums contributions, and the
            # later parameter-gradient mean cancels replicated loss consumption.
            return self.collectives.gather(real), self.collectives.gather(fake)
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
            return self._update(batch, latent_draw)
        except BaseException:
            self._poisoned = True
            raise

    def _update(self, batch, latent_draw):
        cfg, step = self.config, self.step + 1
        settings = cfg['training']
        scale = learning_rate_scale(step - 1, settings['steps'], start=settings['lr_anneal_start'], floor=settings['lr_floor'])
        for optimizer, rates in zip((self.opt_g, self.opt_d), self.base_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate * scale

        self._agree('input source', {'sample_data': batch is None, 'sample_prior': latent_draw is None})
        if batch is None:
            batch = self.batch()

        def prepare():
            local_batch = batch
            self._validate_batch(local_batch)
            z, ids = self.prior.sample(self.local_batch_size, generator=self.streams['prior']) if latent_draw is None else latent_draw
            if not isinstance(z, torch.Tensor) or z.device.type != 'cpu' or z.ndim < 1 or len(z) != self.local_batch_size or not torch.isfinite(z).all():
                raise ValueError('Latent draw must contain finite CPU local-batch values')
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
            critic(real), critic(fake.detach()), self.penalty(critic, real, fake.detach(), step=step, generator=self.streams['penalty'])))
        dr, df = self._logits(dr, df)
        d_adversarial = self.gan.d_loss(dr, df)
        d_loss = cfg['adversarial']['weight'] * d_adversarial + d_penalty
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
            g_loss = cfg['adversarial']['weight'] * g_adversarial + prior_loss + sum(objective_losses)
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
        values = torch.tensor([float(value.detach()) for value in [d_loss, g_loss, g_adversarial, prior_loss, d_penalty, *objective_losses]], dtype=torch.float64)
        dist.all_reduce(values)
        values /= self.world_size
        self.step = step
        self.checkpoint_ready = True
        row = dict(zip(('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty'), values[:5].tolist()))
        row.update(event='train', step=step, objectives=values[5:].tolist(), lr_scale=scale,
                   global_batch_size=self.global_batch_size, local_batch_size=self.local_batch_size, world_size=self.world_size)
        return row, detach(batch)

    @staticmethod
    def _finite_loss(value):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite loss; optimizer step refused')
