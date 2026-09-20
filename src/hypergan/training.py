"""HyperGAN-owned single-process CPU/CUDA reference loop.

Fixed-runtime recovery is supported; image quality and DDP require separate qualification.
"""
import copy
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import platform
import random
import inspect
import os

import numpy as np

import torch
from particlegan import GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale

from .recipes import ComponentGraph, construct, detach, make_prior, execution_device, move_tensors
from .checkpoints import data_contract
from .numerical_policy import apply_backend_policy


class _MetricScalarTransfer:
    """Stage scalar observations with one CUDA stream fence and no GPU kernels.

    Reusable pinned buffers are local to this trainer and grouped by source dtype
    so mixed objectives retain their exact Python-float conversion. Copies remain
    one per scalar; only the redundant host waits are removed. CPU fixtures use
    the direct conversion without initializing CUDA or allocating pinned memory.
    """
    def __init__(self, device):
        self.device = device
        self._signature = None
        self._groups = []

    def __call__(self, values):
        if self.device.type != 'cuda':
            return [float(value.detach()) for value in values]
        signature = tuple(value.dtype for value in values)
        if signature != self._signature:
            positions = {}
            for index, dtype in enumerate(signature):
                positions.setdefault(dtype, []).append(index)
            self._groups = []
            for dtype, indexes in positions.items():
                buffer = torch.empty(len(indexes), dtype=dtype, device='cpu', pin_memory=True)
                self._groups.append((indexes, buffer, buffer.unbind()))
            self._signature = signature
        for indexes, buffer, destinations in self._groups:
            for index, destination in zip(indexes, destinations):
                value = values[index].detach()
                if value.device.type == 'cuda' and value.device != self.device:
                    raise ValueError('Metric scalar is on a different CUDA device from its trainer')
                if value.ndim:
                    value = value.reshape(())
                destination.copy_(value, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        result = [None] * len(values)
        for indexes, buffer, _ in self._groups:
            for index, value in zip(indexes, buffer.tolist()):
                result[index] = float(value)
        return result


@torch.no_grad()
def update_ema(average, current, decay):
    for target, source in zip(average.parameters(), current.parameters()):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(average.buffers(), current.buffers()):
        target.copy_(source)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown (source import without distribution metadata)"


def runtime_info(device='cpu'):
    device = execution_device(device)
    result = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.system(), "machine": platform.machine(), "threads": 1, "particlegan": _version("particlegan"), "hypergan": _version("hypergan"), "device": str(device), "dtype": "float32", "world_size": 1, "default_dtype": str(torch.get_default_dtype()), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result['cuda'] = {'version': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'name': properties.name, 'capability': list(torch.cuda.get_device_capability(device)),
            'uuid': str(getattr(properties, 'uuid', 'unavailable')),
            'visible_devices': [str(getattr(torch.cuda.get_device_properties(index), 'uuid', 'unavailable')) for index in range(torch.cuda.device_count())],
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    return result


def source_info():
    from .provenance import hypergan_source
    result = {"integration_reference": {"repository": "https://github.com/255BITS/ParticleGAN", "commit": "f946b4ed468ff3b3eae5a3bca11411d5725f1181"}, "particlegan_distribution_version": _version("particlegan"), "particlegan_distribution_commit": None, **hypergan_source()}
    result["distribution_records"] = {}
    for name in ("hypergan", "particlegan"):
        try:
            distribution = importlib.metadata.distribution(name)
            record = distribution.read_text("RECORD")
            result["distribution_records"][name] = {"record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None}
        except importlib.metadata.PackageNotFoundError:
            result["distribution_records"][name] = None
    return result


class DeviceAdam(torch.optim.Adam):
    """CPU parameters cannot be CUDA-graph captured; avoid initializing a GPU.

    Torch 2.14's generic capture guard queries the accelerator even for an
    entirely CPU optimizer. Keep its normal guard for accelerator parameters.
    """
    def _accelerator_graph_capture_health_check(self):
        if any(parameter.device.type != 'cpu' for group in self.param_groups for parameter in group['params']):
            return super()._accelerator_graph_capture_health_check()

    def _cuda_graph_capture_health_check(self):
        if any(parameter.device.type != 'cpu' for group in self.param_groups for parameter in group['params']):
            return super()._cuda_graph_capture_health_check()


class ReferenceTrainer:
    """Small inspectable state machine; future distributed strategies wrap this contract."""
    def __init__(self, config):
        apply_backend_policy(config)
        self.device = execution_device(config['training']['device'])
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                self._initialize(config)
        else:
            self._initialize(config)

    def _initialize(self, config):
        self.config = config
        settings = config["training"]
        torch.manual_seed(settings["seed"])
        random.seed(settings["seed"])
        np.random.seed(settings["seed"] % (2 ** 32))
        self.graph = ComponentGraph(config["components"]).float().to(self.device)
        self.prior = make_prior(config["prior"], device=self.device).float()
        self.data = construct(config["data"])
        self.gan = GANLoss(**{k: v for k, v in config["adversarial"].items() if k != "weight"})
        self.penalty = GradientPenalty(**config["gradient_penalty"])
        self.spread = ParticleRegularizer(**{k: v for k, v in config["prior_regularizer"].items() if k != "rows"})
        self.objectives = [construct(term) for term in config["objectives"]]
        for objective in self.objectives:
            if isinstance(objective, torch.nn.Module):
                objective.to(self.device)
        if any(isinstance(term, torch.nn.Module) and any(p.requires_grad for p in term.parameters()) for term in self.objectives):
            raise ValueError("Objective constructors must not own trainable parameters; declare trainable transforms as components and bind their outputs into an objective")
        if any(isinstance(term, torch.nn.Module) and list(term.buffers()) for term in self.objectives):
            raise ValueError("Stateful objective buffers are not supported by this reference loop; declare stateful transforms as components")
        opt = config["optimizer"]
        groups = [{"params": self.graph.generator_parameters(), "lr": opt["lr"]}]
        prior_parameters = [p for p in self.prior.parameters() if p.requires_grad]
        if prior_parameters:
            groups.append({"params": prior_parameters, "lr": opt["lr"] * opt["prior_lr_mult"], "betas": tuple(opt["prior_betas"])})
        d_parameters = [p for p in self.graph.models["discriminator"].parameters() if p.requires_grad]
        if not groups[0]["params"] or not d_parameters:
            raise ValueError("The reference adversarial loop requires trainable generator and discriminator parameters")
        optimizer_options = {'fused': True} if opt['implementation'] == 'torch_fused_adam' else {}
        self.opt_g = DeviceAdam(groups, betas=tuple(opt["betas"]), **optimizer_options)
        self.opt_d = DeviceAdam(d_parameters, lr=opt["lr"] * opt["d_lr_mult"], betas=tuple(opt["betas"]), **optimizer_options)
        self.base_lrs = [[g["lr"] for g in optimizer.param_groups] for optimizer in (self.opt_g, self.opt_d)]
        self.ema_graph = copy.deepcopy(self.graph).eval().requires_grad_(False)
        self.ema_prior = copy.deepcopy(self.prior).eval().requires_grad_(False)
        data_device = self.device if settings['data_rng_device'] == 'execution' else 'cpu'
        self.streams = {"data": torch.Generator(device=data_device).manual_seed(settings["seed"] + settings['data_seed_offset']), "prior": torch.Generator(device=self.device).manual_seed(settings["seed"] + settings['prior_seed_offset']), "penalty": torch.Generator(device=self.device).manual_seed(settings["seed"] + 3)}
        self.step = 0
        self._metric_transfer = _MetricScalarTransfer(self.device)

    def batch(self):
        batch = self.data(self.config["training"]["batch_size"], generator=self.streams["data"])
        if not isinstance(batch, dict) or not isinstance(batch.get("real"), torch.Tensor):
            raise ValueError("Data constructor must return a callable producing a dict containing tensor 'real'")
        if not batch["real"].is_floating_point():
            raise ValueError("Reference real data must be floating-point tensors")
        return move_tensors(batch, self.device)

    def update(self, batch=None, latent_draw=None, *, generator_batch=None, generator_latent_draw=None):
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                result = self._update(batch, latent_draw, generator_batch=generator_batch, generator_latent_draw=generator_latent_draw)
                torch.cuda.synchronize(self.device)
                return result
        return self._update(batch, latent_draw, generator_batch=generator_batch, generator_latent_draw=generator_latent_draw)

    def _draw(self, batch, latent_draw):
        batch = self.batch() if batch is None else move_tensors(batch, self.device)
        if len(batch['real']) != self.config['training']['batch_size']:
            raise ValueError('Data batch length must match training.batch_size')
        z, ids = self.prior.sample(len(batch['real']), generator=self.streams['prior']) if latent_draw is None else move_tensors(latent_draw, self.device)
        context = self.graph.generate(z, batch, prior=self.prior)
        if not isinstance(context['generated'], torch.Tensor) or context['generated'].shape != batch['real'].shape:
            raise ValueError(f"Generator output must match real data shape {tuple(batch['real'].shape)}")
        return batch, ids, context

    def _update(self, batch=None, latent_draw=None, *, generator_batch=None, generator_latent_draw=None):
        """Execute one D update followed by G/prior/aux update and matched EMA.

        Explicit batch and (latent, indices) enable controlled numerical comparisons.
        """
        cfg = self.config
        settings = cfg["training"]
        independent = settings['phase_draws'] == 'independent'
        if not independent and (generator_batch is not None or generator_latent_draw is not None):
            raise ValueError('Explicit generator phase draws require training.phase_draws=independent')
        step = self.step + 1
        scale = learning_rate_scale(step - 1, settings["steps"], start=settings["lr_anneal_start"], floor=settings["lr_floor"])
        for optimizer, rates in zip((self.opt_g, self.opt_d), self.base_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * scale
        if independent:
            with torch.no_grad():
                batch, ids, context = self._draw(batch, latent_draw)
        else:
            batch, ids, context = self._draw(batch, latent_draw)
        fake, real = context["generated"], batch["real"]
        critic = lambda x: self.graph.critic(x, context)
        self.opt_d.zero_grad(set_to_none=True)
        d_adversarial = self.gan.d_loss(critic(real), critic(fake.detach()))
        d_penalty = self.penalty(critic, real, fake.detach(), step=step, generator=self.streams["penalty"])
        d_adversarial_weighted = cfg["adversarial"]["weight"] * d_adversarial
        d_loss = d_adversarial_weighted + d_penalty
        if not torch.isfinite(d_loss).all():
            raise ValueError("Nonfinite discriminator loss; run stopped")
        d_loss.backward()
        self._check_gradients(self.graph.models["discriminator"].parameters(), "discriminator")
        self.opt_d.step()
        discriminator = self.graph.models["discriminator"]
        flags = [p.requires_grad for p in discriminator.parameters()]
        discriminator.requires_grad_(False)
        try:
            if independent:
                batch, ids, context = self._draw(generator_batch, generator_latent_draw)
                fake, real = context['generated'], batch['real']
            self.opt_g.zero_grad(set_to_none=True)
            g_adversarial = self.gan.g_loss(critic(fake), critic(real).detach())
            if ids is None:
                prior_loss = fake.new_zeros(())
            else:
                rows = self.prior.z if cfg["prior_regularizer"]["rows"] == "full" else self.prior.z[ids.unique()]
                prior_loss = self.spread(rows)
            objective_losses = []
            for term, objective in zip(cfg["objectives"], self.objectives):
                inputs = {arg: self.graph.resolve(path, context) for arg, path in term["inputs"].items()}
                for arg in term["detach"]:
                    inputs[arg] = detach(inputs[arg])
                value = objective(**inputs)
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError("Each objective must return one scalar tensor")
                objective_losses.append(term["weight"] * value)
            g_adversarial_weighted = cfg["adversarial"]["weight"] * g_adversarial
            g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
            if not torch.isfinite(g_loss).all():
                raise ValueError("Nonfinite generator loss; run stopped")
            g_loss.backward()
            self._check_gradients(self.graph.generator_parameters(), "generator/auxiliary")
            self._check_gradients(self.prior.parameters(), "prior")
            self.opt_g.step()
        finally:
            for parameter, flag in zip(discriminator.parameters(), flags):
                parameter.requires_grad_(flag)
        update_ema(self.ema_graph, self.graph, settings["ema"])
        update_ema(self.ema_prior, self.prior, settings["ema"])
        self.step = step
        values = self._metric_transfer([d_loss, d_adversarial, d_adversarial_weighted,
            g_adversarial_weighted, g_loss, g_adversarial, prior_loss, d_penalty,
            *objective_losses])
        row = dict(zip(('d_loss', 'd_adversarial', 'd_adversarial_weighted',
                        'g_adversarial_weighted', 'g_loss', 'g_adversarial',
                        'prior_loss', 'gradient_penalty'), values[:8]))
        row.update(event='train', step=step, objectives=values[8:], lr_scale=scale)
        return row, detach(batch)

    @staticmethod
    def _check_gradients(parameters, name):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f"Nonfinite {name} gradient; run stopped")


def _implementation(trainer):
    """Hash imported implementation bytes without depending on checkout location."""
    import hypergan.checkpoints
    import hypergan.config
    import hypergan.metrics
    import hypergan.recipes
    import hypergan.run_controller
    import hypergan.single_execution
    import hypergan.numerical_policy
    objects = [hypergan.checkpoints, hypergan.config, hypergan.metrics, hypergan.recipes,
               hypergan.run_controller, hypergan.single_execution, hypergan.numerical_policy, ReferenceTrainer,
               GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale,
               type(trainer.data), type(trainer.prior), *[type(x) for x in trainer.graph.modules()],
               *[x if inspect.isfunction(x) else type(x) for x in trainer.objectives]]
    specifications = [trainer.config['data'], *trainer.config['components'].values(), *trainer.config['objectives']]
    for specification in specifications:
        factory = specification['factory']
        if ':' in factory:
            objects.append(importlib.import_module(factory.split(':', 1)[0]))
    result = {}
    for obj in objects:
        module = inspect.getmodule(obj)
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            result[module.__name__] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def _recovery_contract(trainer):
    contract = data_contract(trainer.data, trainer.config['data'])
    reasons = []
    if not contract['supported']:
        reasons.append('Custom data must declare resume_stateless=True or paired state_dict/load_state_dict methods for recovery')
    for spec, objective in zip(trainer.config['objectives'], trainer.objectives):
        if spec['factory'] not in ('mse', 'l1') and getattr(objective, 'resume_stateless', False) is not True:
            reasons.append('Custom objectives must declare resume_stateless=True for recovery; move mutable state into registered component buffers')
    return contract, reasons


def train(config_path, run_dir, steps=None, *, checkpoint_every=100, max_seconds=None,
          stop_after_steps=None, on_event=None, preview_every=0, preview_keep=3):
    """Create a run; budgets stop only at complete D/G/EMA update boundaries."""
    from .run_controller import run_train
    from .single_execution import SingleProcessExecution
    return run_train(config_path, run_dir, steps, checkpoint_every=checkpoint_every,
                     max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
                     preview_every=preview_every, preview_keep=preview_keep,
                     execution_factory=SingleProcessExecution)


def resume(run_dir, checkpoint=None, config_path=None, *, checkpoint_every=None,
           max_seconds=None, stop_after_steps=None, on_event=None, preview_every=None, preview_keep=None,
           steps=None, require_same_config=False):
    """Resume a full checkpoint from this run, retaining the original schedule."""
    from .run_controller import run_resume
    from .single_execution import SingleProcessExecution
    return run_resume(run_dir, checkpoint, config_path, checkpoint_every=checkpoint_every,
                      max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
                      preview_every=preview_every, preview_keep=preview_keep,
                      steps=steps, require_same_config=require_same_config,
                      execution_factory=SingleProcessExecution)
