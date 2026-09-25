"""HyperGAN-owned single-process CPU/CUDA reference loop.

Fixed-runtime recovery is supported; image quality and DDP require separate qualification.
"""
import copy
import functools
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
from particlegan import GANLoss, ParticlePrior, ParticleRegularizer, Recipe, learning_rate_scales
from particlegan.grad_regularizers import GradientPenalty
from particlegan.training import input_noise_std, output_noise_std

from .recipes import ComponentGraph, construct, make_prior, execution_device, move_tensors
from .checkpoints import data_contract
from .numerical_policy import apply_backend_policy
from .objective_program import compile_legacy_program, run_native_program


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


def _paired_groups(targets, sources):
    """Pair tensors positionally, grouped so one foreach call sees one kind.

    ``zip`` keeps the previous per-tensor pairing; the grouping key is every
    property a foreach fast route requires to be uniform, so mixed device or
    dtype inventories simply produce more groups instead of a silent fallback.
    """
    groups = {}
    for target, source in zip(targets, sources):
        key = (target.device, target.dtype, source.device, source.dtype)
        group = groups.setdefault(key, ([], []))
        group[0].append(target)
        group[1].append(source)
    return list(groups.values())


@torch.no_grad()
def update_ema(average, current, decay):
    """Fused EMA update; bitwise identical to the previous per-tensor loop.

    ``torch._foreach_lerp_``/``torch._foreach_copy_`` evaluate the same element
    math as ``Tensor.lerp_``/``Tensor.copy_`` while issuing one launch per group
    instead of one per tensor. Integer buffers such as BatchNorm
    ``num_batches_tracked`` copy unchanged; empty inventories are skipped
    because the foreach operators reject empty tensor lists.
    """
    weight = 1.0 - decay
    for targets, sources in _paired_groups(average.parameters(), current.parameters()):
        torch._foreach_lerp_(targets, sources, weight)
    for targets, sources in _paired_groups(average.buffers(), current.buffers()):
        torch._foreach_copy_(targets, sources)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown (source import without distribution metadata)"


def runtime_info(device='cpu'):
    device = execution_device(device)
    result = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.system(), "machine": platform.machine(), "threads": 1, "particlegan": _version("particlegan"), "hypergan": _version("hypergan"), "hndl": _version("hndl"), "device": str(device), "dtype": "float32", "world_size": 1, "default_dtype": str(torch.get_default_dtype()), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
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
    result = {"integration_reference": {"repository": "https://github.com/255BITS/ParticleGAN", "commit": "407c2f7aad143badfa92ef4e7d5281b95b946542"}, "particlegan_distribution_version": _version("particlegan"), "particlegan_distribution_commit": None, **hypergan_source()}
    result["distribution_records"] = {}
    for name in ("hypergan", "particlegan", "hndl"):
        try:
            distribution = importlib.metadata.distribution(name)
            record = distribution.read_text("RECORD")
            result["distribution_records"][name] = {"record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None}
        except importlib.metadata.PackageNotFoundError:
            result["distribution_records"][name] = None
    return result


class DeviceAdam:
    """Mixin for the recipe-built Adam optimizers the trainer steps.

    CPU parameters cannot be CUDA-graph captured; avoid initializing a GPU.
    Torch 2.14's generic capture guard queries the accelerator even for an
    entirely CPU optimizer. Keep its normal guard for accelerator parameters.
    ``step`` is the recipe optimizer's own step, named here so tests can
    intercept every trainer optimizer step in one place.
    """
    def step(self, closure=None):
        return super().step(closure)

    def _accelerator_graph_capture_health_check(self):
        if any(parameter.device.type != 'cpu' for group in self.param_groups for parameter in group['params']):
            return super()._accelerator_graph_capture_health_check()

    def _cuda_graph_capture_health_check(self):
        if any(parameter.device.type != 'cpu' for group in self.param_groups for parameter in group['params']):
            return super()._cuda_graph_capture_health_check()


def _device_step(self, closure=None):
    return DeviceAdam.step(self, closure)


# The recipe optimizer's step is already wrapped with torch's step hooks.
# Marked so torch does not wrap this class's step again (Optimizer.__setstate__
# does, e.g. on deepcopy), which would run the hooks twice and pin the step
# function that ``DeviceAdam.step`` interception relies on looking up late.
_device_step.hooked = True


@functools.cache
def _device_class(cls):
    return type(f'Device{cls.__name__}', (DeviceAdam, cls), {'__module__': __name__, 'step': _device_step})


def _device_optimizer(optimizer):
    """Mix ``DeviceAdam`` into a recipe-built Adam subclass; its update is unchanged."""
    optimizer.__class__ = _device_class(type(optimizer))
    return optimizer


def particlegan_recipe(config):
    """The ParticleGAN recipe holding this configuration's training formulation.

    Only update settings are set. Model-shape fields (z_dim, prior kind,
    particle count) keep their defaults and are unused: HyperGAN builds the
    networks and prior from its own configuration.
    """
    opt, penalty, training = config['optimizer'], config['gradient_penalty'], config['training']
    return Recipe(
        total_steps=training['steps'], batch_size=training['batch_size'],
        lr=opt['lr'], d_lr_mult=opt['d_lr_mult'], prior_lr_mult=opt['prior_lr_mult'],
        betas=tuple(opt['betas']), prior_betas=tuple(opt['prior_betas']),
        d_guard_ratio=opt['d_guard_ratio'], d_guard_min_steps=opt['d_guard_min_steps'],
        latent_damping_max_rate=opt['latent_damping_max_rate'],
        reg_coeff=penalty['coeff'], reg_kappa=penalty['kappa'], reg_every=penalty['lazy_k'],
        reg_anchor_weight=penalty['anchor_weight'], reg_anchor_decay=penalty['anchor_decay'],
        prior_reg=config['prior_regularizer']['weight'], ema_decay=training['ema'],
        lr_anneal_start=training['lr_anneal_start'], lr_floor=training['lr_floor'],
        network_lr_floor=training['network_lr_floor'], network_lr_horizon_cap=training['network_lr_horizon_cap'],
        input_noise_std=training['input_noise_std'], input_noise_anneal_end=training['input_noise_anneal_end'],
        output_noise_std=training['output_noise_std'], output_noise_warmup=training['output_noise_warmup'])


class ScoredCritic(torch.nn.Module):
    """``score(critic, x)`` as a module whose only child is ``critic``.

    The ParticleGAN critic penalty evaluates the EMA critic by swapping this
    child, so candidate routing, conditioning and input noise are the same for
    the live critic and its EMA.
    """
    def __init__(self, critic, score):
        super().__init__()
        self.critic = critic
        self.score = score

    def forward(self, candidate):
        return self.score(self.critic, candidate)


def with_noise(value, std, generator):
    """``value + std * eps`` from ``generator``; no draw when ``std`` is zero."""
    if std == 0:
        return value
    return value + std * torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)


def schedule_learning_rates(trainer, completed_steps):
    """Set every group's LR for the next update; return the network multiplier.

    Generator and critic groups follow the network schedule and the prior
    group the prior schedule of ``particlegan.learning_rate_scales``. The
    critic penalty's handover reads the critic LR this sets.
    """
    network, prior = learning_rate_scales(completed_steps, trainer.recipe)
    for optimizer, rates, roles in zip((trainer.opt_g, trainer.opt_d), trainer.base_lrs, trainer.lr_roles):
        for group, rate, role in zip(optimizer.param_groups, rates, roles):
            group['lr'] = rate * (prior if role == 'prior' else network)
    return network


def noise_levels(trainer, completed_steps):
    """(critic input, generator output) noise std for the next update."""
    return input_noise_std(trainer.recipe, completed_steps), output_noise_std(trainer.recipe, completed_steps)


# The penalty stream is reserved (the K3P penalty draws no randomness); input
# and output noise draw from their own stream.
NOISE_SEED_OFFSET = 5


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
        self.recipe = particlegan_recipe(config)
        self.gan = self.recipe.make_loss()
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
        optimizer_options = {'fused': True} if opt['implementation'] == 'torch_fused_adam' else {}
        # Every scoring module, in term order, under one root: one critic
        # optimizer, one EMA critic (the penalty's anchor) and one handover.
        names = ["discriminator"] + [term["component"] for term in config.get("adversarial_terms") or ()]
        self.critic = torch.nn.ModuleDict({name: self.graph.models[name] for name in dict.fromkeys(names)})
        ema_critic = copy.deepcopy(self.critic) if config["gradient_penalty"]["anchor_weight"] else None
        self.opt_d = _device_optimizer(self.recipe.make_critic_optimizer(self.critic, ema_critic=ema_critic, **optimizer_options))
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        make_penalty = lambda coeff: self.recipe.make_critic_penalty(self.opt_d, coeff=coeff)
        self.program = compile_legacy_program(
            self.graph, self.prior, config, self.objectives, self.gan, self.penalty, self.spread, make_penalty)
        groups = [{"params": list(self.program.generator_parameters), "lr": opt["lr"]}]
        self.lr_roles = [["generator"], ["critic"] * len(self.opt_d.param_groups)]
        if self.program.prior_parameters:
            groups.append({"params": list(self.program.prior_parameters), "lr": opt["lr"] * opt["prior_lr_mult"], "betas": tuple(opt["prior_betas"])})
            self.lr_roles[0].append("prior")
        # Latent damping acts on a plain particle table alone in its group.
        table = getattr(self.prior, "z", None)
        latent_table = table if (type(self.prior) is ParticlePrior and table.requires_grad
                                 and self.program.prior_parameters == (table,)) else None
        self.opt_g = _device_optimizer(self.recipe.make_generator_optimizer(groups, latent_table=latent_table, **optimizer_options))
        self.base_lrs = [[g["lr"] for g in optimizer.param_groups] for optimizer in (self.opt_g, self.opt_d)]
        self.ema_graph = copy.deepcopy(self.graph).eval().requires_grad_(False)
        self.ema_prior = copy.deepcopy(self.prior).eval().requires_grad_(False)
        data_device = self.device if settings['data_rng_device'] == 'execution' else 'cpu'
        self.streams = {"data": torch.Generator(device=data_device).manual_seed(settings["seed"] + settings['data_seed_offset']), "prior": torch.Generator(device=self.device).manual_seed(settings["seed"] + settings['prior_seed_offset']), "penalty": torch.Generator(device=self.device).manual_seed(settings["seed"] + 3),
                        "noise": torch.Generator(device=self.device).manual_seed(settings["seed"] + NOISE_SEED_OFFSET)}
        self.step = 0
        self._metric_transfer = _MetricScalarTransfer(self.device)
        self._unscale_scalars = {}

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
        Sample bindings, gradient policy, and parameter membership come from the compiled program.
        """
        return run_native_program(self, batch, latent_draw, generator_batch, generator_latent_draw)

    def _unscale(self, device):
        """A reusable ``inv_scale`` of exactly one per gradient device."""
        if device not in self._unscale_scalars:
            self._unscale_scalars[device] = torch.ones((), dtype=torch.float32, device=device)
        return self._unscale_scalars[device]

    def _gradient_flags(self, parameters):
        """Device-side nonfinite flags for one parameter group; no host read.

        ``torch._amp_foreach_non_finite_check_and_unscale_`` is the AMP
        GradScaler kernel: it records any nonfinite element in ``found_inf`` and
        multiplies each gradient by ``inv_scale``. With ``inv_scale`` exactly one
        that multiplication is exact for normals, subnormals and signed zero and
        preserves inf/nan, so gradients stay bitwise unchanged while a whole
        group is screened in one launch per device and dtype instead of one
        blocking host read per tensor. Gradients the kernel does not accept
        (integer, complex or non-strided) keep the previous elementwise
        ``isfinite`` reduction, which is also computed without a host read.
        """
        fused, flags = {}, []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_floating_point() and gradient.layout == torch.strided:
                fused.setdefault((gradient.device, gradient.dtype), []).append(gradient)
            else:
                flags.append(torch.isfinite(gradient).all().logical_not().to(device=self.device))
        for (device, _), gradients in fused.items():
            found = torch.zeros((), dtype=torch.float32, device=device)
            torch._amp_foreach_non_finite_check_and_unscale_(gradients, found, self._unscale(device))
            flags.append(found.to(device=self.device, dtype=torch.bool))
        return flags

    def _refuse_nonfinite(self, loss, loss_message, groups):
        """Screen one phase's loss and gradients with exactly one host read.

        ``groups`` pairs a message with the parameters it describes. Every flag
        is computed on the device, stacked into one small tensor and read once,
        so the per-tensor synchronizations are gone while the optimizer step is
        still reached only when nothing was flagged: a refused step leaves
        parameters and optimizer state untouched. The loss flag is stacked first
        so a nonfinite loss keeps reporting the loss message.
        """
        messages = [loss_message]
        flags = [torch.isfinite(loss).all().logical_not().to(device=self.device)]
        for message, parameters in groups:
            for flag in self._gradient_flags(parameters):
                messages.append(message)
                flags.append(flag)
        for message, nonfinite in zip(messages, torch.stack(flags).tolist()):
            if nonfinite:
                raise ValueError(message)


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
               GANLoss, GradientPenalty, ParticleRegularizer, Recipe, learning_rate_scales, input_noise_std,
               type(trainer.penalty),
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
          stop_after_steps=None, on_event=None, preview_every=0, preview_keep=None,
          preview_keep_source=None, preview_name=None):
    """Create a run; budgets stop only at complete D/G/EMA update boundaries."""
    from .run_controller import run_train
    from .single_execution import SingleProcessExecution
    from .previews import sample_name
    return run_train(config_path, run_dir, steps, checkpoint_every=checkpoint_every,
                     max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
                     preview_every=preview_every, preview_keep=preview_keep,
                     preview_keep_source=preview_keep_source,
                     preview_name=sample_name(preview_name),
                     execution_factory=SingleProcessExecution)


def resume(run_dir, checkpoint=None, config_path=None, *, checkpoint_every=None,
           max_seconds=None, stop_after_steps=None, on_event=None, preview_every=None, preview_keep=None,
           preview_keep_source=None, preview_name=None, steps=None, require_same_config=False):
    """Resume full state, allowing a longer target for constant-rate training."""
    from .run_controller import run_resume
    from .single_execution import SingleProcessExecution
    return run_resume(run_dir, checkpoint, config_path, checkpoint_every=checkpoint_every,
                      max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
                      preview_every=preview_every, preview_keep=preview_keep,
                      preview_keep_source=preview_keep_source, preview_name=preview_name,
                      steps=steps, require_same_config=require_same_config,
                      execution_factory=SingleProcessExecution)
