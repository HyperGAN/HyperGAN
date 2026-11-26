"""Small torch component registry and explicit forward bindings."""
import contextlib
import importlib
import random
import sys
import torch
from torch import nn
from .data import ImageFolder


class MLP(nn.Module):
    """Legacy argument adapter; every layer is compiled by HNDL."""
    def __init__(self, input_dim=None, output_dim=None, hidden=(64, 64), negative_slope=0.2,
                 source=None, input_shape=None, output_shape=None, **kwargs):
        super().__init__()
        from .hndl_networks import build_network
        if source is None:
            source = '\n'.join(f'linear({width})\nleaky_relu({negative_slope!r})' for width in hidden)
            source += '\nlinear()'
        self.network = build_network(source, input_shape=input_shape or ('B', input_dim),
                                     output_shape=output_shape or ('B', output_dim), **kwargs)

    def forward(self, x, **inputs):
        return self.network(x=x, **inputs) if inputs else self.network(x)


def linear(in_features, out_features, bias=True):
    from .hndl_networks import HNDLNetwork
    return HNDLNetwork(f'linear(bias={bias!r})', ('B', in_features), ('B', out_features))


class Identity(nn.Module):
    """Shape-polymorphic binding adapter for legacy identity components."""
    def forward(self, input):
        # Identity carries no architecture or parameters.
        return input


class GaussianGrid:
    def __init__(self, side=10, noise=0.015):
        if type(side) is not int or side < 1 or noise < 0:
            raise ValueError("gaussian_grid requires positive side and nonnegative noise")
        self.centers = torch.cartesian_prod(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side))
        self.noise = noise

    def __call__(self, batch_size, *, generator):
        ids = torch.randint(len(self.centers), (batch_size,), generator=generator)
        return {"real": self.centers[ids] + self.noise * torch.randn(batch_size, 2, generator=generator)}


class PairedLinear:
    """Synthetic paired-input fixture; deliberately makes no application claim."""
    def __init__(self, dimensions=2, scale=2.0, offset=0.0):
        self.dimensions, self.scale, self.offset = dimensions, scale, offset

    def __call__(self, batch_size, *, generator):
        condition = torch.randn(batch_size, self.dimensions, generator=generator)
        return {"condition": condition, "real": condition * self.scale + self.offset}


BUILTINS = {"mlp": MLP, "linear": linear, "identity": Identity, "mse": nn.MSELoss, "l1": nn.L1Loss, "gaussian_grid": GaussianGrid, "paired_linear": PairedLinear, "image_folder": ImageFolder}


def execution_device(value, fallback=True):
    """Resolve an explicit native device; falls back to visible device on unavailable hardware."""
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Native execution requires cpu or cuda[:index]')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            if fallback:
                from .checkpoint_compatibility import HyperGANHardwareWarning
                import warnings
                warnings.warn(
                    f"CUDA device requested ({value!r}) but CUDA is unavailable on this host; falling back to cpu",
                    HyperGANHardwareWarning,
                    stacklevel=2,
                )
                return torch.device('cpu')
            raise ValueError('CUDA was requested but is unavailable; install a CUDA-enabled PyTorch build and check the NVIDIA driver, or explicitly select cpu')
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            if fallback and torch.cuda.device_count() > 0:
                from .checkpoint_compatibility import HyperGANHardwareWarning
                import warnings
                warnings.warn(
                    f"CUDA device index {index} is unavailable; falling back to cuda:0",
                    HyperGANHardwareWarning,
                    stacklevel=2,
                )
                return torch.device('cuda', 0)
            elif fallback:
                from .checkpoint_compatibility import HyperGANHardwareWarning
                import warnings
                warnings.warn(
                    f"CUDA device index {index} is unavailable; falling back to cpu",
                    HyperGANHardwareWarning,
                    stacklevel=2,
                )
                return torch.device('cpu')
            raise ValueError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
    return device



def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        target = device if isinstance(device, torch.device) else torch.device(device)
        if target.type == 'cuda' and value.device.type == 'cpu' and not value.requires_grad and value.numel() >= 4096:
            return value.pin_memory().to(target, non_blocking=True)
        return value.to(target)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_tensors(item, device) for item in value)
    return value


def make_prior(spec, *, device=None, recipe=None):
    from particlegan import GaussianPrior, MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
    args = dict(spec['args'])
    initial_device = 'cpu' if spec.get('initialization_device') == 'cpu' else device
    if initial_device is not None:
        args['device'] = initial_device
    if spec.get('initialization_seed') is not None:
        args['generator'] = torch.Generator(device=initial_device or 'cpu').manual_seed(spec['initialization_seed'])
    if spec['kind'] == 'mog':
        sigma_rel = args.pop('sigma_rel', .025)
        sigma = args.pop('sigma', None)
        if spec.get('fixed_sigma') is not None:
            if sigma is not None:
                raise ValueError('Choose prior.fixed_sigma or prior.args.sigma, not both')
            sigma = spec['fixed_sigma']
        prior = MoGParticlePrior(sigma=0.0 if sigma is None else sigma, **args)
        if sigma is None:
            # Preserve historical relative-noise recipes explicitly. Fixed-noise
            # recipes never search neighbors, including during checkpoint setup.
            value, distance = calibrate_mog_sigma(prior.means(), sigma_rel)
            prior.set_sigma(value)
            prior.d0.copy_(distance)
            prior.sigma_rel = float(sigma_rel)
    else:
        prior = {"particles": ParticlePrior, "gaussian": GaussianPrior}[spec["kind"]](**args)
    prior = prior.to(device) if device is not None else prior
    if recipe:
        # Inference workers must reconstruct persistent recipe-owned buffers
        # before strict loading, including initial prior reference buffers.
        from .particlegan_integration import restore_particlegan_recipe
        restore_particlegan_recipe(recipe).apply_prior_policy(prior)
    return prior


def construct(spec):
    name = spec["factory"]
    if name == "hndl":
        from .hndl_networks import HNDLNetwork
        return HNDLNetwork(**spec["args"])
    if name in BUILTINS:
        constructor = BUILTINS[name]
    else:
        module, attribute = name.split(":")
        try:
            constructor = importlib.import_module(module)
            for part in attribute.split("."):
                constructor = getattr(constructor, part)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Unavailable custom constructor {name}: {exc}") from exc
    try:
        return constructor(**spec.get("args", {}))
    except TypeError as exc:
        raise ValueError(f"Invalid constructor arguments for {name}: {exc}") from exc


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach(v) for v in value)
    return value


def generation_output(graph, context, sampling):
    """Select inference output without changing the adversarial generated binding."""
    return graph.resolve(sampling.get('generated', 'generated'), context)


def generation_particle_ids(graph, context, prior_ids, sampling):
    """Resolve explicitly declared routed IDs without reporting unused prior draws."""
    if 'particle_ids' not in sampling:
        return prior_ids
    ids = graph.resolve(sampling['particle_ids'], context)
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1
            or len(ids) != len(generation_output(graph, context, sampling))
            or ids.dtype not in (torch.int32, torch.int64) or (ids < 0).any()):
        raise ValueError('sampling.particle_ids must produce one nonnegative integer per generated sample')
    return ids


def _call_component(model, kwargs):
    """Call single-input HNDL/MLP components positionally so dependency forward hooks see args[0]."""
    if len(kwargs) == 1 and not getattr(model, "named_inputs", False):
        from .hndl_networks import HNDLNetwork
        if isinstance(model, (HNDLNetwork, MLP)):
            return model(next(iter(kwargs.values())))
    return model(**kwargs)


def _has_cuda_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == 'cuda'
    if isinstance(value, dict):
        return any(_has_cuda_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_has_cuda_tensor(item) for item in value)
    return False


def _to_channels_last(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 4 and value.is_floating_point() and value.device.type == 'cuda':
            return value.contiguous(memory_format=torch.channels_last)
        return value
    if isinstance(value, dict):
        converted = {key: _to_channels_last(item) for key, item in value.items()}
        return value if all(converted[key] is value[key] for key in value) else converted
    if isinstance(value, tuple):
        converted = tuple(_to_channels_last(item) for item in value)
        return value if all(new is old for new, old in zip(converted, value)) else converted
    if isinstance(value, list):
        converted = [_to_channels_last(item) for item in value]
        return value if all(new is old for new, old in zip(converted, value)) else converted
    return value


_UNSET = object()
_PG_FLOAT_OUTPUT = _UNSET
_PG_CRITIC_SOURCE_CONTEXT = _UNSET


def _fallback_float_output(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype == torch.bfloat16 else value
    if isinstance(value, tuple):
        converted = tuple(_fallback_float_output(item) for item in value)
        return value if all(new is old for new, old in zip(converted, value)) else converted
    if isinstance(value, list):
        converted = [_fallback_float_output(item) for item in value]
        return value if all(new is old for new, old in zip(converted, value)) else converted
    if isinstance(value, dict):
        converted = {k: _fallback_float_output(v) for k, v in value.items()}
        return value if all(converted[k] is value[k] for k in value) else converted
    return value


def _noop_critic_source_context(_critic, _source):
    return contextlib.nullcontext()


def _get_pg_float_output():
    global _PG_FLOAT_OUTPUT
    if _PG_FLOAT_OUTPUT is _UNSET:
        try:
            import particlegan.training as pg_training
            helper = getattr(pg_training, 'float_output', None) or getattr(pg_training, '_float_output', None)
        except ImportError:
            helper = None
        _PG_FLOAT_OUTPUT = helper if callable(helper) else _fallback_float_output
    return _PG_FLOAT_OUTPUT


def _get_pg_critic_source_context():
    global _PG_CRITIC_SOURCE_CONTEXT
    if _PG_CRITIC_SOURCE_CONTEXT is _UNSET:
        try:
            import particlegan.training as pg_training
            helper = getattr(pg_training, 'critic_source_context', None)
        except ImportError:
            helper = None
        _PG_CRITIC_SOURCE_CONTEXT = helper if callable(helper) else _noop_critic_source_context
    return _PG_CRITIC_SOURCE_CONTEXT


class _ChannelsLastInputPreHook:
    """Keep 4D CUDA inputs channels_last entering NHWC-sensitive ops (e.g. bilinear Upsample)."""

    def __init__(self, owner):
        self._owner = owner

    def __call__(self, _module, args):
        if (
            not args
            or not getattr(self._owner, '_hypergan_channels_last', False)
            or not torch.is_autocast_enabled('cuda')
        ):
            return None
        x = args[0]
        if (
            isinstance(x, torch.Tensor)
            and x.ndim == 4
            and x.is_floating_point()
            and x.device.type == 'cuda'
            and not x.is_contiguous(memory_format=torch.channels_last)
        ):
            return (x.contiguous(memory_format=torch.channels_last), *args[1:])
        return None


class _ChannelsLastOutputHook:
    """Restore channels_last after ops (e.g. autocast GroupNorm) that emit NCHW 4D tensors."""

    def __init__(self, owner):
        self._owner = owner

    def __call__(self, _module, _args, output):
        if (
            not getattr(self._owner, '_hypergan_channels_last', False)
            or not torch.is_autocast_enabled('cuda')
        ):
            return output
        if (
            isinstance(output, torch.Tensor)
            and output.ndim == 4
            and output.is_floating_point()
            and output.device.type == 'cuda'
            and not output.is_contiguous(memory_format=torch.channels_last)
        ):
            return output.contiguous(memory_format=torch.channels_last)
        return output


def install_channels_last_hooks(owner):
    """Attach deepcopy-safe NHWC preservation hooks to Upsample and GroupNorm submodules."""
    pre_hook = _ChannelsLastInputPreHook(owner)
    post_hook = _ChannelsLastOutputHook(owner)
    for sub in owner.modules():
        if isinstance(sub, nn.Upsample):
            sub.register_forward_pre_hook(pre_hook)
        elif isinstance(sub, nn.GroupNorm):
            sub.register_forward_hook(post_hook)


def clear_channels_last_hooks(owner):
    """Remove NHWC preservation hooks from ``owner`` and its submodules (e.g. on ``ema_graph``)."""
    for sub in owner.modules():
        for key in [k for k, fn in sub._forward_pre_hooks.items() if isinstance(fn, _ChannelsLastInputPreHook)]:
            sub._forward_pre_hooks.pop(key, None)
        for key in [k for k, fn in sub._forward_hooks.items() if isinstance(fn, _ChannelsLastOutputHook)]:
            sub._forward_hooks.pop(key, None)


def _call_component_with_precision(model, kwargs, *, bf16=False, channels_last=False):
    """Invoke a component with optional CUDA BF16 autocast, NHWC 4D inputs, and FP32 output cast."""
    if bf16 and _has_cuda_tensor(kwargs):
        float_output = _get_pg_float_output()
        call_kwargs = _to_channels_last(kwargs) if channels_last else kwargs
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return float_output(_call_component(model, call_kwargs))
    if not bf16 and torch.is_autocast_enabled('cuda'):
        with torch.autocast('cuda', enabled=False):
            return _call_component(model, kwargs)
    return _call_component(model, kwargs)


class _GeneratorProbeModule(nn.Module):
    """Replay the complete generator computation from original latent and conditioning to configured output.

    Held outside ``ComponentGraph._modules`` so ``graph.state_dict()`` and
    ``graph.named_modules()`` remain unchanged for checkpoints and resume.
    """
    def __init__(self, graph, output_path='generated', prior=None):
        super().__init__()
        models, trainable_names, batch_keys, prior_keys = graph.output_subgraph(output_path)
        object.__setattr__(self, '_graph', graph)
        object.__setattr__(self, '_output_path', output_path)
        object.__setattr__(self, '_prior', prior)
        object.__setattr__(self, '_trainable_names', tuple(trainable_names))
        object.__setattr__(self, '_batch_keys', batch_keys)
        object.__setattr__(self, '_prior_keys', tuple(prior_keys))
        self._components = nn.ModuleDict(models)

    def trainable_parameters(self, *, allowed_param_ids=None):
        seen = set()
        params = []
        for name in self._trainable_names:
            module = self._components[name]
            for parameter in module.parameters():
                pid = id(parameter)
                if not parameter.requires_grad or pid in seen:
                    continue
                if allowed_param_ids is not None and pid not in allowed_param_ids:
                    continue
                seen.add(pid)
                params.append(parameter)
        return tuple(params)

    def has_training_component(self):
        names = self._trainable_names or tuple(self._components.keys())
        return any(self._components[name].training for name in names)

    def capture_inputs(self, latent, batch, context):
        kwargs = {}
        if self._batch_keys is None:
            kwargs['batch'] = batch
        elif self._batch_keys:
            kwargs['batch'] = {key: batch[key] for key in self._batch_keys if key in batch}
        if self._prior_keys:
            prior_ctx = context.get('prior') or {}
            captured_prior = {}
            for key in self._prior_keys:
                if key in prior_ctx:
                    captured_prior[key] = detach(prior_ctx[key])
                elif self._prior is not None:
                    captured_prior[key] = detach(self._prior.means() if key == 'means' else self._prior.sigma)
            if captured_prior:
                kwargs['prior'] = captured_prior
        return (latent,), kwargs

    def forward(self, latent, *, batch=None, prior=None):
        cpu_rng = torch.get_rng_state()
        py_rng = random.getstate()
        np_mod = sys.modules.get('numpy')
        np_rng = np_mod.random.get_state() if np_mod is not None and hasattr(np_mod, 'random') else None
        cuda_rng = None
        cuda_dev = None
        if torch.cuda.is_initialized():
            cuda_dev = torch.cuda.current_device()
            cuda_rng = torch.cuda.get_rng_state(cuda_dev)
        try:
            context = {
                'latent': latent,
                'batch': {} if batch is None else batch,
                'components': {},
                'prior': {} if prior is None else dict(prior),
                '_prior': self._prior,
            }
            if self._output_path == 'generated' or self._output_path.startswith('generated.'):
                context['generated'] = self._graph.call('generator', context)
                return self._graph.resolve(self._output_path, context)
            return self._graph.resolve(self._output_path, context)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None and cuda_dev is not None:
                torch.cuda.set_rng_state(cuda_rng, cuda_dev)
            random.setstate(py_rng)
            if np_rng is not None and np_mod is not None:
                np_mod.random.set_state(np_rng)


def _eval_probe_with_preserved_state(module, args, kwargs, record):
    """Fallback RNG/mode/BN-safe probe evaluator when ParticleGAN lacks ``record_unified_v11_generator_probe``."""
    modes = [(m, m.training) for m in module.modules()]
    prev_probe = bool(getattr(record, '_in_probe', False)) if record is not None else False
    cpu_rng = torch.get_rng_state()
    py_rng = random.getstate()
    np_mod = sys.modules.get('numpy')
    np_rng = np_mod.random.get_state() if np_mod is not None and hasattr(np_mod, 'random') else None
    cuda_rng = None
    cuda_dev = None
    if torch.cuda.is_initialized():
        cuda_dev = torch.cuda.current_device()
        cuda_rng = torch.cuda.get_rng_state(cuda_dev)
    if record is not None:
        record._in_probe = True
    try:
        with torch.no_grad():
            module.eval()
            first_p = next(module.parameters(), None)
            use_bf16 = (
                first_p is not None
                and first_p.device.type == 'cuda'
                and bool(
                    getattr(module, '_particlegan_bf16_autocast', False)
                    or getattr(module, '_hypergan_bf16_autocast', False)
                    or getattr(module, '_axiom_bf16_autocast', False)
                    or getattr(module, '_unified_v4_bf16_autocast', False)
                )
            )
            if use_bf16:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = module(*args, **kwargs)
            else:
                out = module(*args, **kwargs)
            target = out[0] if isinstance(out, (tuple, list)) and out else out
            if not isinstance(target, torch.Tensor):
                return None
            return target.detach().to(dtype=torch.float32)
    finally:
        if record is not None:
            record._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and cuda_dev is not None:
            torch.cuda.set_rng_state(cuda_rng, cuda_dev)
        random.setstate(py_rng)
        if np_rng is not None and np_mod is not None:
            np_mod.random.set_state(np_rng)


class ComponentGraph(nn.Module):
    """Auxiliary components execute on demand; all trainable auxiliaries belong to G.

    A discriminator sees detached conditioning. Only its candidate is differentiated
    during a generator update or candidate-only gradient penalty.
    """
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        modules = {}
        order = [name for name in ('generator', 'discriminator') if name in specs]
        order += sorted(set(specs) - set(order))
        for name in order:
            spec = specs[name]
            if 'reuse' in spec:
                continue
            module = construct(spec)
            if not isinstance(module, nn.Module):
                raise ValueError(f"Component {name} must construct a torch.nn.Module")
            if not spec["trainable"]:
                module.requires_grad_(False)
                module.eval()
            modules[name] = module
        self.models = nn.ModuleDict(modules)
        self._probe_hook = None
        self._bf16_components = frozenset()
        self._channels_last_components = frozenset()
        self._training_bf16_active = False

    def component_name_for(self, module):
        """Return the registered component name owning ``module``, else None."""
        for name, candidate in self.models.items():
            if candidate is module:
                return name
        return None

    def uses_bf16(self, module_or_name):
        """Return True when the named component or module is configured for BF16 autocast."""
        if isinstance(module_or_name, str):
            return module_or_name in self._bf16_components
        name = self.component_name_for(module_or_name)
        if name is not None:
            return name in self._bf16_components
        return bool(getattr(module_or_name, '_hypergan_bf16_autocast', False))

    def uses_channels_last(self, module_or_name):
        """Return True when the named component or module is configured for CUDA NHWC layout."""
        if isinstance(module_or_name, str):
            return module_or_name in self._channels_last_components
        name = self.component_name_for(module_or_name)
        if name is not None:
            return name in self._channels_last_components
        return bool(getattr(module_or_name, '_hypergan_channels_last', False))

    def call_critic(self, active_module, kwargs, *, source_candidate=None, bf16=None, channels_last=None):
        """Execute a critic forward with training-step BF16/NHWC and live-endpoint source binding."""
        use_bf16 = (
            bool(bf16)
            if bf16 is not None
            else bool(self._training_bf16_active and self.uses_bf16(active_module))
        )
        use_nhwc = (
            bool(channels_last)
            if channels_last is not None
            else bool(use_bf16 and self.uses_channels_last(active_module))
        )
        if source_candidate is not None:
            critic_source_context = _get_pg_critic_source_context()
            with critic_source_context(active_module, source_candidate):
                return _call_component_with_precision(
                    active_module, kwargs, bf16=use_bf16, channels_last=use_nhwc
                )
        return _call_component_with_precision(
            active_module, kwargs, bf16=use_bf16, channels_last=use_nhwc
        )

    def output_subgraph(self, output_path='generated'):
        """Trace unique component modules and external conditioning inputs feeding ``output_path``."""
        if output_path == 'generated' or output_path.startswith('generated.'):
            root = 'generator'
        elif output_path.startswith('components.'):
            parts = output_path.split('.')
            if len(parts) < 2 or parts[1] not in self.specs:
                raise ValueError(f"Unavailable output component binding '{output_path}'")
            root = parts[1]
        else:
            raise ValueError(f"Unsupported generated output binding '{output_path}'")

        models = {}
        trainable_names = []
        trainable_seen = set()
        batch_all = False
        batch_keys = set()
        prior_keys = set()
        visited = set()

        def walk(name, frozen_edge):
            nonlocal batch_all
            key = (name, frozen_edge)
            if key in visited:
                return
            visited.add(key)
            spec = self.specs[name]
            edge_frozen = frozen_edge or bool(spec.get('freeze_parameters', False)) or not spec.get('trainable', True)
            for path in spec.get('inputs', {}).values():
                if path == 'batch':
                    batch_all = True
                elif path.startswith('batch.'):
                    batch_keys.add(path.split('.', 2)[1])
                elif path.startswith('prior.'):
                    prior_keys.add(path.split('.', 2)[1])
                elif path.startswith('components.'):
                    walk(path.split('.', 2)[1], frozen_edge)
            model_name = spec.get('reuse', name)
            if model_name not in models:
                models[model_name] = self.models[model_name]
            if not edge_frozen and model_name not in trainable_seen:
                trainable_seen.add(model_name)
                trainable_names.append(model_name)

        walk(root, False)
        return models, trainable_names, (None if batch_all else tuple(sorted(batch_keys))), tuple(sorted(prior_keys))

    def build_generator_probe(self, output_path='generated', *, prior=None):
        """Construct a non-submodule replay adapter for the configured generated output."""
        return _GeneratorProbeModule(self, output_path=output_path, prior=prior)

    def resolve(self, path, context, active=None):
        parts = path.split(".")
        if parts[0] == 'prior' and parts[1] not in context['prior']:
            prior = context.get('_prior')
            if prior is None:
                raise ValueError('This component binding requires the current prior')
            context['prior'][parts[1]] = prior.means() if parts[1] == 'means' else prior.sigma
        if parts[0] == "components" and parts[1] not in context["components"]:
            self.call(parts[1], context, active)
        value = context
        try:
            for part in parts:
                value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Unavailable I/O binding '{path}'") from exc
        return value

    def call(self, name, context, active=None):
        active = set() if active is None else active
        if name in active:
            raise ValueError(f"Cyclic component binding at {name}")
        active.add(name)
        try:
            kwargs = {arg: self.resolve(path, context, active) for arg, path in self.specs[name]["inputs"].items()}
            spec = self.specs[name]
            model_name = spec.get('reuse', name)
            model = self.models[model_name]
            flags = [parameter.requires_grad for parameter in model.parameters()]
            if spec.get('freeze_parameters', False):
                model.requires_grad_(False)
            try:
                if self._training_bf16_active:
                    use_bf16 = model_name in self._bf16_components
                    use_nhwc = use_bf16 and model_name in self._channels_last_components
                    value = _call_component_with_precision(
                        model, kwargs, bf16=use_bf16, channels_last=use_nhwc
                    )
                else:
                    value = _call_component(model, kwargs)
            finally:
                if spec.get('freeze_parameters', False):
                    for parameter, flag in zip(model.parameters(), flags):
                        parameter.requires_grad_(flag)
            context["components"][name] = value
            return value
        finally:
            active.remove(name)

    def generate(self, latent, batch, *, prior=None):
        context = {"latent": latent, "batch": batch, "components": {}, 'prior': {}, '_prior': prior}
        context["generated"] = self.call("generator", context)
        if self._probe_hook is not None:
            self._probe_hook(latent, batch, context, prior)
        return context

    def critic(self, candidate, context, module=None, *, bf16=None):
        # Auxiliary conditioning runs with detached inputs and contributes no D loss
        # gradients to G/encoder parameters. The candidate path remains attached.
        # ``module`` replaces the discriminator (the penalty passes its EMA).
        fixed = detach(context)
        fixed["candidate"] = candidate
        kwargs = {}
        for arg, path in self.specs["discriminator"]["inputs"].items():
            value = self.resolve(path, fixed)
            kwargs[arg] = value if path == "candidate" else detach(value)
        active_module = self.models["discriminator"] if module is None else module
        return self.call_critic(active_module, kwargs, source_candidate=candidate, bf16=bf16)

    def generator_parameters(self):
        return [p for name, model in self.models.items() if name != "discriminator" for p in model.parameters() if p.requires_grad]
