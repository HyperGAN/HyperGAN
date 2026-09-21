"""Small torch component registry and explicit forward bindings."""
import importlib
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

    def forward(self, x, condition=None):
        return self.network(torch.cat((x, condition), dim=-1) if condition is not None else x)


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


def execution_device(value):
    """Resolve an explicit native device; requesting CUDA never falls back to CPU."""
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Native execution requires cpu or cuda[:index]')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable; install a CUDA-enabled PyTorch build and check the NVIDIA driver, or explicitly select cpu')
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
    return device


def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_tensors(item, device) for item in value)
    return value


def make_prior(spec, *, device=None):
    from particlegan import GaussianPrior, MoGParticlePrior, ParticlePrior
    args = dict(spec['args'])
    initial_device = 'cpu' if spec.get('initialization_device') == 'cpu' else device
    if initial_device is not None:
        args['device'] = initial_device
    if spec.get('initialization_seed') is not None:
        args['generator'] = torch.Generator(device=initial_device or 'cpu').manual_seed(spec['initialization_seed'])
    prior = {"particles": ParticlePrior, "mog": MoGParticlePrior, "gaussian": GaussianPrior}[spec["kind"]](**args)
    if spec.get('fixed_sigma') is not None:
        with torch.no_grad():
            prior.sigma.fill_(spec['fixed_sigma'])
        prior._noise_enabled = bool(spec['fixed_sigma'] > 0)
    return prior.to(device) if device is not None else prior


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
            model = self.models[spec.get('reuse', name)]
            flags = [parameter.requires_grad for parameter in model.parameters()]
            if spec.get('freeze_parameters', False):
                model.requires_grad_(False)
            try:
                value = model(**kwargs)
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
        return context

    def critic(self, candidate, context):
        # Auxiliary conditioning runs with detached inputs and contributes no D loss
        # gradients to G/encoder parameters. The candidate path remains attached.
        fixed = detach(context)
        fixed["candidate"] = candidate
        kwargs = {}
        for arg, path in self.specs["discriminator"]["inputs"].items():
            value = self.resolve(path, fixed)
            kwargs[arg] = value if path == "candidate" else detach(value)
        return self.models["discriminator"](**kwargs)

    def generator_parameters(self):
        return [p for name, model in self.models.items() if name != "discriminator" for p in model.parameters() if p.requires_grad]
