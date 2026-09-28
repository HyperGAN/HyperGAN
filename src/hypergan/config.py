"""Torch-free, explicit configuration for the first reference runtime.

Constructor paths are trusted Python code, imported only by train/sample.
"""
from copy import deepcopy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

from .network_config import packaged_source, read_source, validate_network_args, materialize_networks

from .metrics import DEFAULT_METRICS, evaluation_warnings, objective_id, validate_metrics

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


DEFAULT = {
    "schema_version": 1,
    "metrics": deepcopy(DEFAULT_METRICS),
    "name": "reference/100gaussians",
    "data": {"factory": "gaussian_grid", "args": {"side": 10, "noise": 0.015}},
    "components": {
        "generator": {"factory": "hndl", "args": {"source": packaged_source("reference.hndl"), "input_shape": ["B", 4], "output_shape": ["B", 2]}, "inputs": {"x": "latent"}, "trainable": True},
        "discriminator": {"factory": "hndl", "args": {"source": packaged_source("reference.hndl"), "input_shape": ["B", 2], "output_shape": ["B", 1]}, "inputs": {"x": "candidate"}, "trainable": True},
    },
    "prior": {"kind": "particles", "args": {"num_particles": 20000, "z_dim": 4}, "initialization_device": "execution", "initialization_seed": None, "fixed_sigma": None},
    "adversarial": {"loss_type": "logistic", "mode": "rp", "weight": 1.0},
    "gradient_penalty": {"arm": "b_cap", "coeff": 1.0, "kappa": 1.0, "lazy_k": 1, "norm": "l2", "target_anneal": "none", "total_steps": 0, "method": "autograd", "fd_eps": 0.05},
    "prior_regularizer": {"weight": 1.0, "target_std": 1.0, "eps": 0.0001, "rows": "sampled_unique"},
    "objectives": [],
    "optimizer": {"lr": 0.0006, "d_lr_mult": 1.5, "prior_lr_mult": 10.0, "betas": [0.0, 0.999], "prior_betas": [0.0, 0.999], "implementation": "device_adam"},
    "training": {"steps": 5, "batch_size": 16, "seed": 42, "device": "cpu", "ema": 0.995, "lr_anneal_start": 0.6, "lr_floor": 0.05,
                 "phase_draws": "shared", "data_rng_device": "cpu", "data_seed_offset": 1, "prior_seed_offset": 2, "backend": {}},
    "sampling": {"count": 256, "seed": 123},
}

DEFAULT_TOML = '''# A five-step numerical reference, not a converged image model.
schema_version = 1
name = "reference/100gaussians"

[data]
factory = "gaussian_grid"
[data.args]
side = 10
noise = 0.015

[components.generator]
factory = "hndl"
inputs = { x = "latent" }
[components.generator.args]
input_shape = ["B", 4]
output_shape = ["B", 2]
source = """
# Two-dimensional Gaussian-grid numerical reference.
linear(64)
leaky_relu(0.2)
linear(64)
leaky_relu(0.2)
linear()
"""

[components.discriminator]
factory = "hndl"
inputs = { x = "candidate" }
[components.discriminator.args]
input_shape = ["B", 2]
output_shape = ["B", 1]
source = """
# Two-dimensional Gaussian-grid numerical reference.
linear(64)
leaky_relu(0.2)
linear(64)
leaky_relu(0.2)
linear()
"""

[prior]
kind = "particles"
[prior.args]
num_particles = 20000
z_dim = 4

[adversarial]
loss_type = "logistic"
mode = "rp"
weight = 1.0

[gradient_penalty]
arm = "b_cap"
coeff = 1.0
kappa = 1.0
lazy_k = 1

[prior_regularizer]
rows = "sampled_unique"
weight = 1.0

[optimizer]
lr = 0.0006
d_lr_mult = 1.5
prior_lr_mult = 10.0
betas = [0.0, 0.999]
prior_betas = [0.0, 0.999]

[training]
steps = 5
batch_size = 16
seed = 42
device = "cpu"
ema = 0.995
lr_anneal_start = 0.6
lr_floor = 0.05

[sampling]
count = 256
seed = 123
'''


# `defaults = "particlegan"` fills these omitted fields from the installed
# particlegan.Recipe field defaults instead of DEFAULT. Resolved configurations
# record the concrete values, so a later ParticleGAN release cannot silently
# change an existing run; its resume then fails the configuration check.
PARTICLEGAN_DEFAULT_FIELDS = {
    ("adversarial", "loss_type"): "loss_type",
    ("adversarial", "mode"): "gan_mode",
    ("gradient_penalty", "arm"): "reg_arm",
    ("gradient_penalty", "coeff"): "reg_coeff",
    ("gradient_penalty", "kappa"): "reg_kappa",
    ("gradient_penalty", "lazy_k"): "reg_every",
    ("gradient_penalty", "method"): "reg_method",
    ("prior_regularizer", "weight"): "prior_reg",
    ("optimizer", "lr"): "lr",
    ("optimizer", "d_lr_mult"): "d_lr_mult",
    ("optimizer", "prior_lr_mult"): "prior_lr_mult",
    ("optimizer", "betas"): "betas",
    ("optimizer", "prior_betas"): "prior_betas",
    ("training", "ema"): "ema_decay",
    ("training", "lr_anneal_start"): "lr_anneal_start",
    ("training", "lr_floor"): "lr_floor",
}


@functools.cache
def particlegan_defaults():
    """Installed particlegan.Recipe defaults, keyed by HyperGAN (section, field).

    Reads the dataclass from recipes.py without importing the package, whose
    __init__ imports Torch; configuration loading stays Torch-free.
    """
    package = importlib.util.find_spec("particlegan")
    if package is None or not package.submodule_search_locations:
        raise ValueError('defaults = "particlegan" requires ParticleGAN (the train extra)')
    name = "_hypergan_particlegan_recipes"
    spec = importlib.util.spec_from_file_location(name, Path(package.submodule_search_locations[0]) / "recipes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        recipe = {field.name: field.default for field in dataclasses.fields(module.Recipe)}
    finally:
        del sys.modules[name]
    if recipe["prior_betas"] is None:
        recipe["prior_betas"] = recipe["betas"]
    return {key: list(recipe[field]) if isinstance(recipe[field], tuple) else recipe[field]
            for key, field in PARTICLEGAN_DEFAULT_FIELDS.items()}


def _keys(value, allowed, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a table")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {location} fields: {', '.join(sorted(unknown))}")


def _positive(value, location, integer=False, zero=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{location} must be a {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}")


def _factory(value, builtins, location):
    if not isinstance(value, str) or (value not in builtins and (value.count(":") != 1 or not all(value.split(":")))):
        raise ValueError(f"{location}: use a builtin ID or module:object constructor")


def _spec(value, location, objectives=False):
    allowed = {"id", "factory", "args", "inputs", "weight", "detach"} if objectives else {"factory", "args", "inputs", "trainable", "reuse", "freeze_parameters"}
    _keys(value, allowed, location)
    if not objectives and 'reuse' in value:
        value.setdefault('factory', 'reuse')
        if value['factory'] != 'reuse' or value.get('args') or not isinstance(value['reuse'], str):
            raise ValueError(f'{location}: reuse requires a component name and no factory/args')
    else:
        _factory(value.get("factory"), {"mse", "l1"} if objectives else {"hndl", "mlp", "linear", "identity"}, location)
    value.setdefault("args", {})
    if not isinstance(value["args"], dict):
        raise ValueError(f"{location}.args must be a table")
    if not objectives and value.get("factory") == "hndl":
        validate_network_args(value["args"], f"{location}.args")
        if "file" in value["args"]:
            value["args"]["source"] = read_source(value["args"].pop("file"))
    if not isinstance(value.get("inputs"), dict) or not value["inputs"] or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in value["inputs"].items()):
        raise ValueError(f"{location}.inputs must bind argument names to context paths")
    if objectives:
        value.setdefault("weight", 1.0)
        value.setdefault("detach", ["target"] if "target" in value["inputs"] else [])
        _positive(value["weight"], f"{location}.weight", zero=True)
        if not isinstance(value["detach"], list) or any(k not in value["inputs"] for k in value["detach"]):
            raise ValueError(f"{location}.detach must list bound argument names")
    else:
        value.setdefault("trainable", True)
        if type(value["trainable"]) is not bool:
            raise ValueError(f"{location}.trainable must be boolean")
        if 'freeze_parameters' in value and (type(value['freeze_parameters']) is not bool or 'reuse' not in value):
            raise ValueError(f'{location}.freeze_parameters requires a boolean on a reused component')
        if 'reuse' in value and not value['trainable']:
            raise ValueError(f'{location}: reuse inherits trainability; use freeze_parameters for a frozen forward')


_TERM_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
_ADVERSARIAL_TERM_FIELDS = {"id", "component", "weight", "loss_type", "mode", "penalty", "penalty_coeff", "real", "fake", "inputs"}


def _resolve_adversarial_terms(result):
    """Normalize optional extra terms. Absent means the key is not stored.

    ``[adversarial]`` stays the implicit legacy term. These tables are further
    terms only, so they are not part of ``DEFAULT`` and do not change a legacy fingerprint.
    """
    if "adversarial_terms" not in result:
        return
    terms = result["adversarial_terms"]
    if not isinstance(terms, list) or not terms or any(not isinstance(term, dict) for term in terms):
        raise ValueError("adversarial_terms must be a non-empty list of tables")
    components = result["components"]
    seen = set()
    for index, term in enumerate(terms):
        location = f"adversarial_terms[{index}]"
        _keys(term, _ADVERSARIAL_TERM_FIELDS, location)
        ident = term.get("id")
        if not isinstance(ident, str) or _TERM_ID.fullmatch(ident) is None:
            raise ValueError(f"{location}.id must be 1–128 letters, digits, dots, underscores or hyphens")
        if ident in seen:
            raise ValueError(f"Adversarial term ids must be unique; {ident} is repeated")
        seen.add(ident)
        component = term.get("component")
        if not isinstance(component, str) or component not in components or "reuse" in components[component]:
            raise ValueError(f"{location}.component must name an existing non-reuse component")
        term.setdefault("weight", 1.0)
        _positive(term["weight"], f"{location}.weight", zero=True)
        for key in ("loss_type", "mode"):
            term.setdefault(key, result["adversarial"][key])
        if term["mode"] not in {"vanilla", "rp", "ra"} or term["loss_type"] not in {"hinge", "logistic", "wasserstein", "lsgan"}:
            raise ValueError(f"Unsupported {location} loss_type or mode")
        term.setdefault("penalty", False)
        if type(term["penalty"]) is not bool:
            raise ValueError(f"{location}.penalty must be boolean")
        term.setdefault("penalty_coeff", result["gradient_penalty"]["coeff"])
        _positive(term["penalty_coeff"], f"{location}.penalty_coeff", zero=True)
        for key in ("real", "fake"):
            if not isinstance(term.get(key), str) or not term[key]:
                raise ValueError(f"{location}.{key} must be a binding path")
        if "inputs" not in term:
            term["inputs"] = deepcopy(components[component]["inputs"])
        inputs = term["inputs"]
        if not isinstance(inputs, dict) or not inputs or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in inputs.items()):
            raise ValueError(f"{location}.inputs must bind argument names to context paths")
        if sum(path == "candidate" for path in inputs.values()) != 1:
            raise ValueError(f"{location} must bind exactly one input to 'candidate'")
    result["adversarial_terms"] = terms


def resolve_config(raw):
    """Resolve omitted defaults without importing or executing custom constructors."""
    _keys(raw, set(DEFAULT) | {"adversarial_terms", "defaults"}, "configuration")
    raw = dict(raw)
    source = raw.pop("defaults", "hypergan")
    result = deepcopy(DEFAULT)
    if source == "particlegan":
        for (section, key), value in particlegan_defaults().items():
            result[section][key] = deepcopy(value)
    elif source != "hypergan":
        raise ValueError('defaults must be "hypergan" or "particlegan"')
    for key, value in raw.items():
        if key == "components":
            # Explicit components replace the graph; no hidden old bindings survive.
            result[key] = deepcopy(value)
        elif key == "adversarial_terms":
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict):
            allowed = set(result[key]) | ({'particle_ids', 'generated', 'views', 'comparison'} if key == 'sampling' else set())
            _keys(value, allowed, key)
            result[key].update(deepcopy(value))
        else:
            result[key] = deepcopy(value)
    if result["schema_version"] != 1 or type(result["schema_version"]) is not int:
        raise ValueError("Only schema_version=1 is supported")
    if not isinstance(result["name"], str) or not result["name"]:
        raise ValueError("name must be a nonempty string")
    components = result["components"]
    if not isinstance(components, dict) or not {"generator", "discriminator"} <= set(components):
        raise ValueError("components must include generator and discriminator")
    for name, spec in components.items():
        if not name.isidentifier():
            raise ValueError("Component names must be Python identifiers")
        _spec(spec, f"components.{name}")
        materialize_networks(spec)
    for name, spec in components.items():
        if 'reuse' in spec and (name in ('generator', 'discriminator') or spec['reuse'] not in components
                               or spec['reuse'] == 'discriminator' or 'reuse' in components[spec['reuse']]):
            raise ValueError('reuse must name a non-discriminator factory component; generator/discriminator cannot be aliases')
    for name in ("generator", "discriminator"):
        if not components[name]["trainable"]:
            raise ValueError(f"The adversarial loop requires trainable {name}; frozen auxiliary modules are supported")
    if "candidate" not in components["discriminator"]["inputs"].values():
        raise ValueError("discriminator must bind its candidate input to 'candidate'")
    _factory(result["data"]["factory"], {"gaussian_grid", "paired_linear", "image_folder"}, "data.factory")
    if not isinstance(result["data"]["args"], dict):
        raise ValueError("data.args must be a table")
    if result["prior"]["kind"] not in {"particles", "mog", "gaussian"}:
        raise ValueError("prior.kind must be particles, mog, or gaussian")
    if not isinstance(result["prior"]["args"], dict):
        raise ValueError("prior.args must be a table")
    prior = result['prior']
    if prior['initialization_device'] not in ('execution', 'cpu'):
        raise ValueError('prior.initialization_device must be execution or cpu')
    if prior['initialization_seed'] is not None:
        _positive(prior['initialization_seed'], 'prior.initialization_seed', integer=True, zero=True)
        if prior['kind'] == 'gaussian':
            raise ValueError('GaussianPrior has no initialization RNG; omit prior.initialization_seed')
    if prior['fixed_sigma'] is not None:
        _positive(prior['fixed_sigma'], 'prior.fixed_sigma', zero=True)
        if prior['kind'] != 'mog':
            raise ValueError('prior.fixed_sigma requires a MoG prior')
    if prior['kind'] == 'mog':
        if 'sigma' in prior['args']:
            _positive(prior['args']['sigma'], 'prior.args.sigma', zero=True)
            if prior['fixed_sigma'] is not None:
                raise ValueError('Choose prior.fixed_sigma or prior.args.sigma, not both')
        if 'sigma_rel' in prior['args']:
            _positive(prior['args']['sigma_rel'], 'prior.args.sigma_rel', zero=True)
    if "dtype" in result["prior"]["args"] or ("device" in result["prior"]["args"] and result["prior"]["args"]["device"] != result["training"]["device"]):
        raise ValueError("Training owns prior device and float32 dtype; omit prior.args.device/dtype or match training.device")
    _positive(result["prior"]["args"].get("z_dim"), "prior.args.z_dim", integer=True)
    if result["adversarial"]["mode"] not in {"vanilla", "rp", "ra"} or result["adversarial"]["loss_type"] not in {"hinge", "logistic", "wasserstein", "lsgan"}:
        raise ValueError("Unsupported adversarial loss_type or mode")
    _positive(result["adversarial"]["weight"], "adversarial.weight", zero=True)
    penalty = result["gradient_penalty"]
    if penalty["arm"] not in {"a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp", "f_none", "g_interp_cap"}:
        raise ValueError("Unknown gradient_penalty.arm")
    if penalty["norm"] not in {"l1", "l2", "linf"} or penalty["target_anneal"] not in {"none", "linear", "delayed"} or penalty["method"] not in {"autograd", "finite_difference"}:
        raise ValueError("Invalid gradient penalty norm, target_anneal, or method")
    for key in ("coeff", "kappa"):
        _positive(penalty[key], f"gradient_penalty.{key}", zero=True)
    _positive(penalty["fd_eps"], "gradient_penalty.fd_eps")
    _positive(penalty["lazy_k"], "gradient_penalty.lazy_k", integer=True)
    _positive(penalty["total_steps"], "gradient_penalty.total_steps", integer=True, zero=True)
    if penalty["target_anneal"] != "none" and penalty["total_steps"] <= 0:
        raise ValueError("Annealed gradient penalty requires its explicit total_steps")
    if penalty["arm"] == "a_r1r2" and penalty["norm"] != "l2":
        raise ValueError("R1/R2 requires the L2 norm")
    if penalty["method"] == "finite_difference" and (penalty["arm"] != "b_cap" or penalty["norm"] != "l2"):
        raise ValueError("Finite-difference penalty supports only L2 b_cap")
    if result["prior_regularizer"]["rows"] not in {"sampled_unique", "full"}:
        raise ValueError("prior_regularizer.rows must be sampled_unique or full")
    for key in ("weight", "target_std"):
        _positive(result["prior_regularizer"][key], f"prior_regularizer.{key}", zero=True)
    _positive(result["prior_regularizer"]["eps"], "prior_regularizer.eps")
    if not isinstance(result["objectives"], list):
        raise ValueError("objectives must be an array of tables")
    for i, term in enumerate(result["objectives"]):
        _spec(term, f"objectives[{i}]", objectives=True)
    objective_ids = []
    for term in result["objectives"]:
        if 'id' in term and (not isinstance(term['id'], str) or _TERM_ID.fullmatch(term['id']) is None):
            raise ValueError('Objective id must be 1–128 letters, digits, dots, underscores or hyphens')
        objective_ids.append(objective_id(term))
    if len(set(objective_ids)) != len(objective_ids):
        raise ValueError('Objective IDs must be unique; give repeated objectives explicit IDs')
    validate_metrics(result)
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        _positive(result["optimizer"][key], f"optimizer.{key}")
    if result['optimizer']['implementation'] not in ('device_adam', 'torch_fused_adam'):
        raise ValueError('optimizer.implementation must be device_adam or torch_fused_adam')
    for key in ("betas", "prior_betas"):
        values = result["optimizer"][key]
        if not isinstance(values, list) or len(values) != 2 or any(type(x) not in (int, float) or not 0 <= x < 1 for x in values):
            raise ValueError(f"optimizer.{key} must contain two numbers in [0,1)")
    for key in ("steps", "batch_size"):
        _positive(result["training"][key], f"training.{key}", integer=True)
    for key in ("ema", "lr_anneal_start", "lr_floor"):
        v = result["training"][key]
        if type(v) not in (int, float) or not 0 <= v <= 1 or (key == "ema" and v == 1):
            raise ValueError(f"Invalid training.{key}")
    validate_device(result["training"]["device"])
    if result['training']['phase_draws'] not in ('shared', 'independent'):
        raise ValueError('training.phase_draws must be shared or independent')
    if result['training']['data_rng_device'] not in ('cpu', 'execution'):
        raise ValueError('training.data_rng_device must be cpu or execution')
    for key in ('data_seed_offset', 'prior_seed_offset'):
        _positive(result['training'][key], 'training.' + key, integer=True, zero=True)
    backend = result['training']['backend']
    _keys(backend, ('deterministic_algorithms', 'cudnn_deterministic', 'cudnn_benchmark',
                    'matmul_allow_tf32', 'cudnn_allow_tf32', 'cublas_workspace_config'), 'training.backend')
    for key, value in backend.items():
        if key == 'cublas_workspace_config':
            if value not in (':4096:8', ':16:8'):
                raise ValueError('training.backend.cublas_workspace_config must be :4096:8 or :16:8')
        elif type(value) is not bool:
            raise ValueError(f'training.backend.{key} must be boolean')
    for section in ("training", "sampling"):
        _positive(result[section]["seed"], f"{section}.seed", integer=True, zero=True)
    _positive(result["sampling"]["count"], "sampling.count", integer=True)
    sampling = result['sampling']
    if 'generated' in sampling and (not isinstance(sampling['generated'], str)
            or not sampling['generated'].startswith('components.')
            or len(sampling['generated'].split('.')) < 2):
        raise ValueError('sampling.generated must bind a component output')
    views = sampling.get('views', {})
    from .previews import NAME, MAX_EXTRA_GRIDS
    if (not isinstance(views, dict) or len(views) > MAX_EXTRA_GRIDS
            or any(not isinstance(k, str) or NAME.fullmatch(k) is None
                   or k in {'g', 'x', 'comparison'} or not isinstance(v, str) or not v
                   for k, v in views.items())):
        raise ValueError('sampling.views requires at most four uniquely named output bindings')
    comparison = sampling.get('comparison', [])
    if (not isinstance(comparison, list) or (comparison and not 2 <= len(comparison) <= 4)
            or any(not isinstance(column, dict) or set(column) != {'label', 'binding'}
                   or not isinstance(column['label'], str) or not 1 <= len(column['label']) <= 32
                   or not column['label'].isascii() or not column['label'].isprintable()
                   or not isinstance(column['binding'], str) or not column['binding']
                   for column in comparison)):
        raise ValueError('sampling.comparison requires two to four labelled image bindings')
    _resolve_adversarial_terms(result)
    paths = [p for c in components.values() for p in c["inputs"].values()] + [p for t in result["objectives"] for p in t["inputs"].values()]
    for term in result.get("adversarial_terms", ()):
        paths.extend((term["real"], term["fake"], *term["inputs"].values()))
    # Particle IDs have their own validation below, including their field name
    # in errors and requiring an output of the selected inference graph.
    inference_paths = sampling_bindings({k: v for k, v in sampling.items() if k != 'particle_ids'}, preview=True)
    paths += inference_paths + evaluation_bindings(result)
    for path in paths:
        if not isinstance(path, str) or not path or not all(path.split('.')):
            raise ValueError('I/O bindings must be nonempty dotted paths')
        parts = path.split(".")
        if parts[0] not in {"batch", "latent", "generated", "candidate", "components", "prior"}:
            raise ValueError(f"Unknown binding root: {path}")
        if parts[0] == 'prior' and (path not in ('prior.means', 'prior.sigma') or prior['kind'] != 'mog'):
            raise ValueError('prior bindings require MoG prior.means or prior.sigma')
        if parts[0] == "components" and (len(parts) < 2 or parts[1] not in components or parts[1] == "discriminator"):
            raise ValueError(f"Unknown or unavailable component binding: {path}")
    # Every declared component must participate; accidental dormant modules are not
    # an experiment. Trace the actual graph, including objective-only encoders.
    reachable = set()
    def visit(name, active):
        if name in active:
            raise ValueError(f"Cyclic component binding at {name}")
        if name in reachable:
            return
        for path in components[name]["inputs"].values():
            if path.startswith("components."):
                visit(path.split(".")[1], active | {name})
        reachable.add(name)
        if 'reuse' in components[name]:
            reachable.add(components[name]['reuse'])
    visit("generator", set())
    training_reachable = set(reachable)
    for binding in inference_paths:
        if binding.startswith('components.'):
            visit(binding.split('.')[1], set())
    if 'particle_ids' in result['sampling']:
        binding = result['sampling']['particle_ids']
        if (not isinstance(binding, str) or len(binding.split('.')) < 3
                or binding.split('.')[0] != 'components'
                or binding.split('.')[1] not in reachable
                or not all(binding.split('.'))):
            raise ValueError('sampling.particle_ids must bind an output of the sampling dependency graph')
    reachable = training_reachable
    g_reachable = set(reachable)
    for term in result["objectives"]:
        for path in term["inputs"].values():
            if path.startswith("components."):
                visit(path.split(".")[1], set())
    for term in result.get("adversarial_terms", ()):
        for path in (term["real"], term["fake"]):
            if path.startswith("components."):
                visit(path.split(".")[1], set())
    g_reachable.update(reachable)
    visit("discriminator", set())
    critic_components = {"discriminator"}
    for term in result.get("adversarial_terms", ()):
        # The scoring module is reachable, like the discriminator. Its non-candidate
        # inputs are detached conditioning and do not make that producer generator-reachable.
        critic_components.add(term["component"])
        reachable.add(term["component"])
        for path in term["inputs"].values():
            if path != "candidate" and path.startswith("components."):
                visit(path.split(".")[1], set())
    if set(components) - reachable:
        raise ValueError(f"Disconnected components have no effect: {sorted(set(components) - reachable)}")
    for name in reachable - g_reachable - critic_components:
        if components[name]["trainable"]:
            raise ValueError(f"Component {name} is only discriminator conditioning, which is detached; mark it trainable=false or bind it into a generator objective")
    warnings = []
    if result["prior"]["kind"] == "gaussian":
        if result["prior_regularizer"]["weight"] != 0:
            raise ValueError("GaussianPrior has no learned table; set prior_regularizer.weight=0 explicitly")
        warnings.append("GaussianPrior has no trainable table: optimizer.prior_lr_mult and prior_betas, and prior_regularizer row settings, are not applicable.")
    elif result["prior"]["args"].get("learnable") is False:
        warnings.append("The prior table is frozen: prior optimizer settings do not apply and its regularizer contributes no trainable gradient.")
    if result['metrics']['custom']:
        warnings.append("Custom metrics execute trusted Python in bounded workers; input/output and deadlines are bounded, arbitrary allocations or descendants are not sandboxed.")
    warnings.extend(evaluation_warnings(result))
    match = {k: v for k, v in result.items() if k != "metrics"} == {k: v for k, v in DEFAULT.items() if k != "metrics"}
    if not match:
        warnings.append("Custom resolved recipe: runnable combinations are unqualified until separately evaluated.")
    if any(":" in s["factory"] for s in components.values()) or ":" in result["data"]["factory"] or any(":" in t["factory"] for t in result["objectives"]):
        warnings.append("Custom constructors execute trusted Python code when training or sampling; configuration is not a sandbox.")
    result["qualification"] = {"status": "pending-runtime" if match else "unqualified", "recipe_match": match, "scope": "numerical-reference", "image_quality": False, "distributed": False, "resume": False}
    result["warnings"] = warnings
    return result


def _listed_values(config, keys):
    """Copy listed keys. Include extra adversarial terms only when that list is non-empty.

    A legacy resolved config and an old checkpoint both lack the key. Looking it
    up with ``get`` keeps those hashes identical and does not KeyError.
    """
    values = {key: deepcopy(config[key]) for key in keys}
    terms = config.get("adversarial_terms")
    if terms:
        values["adversarial_terms"] = deepcopy(terms)
    return values


def config_values(config):
    return _listed_values(config, DEFAULT)


def numerical_values(config):
    return _listed_values(config, (key for key in DEFAULT if key != "metrics"))


def resume_compatible(config, original, *, include_observation=False):
    """Allow constant-LR extension, pinned-image skip opt-in and pixel caching.

    Keep fingerprints unchanged: existing checkpoints retain their original
    identities, and each new checkpoint records the extended configuration.
    JSON comparison preserves distinctions such as boolean vs integer arguments.
    Numerical-only distributed checkpoint configurations are accepted too.
    """
    try:
        values = config_values if include_observation else numerical_values
        current, saved = values(config), values(original)
        factory = 'hypergan.colorization_data:ColorizationData'
        if current['data']['factory'] == saved['data']['factory'] == factory:
            left, right = current['data']['args'], saved['data']['args']
            # Derived pixels are checked and source hashes still verified on
            # every access. Cache placement/presence cannot change training.
            caches = [args.get('cache_dir') for args in (left, right)]
            if all(c is None or (isinstance(c, str) and c.strip()) for c in caches):
                for args in (left, right):
                    args.pop('cache_dir', None)
            keys = {'bad_image_policy': 'error', 'max_bad_images': 100,
                    'max_consecutive_bad_images': 8}
            policies = [{k: args.get(k, default) for k, default in keys.items()}
                        for args in (left, right)]
            valid = all(p['bad_image_policy'] in ('error', 'skip')
                        and all(type(p[k]) is int and p[k] > 0
                                for k in ('max_bad_images', 'max_consecutive_bad_images'))
                        for p in policies)
            enabling = (policies[0]['bad_image_policy'] == 'skip'
                        and policies[1]['bad_image_policy'] == 'error'
                        and left.get('split', 'train') == 'train'
                        and left.get('shuffle', True) is True)
            if valid and (policies[0] == policies[1] or enabling):
                # Keep the pinned inventory and every other recipe field exact.
                # New checkpoints record the explicit policy in their config.
                for args in (left, right):
                    for key in keys:
                        args.pop(key, None)
        new_steps, old_steps = current['training']['steps'], saved['training']['steps']
        if new_steps != old_steps:
            if (type(new_steps) is not int or type(old_steps) is not int
                    or new_steps < old_steps
                    or current['training']['lr_floor'] != 1.0
                    or saved['training']['lr_floor'] != 1.0):
                return False
            current['training']['steps'] = old_steps
        return (json.dumps(current, sort_keys=True, allow_nan=False)
                == json.dumps(saved, sort_keys=True, allow_nan=False))
    except (KeyError, TypeError, ValueError):
        return False


def observation_fingerprint(config):
    return hashlib.sha256(json.dumps(config['metrics'], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fingerprint(config):
    """Numerical recipe identity; observation publication can change on resume."""
    return hashlib.sha256(json.dumps(numerical_values(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path):
    path = Path(path)
    if path.is_dir():
        path = path / "config.toml"
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    for spec in raw.get("components", {}).values():
        args = spec.get("args", {})
        network_files = args.pop('network_files', {})
        if not isinstance(network_files, dict):
            raise ValueError('args.network_files must map template names to .hndl file paths')
        if network_files:
            sources = args.setdefault('networks', {})
            if not isinstance(sources, dict) or set(sources) & set(network_files):
                raise ValueError('Specify each network template once, as source or file')
            for name, file in network_files.items():
                if not isinstance(file, str):
                    raise ValueError('args.network_files paths must be strings')
                sources[name] = read_source(path.parent / file)
        if spec.get("factory") == "hndl" and "file" in args:
            if "source" in args:
                raise ValueError("HNDL component must specify exactly one of source or file")
            args["source"] = read_source(path.parent / args.pop("file"))
    return resolve_config(raw)


def validate_device(device):
    """Check a device request without importing Torch or probing hardware."""
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", device) is None:
        raise ValueError("training.device must be cpu, cuda, or cuda:N (a nonnegative visible GPU index)")
    return device


def write_default(path, *, device="cuda"):
    """Create a GPU-first project; CPU numerical fixtures opt in explicitly."""
    validate_device(device)
    path = Path(path)
    if path.suffix != ".toml":
        path = path / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(DEFAULT_TOML.replace('device = "cpu"', f'device = "{device}"'))
    return path


def list_recipes():
    return [{"name": DEFAULT["name"], "scope": "numerical-reference", "description": "CPU 2D Gaussian-grid integration fixture; no image or distributed qualification"}]


def sampling_bindings(sampling, *, preview=False):
    """Declared inference dependencies, including optional preview-only views."""
    paths = [sampling[key] for key in ('generated', 'particle_ids') if key in sampling]
    if preview:
        paths += list(sampling.get('views', {}).values())
        paths += [column['binding'] for column in sampling.get('comparison', [])]
    return paths


def evaluation_bindings(config):
    """Explicit snapshot output overrides; omitted values use sampling output."""
    return [spec['evaluation']['generated'] for spec in config['metrics']['custom'].values()
            if spec['mode'] == 'snapshot' and 'generated' in spec['evaluation']]
