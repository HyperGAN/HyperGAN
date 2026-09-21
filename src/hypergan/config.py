"""Torch-free, explicit configuration for the first reference runtime.

Constructor paths are trusted Python code, imported only by train/sample.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path

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
        "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [64, 64]}, "inputs": {"x": "latent"}, "trainable": True},
        "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [64, 64]}, "inputs": {"x": "candidate"}, "trainable": True},
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
factory = "mlp"
inputs = { x = "latent" }
[components.generator.args]
input_dim = 4
output_dim = 2
hidden = [64, 64]

[components.discriminator]
factory = "mlp"
inputs = { x = "candidate" }
[components.discriminator.args]
input_dim = 2
output_dim = 1
hidden = [64, 64]

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
        _factory(value.get("factory"), {"mse", "l1"} if objectives else {"mlp", "linear", "identity"}, location)
    value.setdefault("args", {})
    if not isinstance(value["args"], dict):
        raise ValueError(f"{location}.args must be a table")
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


def resolve_config(raw):
    """Resolve omitted defaults without importing or executing custom constructors."""
    _keys(raw, DEFAULT, "configuration")
    result = deepcopy(DEFAULT)
    for key, value in raw.items():
        if key == "components":
            # Explicit components replace the graph; no hidden old bindings survive.
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict):
            allowed = set(result[key]) | ({'particle_ids'} if key == 'sampling' else set())
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
        if 'id' in term and (not isinstance(term['id'], str) or re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', term['id']) is None):
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
    paths = [p for c in components.values() for p in c["inputs"].values()] + [p for t in result["objectives"] for p in t["inputs"].values()]
    for path in paths:
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
    if 'particle_ids' in result['sampling']:
        binding = result['sampling']['particle_ids']
        if (not isinstance(binding, str) or len(binding.split('.')) < 3
                or binding.split('.')[0] != 'components'
                or binding.split('.')[1] not in reachable
                or not all(binding.split('.'))):
            raise ValueError('sampling.particle_ids must bind an output of the generator dependency graph')
    g_reachable = set(reachable)
    for term in result["objectives"]:
        for path in term["inputs"].values():
            if path.startswith("components."):
                visit(path.split(".")[1], set())
    g_reachable.update(reachable)
    visit("discriminator", set())
    if set(components) - reachable:
        raise ValueError(f"Disconnected components have no effect: {sorted(set(components) - reachable)}")
    for name in reachable - g_reachable - {"discriminator"}:
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


def config_values(config):
    return {key: deepcopy(config[key]) for key in DEFAULT}


def numerical_values(config):
    return {key: deepcopy(config[key]) for key in DEFAULT if key != 'metrics'}


def resume_compatible(config, original, *, include_observation=False):
    """Allow only an increased stopping step when the saved LR is constant.

    Keep fingerprints unchanged: existing checkpoints retain their original
    identities, and each new checkpoint records the extended configuration.
    JSON comparison preserves distinctions such as boolean vs integer arguments.
    Numerical-only distributed checkpoint configurations are accepted too.
    """
    try:
        values = config_values if include_observation else numerical_values
        current, saved = values(config), values(original)
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
        return resolve_config(tomllib.load(stream))


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
