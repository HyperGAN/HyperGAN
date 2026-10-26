"""Model files: the editable, reviewable description of a HyperGAN model.

A model file is TOML. Networks declare a ``role``; every loss the model trains
is listed together under ``[[losses]]``; user Python is named by import path
(``module:object``) and never copied. ``lower()`` turns a model file into the
engine's recipe (``hypergan.config.resolve_config`` input) without importing
Torch or user code, so every process, worker or host rebuilds the same model
from the file (or from the lowered recipe a run records) plus importable code.

Layout::

    name = "demo/gaussian-grid"

    [data]                      # one of: builtin / factory (batch level) / dataset (item level)
    builtin = "gaussian_grid"

    [prior]
    kind = "particles"          # particles | mog | gaussian; other keys are prior args
    z_dim = 4
    num_particles = 20000

    [networks.generator]
    role = "generator"          # generator | critic | encoder | auxiliary
    module = "my_project.nets:Generator"   # or hndl = "g.hndl" / source = "..."
    args = { hidden = 64 }
    inputs = { x = "latent" }

    [networks.critic]
    role = "critic"
    hndl = "critic.hndl"
    input_shape = ["B", 2]
    output_shape = ["B", 1]
    inputs = { x = "candidate" }

    [[losses]]
    type = "adversarial"        # ParticleGAN loss and critic regularization
    critic = "critic"
    penalty = 1.0

    [[losses]]
    type = "prior"              # particle-table spread regularizer

    [train]
    steps = 1000
    batch_size = 64

Bindings name a value: ``latent``, ``batch.<field>``, ``generated``,
``candidate``, ``prior.means``/``prior.sigma`` or a network by its name
(``encoder`` or ``encoder.<output>``).
"""
from copy import deepcopy
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .network_config import read_source

FORMAT = 'hypergan-model/1'
TOP_LEVEL = {'format', 'name', 'description', 'defaults', 'recipe', 'data', 'dataset', 'prior', 'networks', 'losses', 'penalty', 'train',
             'sampling', 'samplers', 'metrics', 'evaluations', 'checkpoints', 'run', 'extends'}

CHECKPOINT_FIELDS = {'keep_last', 'keep_best', 'keep_every', 'export_bundle'}
ROLES = ('generator', 'critic', 'encoder', 'auxiliary')
RESERVED = {'latent', 'batch', 'generated', 'candidate', 'prior', 'components', 'update', 'evaluation',
            'reference'}
HNDL_EXTRA_FIELDS = ('parameters', 'pretrained_providers', 'input_dtype', 'networks', 'network_files')
NETWORK_FIELDS = {'role', 'hndl', 'file', 'source', 'module', 'reuse', 'args', 'inputs', 'input_shape', 'output_shape',
                  'trainable', 'freeze_parameters', 'optimizer', *HNDL_EXTRA_FIELDS}
OPTIMIZER_FIELDS = {'lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas', 'implementation',
                    'd_guard_ratio', 'd_guard_min_steps', 'latent_damping_max_rate'}
TRAINING_FIELDS = {'steps', 'batch_size', 'seed', 'device', 'initialization', 'ema', 'lr_anneal_start', 'lr_floor',
                   'network_lr_floor', 'network_lr_horizon_cap', 'input_noise_std', 'input_noise_anneal_end',
                   'output_noise_std', 'output_noise_warmup', 'phase_draws', 'data_rng_device',
                   'data_seed_offset', 'prior_seed_offset', 'backend', 'every_steps'}
RUN_CADENCE_FIELDS = {'checkpoint_every', 'preview_every', 'progress_every'}
RUN_FIELDS = {'devices', 'env', 'viewer', 'cpu_threads', *RUN_CADENCE_FIELDS}
PENALTY_FIELDS = {'kappa', 'lazy_k', 'anchor_weight', 'anchor_decay'}
_EVAL_FIELDS = {'fn', 'factory', 'args', 'inputs', 'every', 'every_steps', 'device', 'samples', 'batch_size',
                'seed', 'data', 'output', 'target', 'timeout', 'on_error', 'on_busy', 'label', 'unit',
                'direction', 'description', 'reduce', 'kind'}
_PATH_KEYS = {'hndl', 'file', 'root', 'manifest', 'cache_dir', 'weights_path', 'source_path', 'path', 'network_files'}
PRIOR_FIELDS = {'kind', 'initialization_device', 'initialization_seed', 'fixed_sigma'}
BUILTIN_DATA = ('gaussian_grid', 'paired_linear', 'image_folder')
ITEM_DATA = 'hypergan.item_data:ItemData'
SCALAR_FUNCTION = 'hypergan.plugin_functions:ScalarFunction'
SNAPSHOT_FUNCTION = 'hypergan.plugin_functions:SnapshotFunction'
UPDATE_SCALARS = ('d_loss', 'g_loss', 'd_adversarial', 'g_adversarial', 'd_adversarial_weighted',
                  'g_adversarial_weighted', 'gradient_penalty', 'prior_loss', 'lr_scale', 'step', 'step_seconds')
_ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}')

DEFAULT_CHECKPOINT_POLICY = {
    'keep_last': 3,
    'keep_best': {'metric': 'fid50k_train', 'mode': 'min', 'count': 2},
    'keep_every': 50000,
    'export_bundle': 'on_eval',
}


def normalize_checkpoint_policy(policy=None):
    """Normalize a [checkpoints] retention policy dict, expanding shorthands and filling defaults."""
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise ValueError(f"Checkpoint retention policy must be a dict, got {type(policy).__name__}")

    unknown = set(policy) - {'keep_last', 'keep_best', 'keep_every', 'export_bundle'}
    if unknown:
        raise ValueError(f"Unknown field(s) in checkpoint retention policy: {', '.join(sorted(unknown))}")

    keep_last = policy.get('keep_last', 3)
    if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 0:
        raise ValueError("keep_last must be a non-negative integer")

    keep_every = policy.get('keep_every', 50000)
    if keep_every is not None and (not isinstance(keep_every, int) or isinstance(keep_every, bool) or keep_every < 0):
        raise ValueError("keep_every must be a non-negative integer or None")

    export_bundle = policy.get('export_bundle', 'on_eval')
    if export_bundle is not None and export_bundle not in ('on_eval', 'never', 'on_checkpoint'):
        raise ValueError("export_bundle must be one of 'on_eval', 'never', 'on_checkpoint', or None")

    raw_best = policy.get('keep_best', {'metric': 'fid50k_train', 'mode': 'min', 'count': 2})
    if raw_best is None or raw_best is False or raw_best == 0:
        keep_best = None
    elif isinstance(raw_best, int) and not isinstance(raw_best, bool):
        if raw_best < 0:
            raise ValueError("keep_best count must be a non-negative integer")
        keep_best = {'metric': 'fid50k_train', 'mode': 'min', 'count': raw_best}
    elif isinstance(raw_best, dict):
        metric = raw_best.get('metric', 'fid50k_train')
        if not isinstance(metric, str):
            raise ValueError("keep_best.metric must be a string")
        mode = raw_best.get('mode', 'min')
        if mode not in ('min', 'max'):
            raise ValueError("keep_best.mode must be 'min' or 'max'")
        count = raw_best.get('count', 2 if 'metric' not in raw_best else 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("keep_best.count must be a non-negative integer")
        keep_best = {'metric': metric, 'mode': mode, 'count': count}
    else:
        raise ValueError("keep_best must be an integer count or table with {metric, mode, count}")

    return {
        'keep_last': keep_last,
        'keep_best': keep_best,
        'keep_every': keep_every,
        'export_bundle': export_bundle,
    }


class ModelFileError(ValueError):
    """A model file field is missing, unknown or inconsistent; the message names it."""


ConfigError = ModelFileError


def is_model_file(raw):
    """Model files have ``networks``; engine recipes have ``components``."""
    if isinstance(raw, dict) and 'networks' in raw and 'components' in raw:
        raise ModelFileError("Cannot mix 'networks' (hypergan-model/1) and 'components' (legacy engine recipe) in the same file")
    return isinstance(raw, dict) and 'networks' in raw and 'components' not in raw


def require_networks(raw):
    """Reject a file that declares no model, so it never silently trains the built-in default.

    Without ``[networks]`` (model file) or ``[components]`` (engine recipe) every
    other table, such as the internal resolved ``[training]``, would only adjust
    the packaged 2-D reference recipe.
    """
    if isinstance(raw, dict) and 'networks' not in raw and 'components' not in raw:
        internal = sorted(key for key in raw if key not in TOP_LEVEL)
        hint = ''
        if internal:
            tables = ', '.join(f'[{key}]' for key in internal)
            hint = (f'; {tables} is an internal resolved-recipe table, not a model-file table' if len(internal) == 1
                    else f'; {tables} are internal resolved-recipe tables, not model-file tables')
        raise ModelFileError(f'No [networks] declared{hint}. A {FORMAT} file declares [networks.<name>] with a role '
                             'and [[losses]], and [train] for training settings (see docs/configuration.md), '
                             'or extends a packaged model such as extends = "gaussian-grid".')


def producer(path):
    """The network whose output a binding reads directly, if any."""
    parts = path.split('.')
    if parts[0] in ('generated', 'candidate'):
        return 'generator'
    if parts[0] == 'components' and len(parts) > 1:
        return parts[1]
    return None


def upstream(path, components, *, seen=None):
    """Networks a binding's value depends on (reuse follows the shared parameters)."""
    seen = set() if seen is None else seen
    source = producer(path)
    if source is None or source in seen or source not in components:
        return seen
    seen.add(source)
    spec = components[source]
    if 'reuse' in spec:
        seen.add(spec['reuse'])
    for value in spec.get('inputs', {}).values():
        upstream(value, components, seen=seen)
    return seen


def _table(value, location, allowed=None):
    if not isinstance(value, dict):
        raise ModelFileError(f'{location} must be a table')
    if allowed is not None:
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ModelFileError(f'Unknown {location} field(s): {", ".join(unknown)}; expected one of {", ".join(sorted(allowed))}')
    return value


def _one_of(spec, keys, location):
    present = [key for key in keys if key in spec]
    if len(present) != 1:
        raise ModelFileError(f'{location} needs exactly one of {", ".join(keys)}' + (f' (found {", ".join(present)})' if present else ''))
    return present[0]


class _Lowering:
    def __init__(self, raw, base):
        self.raw, self.base = raw, Path(base) if base is not None else None
        self.networks = _table(raw.get('networks'), 'networks')
        if not self.networks:
            raise ModelFileError('networks must declare at least a generator and a critic')
        self.names = {}

    # -- names and bindings -------------------------------------------------
    def _plan_names(self, losses):
        roles = {}
        for name, spec in self.networks.items():
            if not isinstance(name, str) or not name.isidentifier() or name in RESERVED:
                raise ModelFileError(f'Network name {name!r} must be a Python identifier and not one of {", ".join(sorted(RESERVED))}')
            _table(spec, f'networks.{name}', NETWORK_FIELDS)
            if 'optimizer' in spec:
                role = spec.get('role')
                group = 'critic Adam (shared by all critics)' if role == 'critic' else 'generator-side Adam'
                raise ModelFileError(f"Engine limit: networks.{name} is trained by {group}; per-network optimizers are not yet supported. Remove optimizer= from {name}.")
            role = spec.get('role')
            if role not in ROLES:
                raise ModelFileError(f'networks.{name}.role must be one of {", ".join(ROLES)}')
            roles.setdefault(role, []).append(name)
        generators = roles.get('generator', [])
        primary_generators = [n for n in generators if 'reuse' not in self.networks[n]]
        if len(primary_generators) != 1:
            raise ModelFileError(f'Exactly one network may have role = "generator" today (found {len(primary_generators)}); '
                                 'more generators need an engine with several generator outputs')
        primary_gen = primary_generators[0]
        adversarial = [loss for loss in losses if loss.get('type') == 'adversarial']
        if not adversarial:
            raise ModelFileError('losses must include at least one type = "adversarial" term naming a critic')
        critics = set(roles.get('critic', []))
        for index, loss in enumerate(adversarial):
            if loss.get('critic') not in critics:
                raise ModelFileError(f'Adversarial loss {index} critic must name a network with role = "critic"; '
                                     f'critics: {", ".join(sorted(critics)) or "none"}')
        unused = critics - {loss['critic'] for loss in adversarial}
        if unused:
            raise ModelFileError(f'Critic(s) {", ".join(sorted(unused))} are in no adversarial loss')
        primary = adversarial[0]['critic']
        self.names = {name: name for name in self.networks}
        self.names[primary_gen] = 'generator'
        self.names[primary] = 'discriminator'
        used_lowered = set(self.names.values())
        for name, lowered in list(self.names.items()):
            if lowered == 'discriminator' and name != primary:
                if self.networks[name].get('role') == 'critic':
                    alt = 'critic_discriminator'
                    while alt in used_lowered:
                        alt += '_2'
                    self.names[name] = alt
                    used_lowered.add(alt)
                else:
                    raise ModelFileError(f'networks.{name}: the names generator/discriminator are reserved for the '
                                         'generator and the first adversarial loss critic; rename this network')
            elif lowered == 'generator' and name != primary_gen:
                if self.networks[name].get('role') == 'generator' and 'reuse' in self.networks[name]:
                    alt = 'generator_reuse'
                    while alt in used_lowered:
                        alt += '_2'
                    self.names[name] = alt
                    used_lowered.add(alt)
                else:
                    raise ModelFileError(f'networks.{name}: the names generator/discriminator are reserved for the '
                                         'generator and the first adversarial loss critic; rename this network')
        self.generator, self.primary = primary_gen, primary

    def bind(self, path, location):
        if not isinstance(path, str) or not path or not all(path.split('.')):
            raise ModelFileError(f'{location} must be a dotted binding such as latent, batch.real or encoder')
        head, _, rest = path.partition('.')
        if head == 'components':
            head, _, rest = rest.partition('.')
        elif head in RESERVED:
            return path
        if head not in self.networks:
            raise ModelFileError(f'{location} = {path!r}: {head!r} is neither a network nor one of latent, batch, generated, candidate, prior')
        return 'components.' + self.names[head] + ('.' + rest if rest else '')

    def bind_all(self, inputs, location):
        _table(inputs, location)
        if not inputs:
            raise ModelFileError(f'{location} must bind at least one argument')
        return {key: self.bind(value, f'{location}.{key}') for key, value in inputs.items()}

    # -- sections --------------------------------------------------------------
    def component(self, name, spec):
        location = f'networks.{name}'
        raw_inputs = spec.get('inputs')
        if raw_inputs is None and spec.get('role') == 'critic':
            for loss in self.raw.get('losses', []):
                if isinstance(loss, dict) and loss.get('type') == 'adversarial' and loss.get('critic') == name and isinstance(loss.get('inputs'), dict):
                    raw_inputs = loss['inputs']
                    break
        if 'reuse' in spec:
            target = spec['reuse']
            if target not in self.networks:
                raise ModelFileError(f'{location}.reuse must name another network')
            result = {'reuse': self.names[target], 'inputs': self.bind_all(raw_inputs, f'{location}.inputs')}
            if 'freeze_parameters' in spec:
                result['freeze_parameters'] = spec['freeze_parameters']
            return result
        kind = _one_of(spec, ('hndl', 'file', 'source', 'module'), location)
        args = deepcopy(_table(spec.get('args', {}), f'{location}.args'))
        if kind == 'module':
            if ('input_shape' in spec or 'output_shape' in spec) or (set(HNDL_EXTRA_FIELDS) & set(spec)):
                raise ModelFileError(f'{location}: input_shape/output_shape describe HNDL networks; pass module settings in args')
            factory = spec['module']
            if callable(factory):
                from .api import import_path
                factory = import_path(factory, f'{location}.module')
        else:
            factory = 'hndl'
            if kind in ('hndl', 'file'):
                file = Path(spec[kind])
                if not file.is_absolute():
                    if self.base is None:
                        raise ModelFileError(f'{location}.{kind} is relative but the model has no file location; use an absolute path or source')
                    file = self.base / file
                try:
                    args['source'] = read_source(file)
                except OSError as exc:
                    raise ModelFileError(f'{location}.{kind}: cannot read {file}: {exc}') from exc
            else:
                args['source'] = spec['source']
            for key in ('input_shape', 'output_shape'):
                if key not in spec:
                    raise ModelFileError(f'{location}.{key} is required for an HNDL network')
                args[key] = deepcopy(spec[key])
            for key in HNDL_EXTRA_FIELDS:
                if key in spec:
                    args.setdefault(key, deepcopy(spec[key]))
            if 'network_files' in args:
                net_files = _table(args.pop('network_files'), f'{location}.network_files')
                inline_nets = deepcopy(_table(args.get('networks', {}), f'{location}.networks'))
                for sub_name, sub_path in net_files.items():
                    sub_file = Path(sub_path)
                    if not sub_file.is_absolute():
                        if self.base is None:
                            raise ModelFileError(f'{location}.network_files.{sub_name} is relative but the model has no file location')
                        sub_file = self.base / sub_file
                    try:
                        inline_nets[sub_name] = read_source(sub_file)
                    except OSError as exc:
                        raise ModelFileError(f'{location}.network_files.{sub_name}: cannot read {sub_file}: {exc}') from exc
                args['networks'] = inline_nets
        result = {'factory': factory, 'args': args, 'inputs': self.bind_all(raw_inputs, f'{location}.inputs')}
        if 'trainable' in spec:
            result['trainable'] = spec['trainable']
        return result

    def data(self, spec, location):
        _table(spec, location)
        kind = _one_of(spec, ('builtin', 'factory', 'dataset'), location)
        spec_copy = deepcopy(spec)
        spec_copy.pop(kind)
        args = _table(spec_copy.pop('args', {}), f'{location}.args')
        spec_copy.pop('batch_size', None)
        if kind == 'builtin':
            if spec['builtin'] != 'image_folder' and ({'shuffle', 'workers'} & set(spec_copy)):
                raise ModelFileError(f'{location}: shuffle/workers apply to item-level datasets (dataset = "module:Class")')
            for k, v in spec_copy.items():
                args.setdefault(k, v)
            if spec['builtin'] == 'cifar10':
                return {'factory': 'hypergan.image_data:CIFAR10Data', 'args': args}
            if spec['builtin'] not in BUILTIN_DATA:
                raise ModelFileError(f'{location}.builtin must be one of {", ".join(BUILTIN_DATA)}')
            if spec['builtin'] == 'image_folder' and 'channels' in args:
                ch = args.pop('channels')
                if 'mode' not in args:
                    args['mode'] = 'L' if ch == 1 else 'RGB'
            return {'factory': spec['builtin'], 'args': args}
        if kind == 'factory':
            for k, v in spec_copy.items():
                args.setdefault(k, v)
            return {'factory': spec['factory'], 'args': args}
        item = {'dataset': spec['dataset'], 'args': args}
        for key in ('shuffle', 'workers'):
            if key in spec_copy:
                item[key] = spec_copy.pop(key)
        for k, v in spec_copy.items():
            args.setdefault(k, v)
        return {'factory': ITEM_DATA, 'args': item}

    def prior(self, spec):
        spec = _table(deepcopy(spec), 'prior')
        if 'type' in spec and 'kind' not in spec:
            spec['kind'] = spec.pop('type')
        result = {key: spec.pop(key) for key in PRIOR_FIELDS if key in spec}
        args = spec.pop('args', {})
        if not isinstance(args, dict):
            raise ModelFileError('prior.args must be a table')
        args.update(spec)
        result['args'] = args
        return result

    def losses(self, losses, recipe):
        recipe['objectives'] = []
        prior_terms = [loss for loss in losses if loss.get('type') == 'prior']
        if len(prior_terms) > 1:
            raise ModelFileError('List at most one type = "prior" loss')
        recipe['prior_regularizer'] = {'weight': 0.0}
        seen_ids = set()
        for index, loss in enumerate(losses):
            location = f'losses[{index}]'
            _table(loss, location)
            kind = loss.get('type')
            if kind == 'adversarial':
                _table(loss, location, {'type', 'id', 'critic', 'weight', 'penalty', 'real', 'fake', 'inputs'} | PENALTY_FIELDS)
                raw_penalty = loss.get('penalty', 1.0)
                penalty_extra = {k: deepcopy(loss[k]) for k in PENALTY_FIELDS if k in loss}
                if isinstance(raw_penalty, dict):
                    _table(raw_penalty, f'{location}.penalty', PENALTY_FIELDS | {'weight', 'coeff'})
                    penalty = raw_penalty.get('weight', raw_penalty.get('coeff', 1.0))
                    for k in PENALTY_FIELDS:
                        if k in raw_penalty:
                            penalty_extra.setdefault(k, deepcopy(raw_penalty[k]))
                else:
                    penalty = raw_penalty
                if type(penalty) not in (int, float) or penalty < 0:
                    raise ModelFileError(f'{location}.penalty is the critic regularization coefficient: a nonnegative number (0 disables it)')
                if penalty_extra:
                    recipe.setdefault('gradient_penalty', {}).update(penalty_extra)
                if loss['critic'] == self.primary and 'adversarial' not in recipe:
                    real_raw, fake_raw = loss.get('real', 'batch.real'), loss.get('fake', 'generated')
                    disc_inputs = self.networks[self.primary].get('inputs', {})
                    cond_vals = {v for v in disc_inputs.values() if v != 'candidate' and not str(v).startswith('candidate.')}
                    if real_raw != 'batch.real' and real_raw in cond_vals and 'inputs' not in loss:
                        raise ModelFileError(f'{location}: the first adversarial loss compares batch.real with generated '
                                             'through its critic\'s own inputs (engine limit); put other comparisons in later adversarial losses')
                    adv = {'weight': loss.get('weight', 1.0)}
                    if 'id' in loss:
                        if not isinstance(loss['id'], str) or _ID.fullmatch(loss['id']) is None:
                            raise ModelFileError(f'{location}.id is required: it names the loss in metrics')
                        if loss['id'] in seen_ids:
                            raise ModelFileError(f'{location}: loss id {loss["id"]!r} is repeated; give each loss an id')
                        seen_ids.add(loss['id'])
                        adv['id'] = loss['id']
                    if real_raw != 'batch.real':
                        adv['real'] = self.bind(real_raw, f'{location}.real')
                    if fake_raw != 'generated':
                        adv['fake'] = self.bind(fake_raw, f'{location}.fake')
                    if 'inputs' in loss:
                        bound_inputs = self.bind_all(loss['inputs'], f'{location}.inputs')
                        non_cand = {k: v for k, v in bound_inputs.items() if v != 'candidate' and not v.startswith('candidate.')}
                        if non_cand:
                            adv['inputs'] = non_cand
                            recipe['components']['discriminator'].setdefault('inputs', {}).update(non_cand)
                    recipe['adversarial'] = adv
                    recipe.setdefault('gradient_penalty', {})['coeff'] = penalty
                    continue
                term = {'id': loss.get('id', loss['critic']), 'component': self.names[loss['critic']],
                        'weight': loss.get('weight', 1.0), 'penalty': penalty > 0,
                        'real': self.bind(loss.get('real', 'batch.real'), f'{location}.real'),
                        'fake': self.bind(loss.get('fake', 'generated'), f'{location}.fake')}
                if penalty > 0:
                    term['penalty_coeff'] = penalty
                if 'inputs' in loss:
                    bound_inputs = self.bind_all(loss['inputs'], f'{location}.inputs')
                    non_cand = {k: v for k, v in bound_inputs.items() if v != 'candidate' and not v.startswith('candidate.')}
                    if non_cand:
                        term['inputs'] = non_cand
                if term['id'] in seen_ids:
                    raise ModelFileError(f'{location}: loss id {term["id"]!r} is repeated; give each loss an id')
                seen_ids.add(term['id'])
                recipe.setdefault('adversarial_terms', []).append(term)
            elif kind in ('objective', 'reconstruction'):
                loss = deepcopy(loss)
                id_cleared = loss.pop('__id_required__', False)
                _table(loss, location, {'type', 'id', 'fn', 'args', 'inputs', 'input', 'target', 'weight', 'detach', 'owner', 'network'})
                if kind == 'reconstruction' and not id_cleared:
                    loss.setdefault('id', 'reconstruction')
                    if 'inputs' not in loss:
                        loss.setdefault('input', 'generated')
                        loss.setdefault('target', 'batch.real')
                if not isinstance(loss.get('id'), str) or _ID.fullmatch(loss['id']) is None:
                    raise ModelFileError(f'{location}.id is required: it names the loss in metrics (loss/objectives/<id>)')
                if loss['id'] in seen_ids:
                    raise ModelFileError(f'{location}: loss id {loss["id"]!r} is repeated')
                seen_ids.add(loss['id'])
                if 'inputs' in loss:
                    if {'input', 'target'} & set(loss):
                        raise ModelFileError(f'{location}: use inputs or input/target, not both')
                    inputs = self.bind_all(loss['inputs'], f'{location}.inputs')
                else:
                    if 'input' not in loss:
                        raise ModelFileError(f'{location} needs input (and usually target) bindings')
                    inputs = {'input': self.bind(loss['input'], f'{location}.input')}
                    if 'target' in loss:
                        inputs['target'] = self.bind(loss['target'], f'{location}.target')
                fn = loss.get('fn', 'mse')
                if callable(fn):
                    from .api import import_path
                    fn = import_path(fn, f'{location}.fn')
                args = deepcopy(loss.get('args', {}))
                if fn in ('mse', 'l1'):
                    factory = fn
                else:
                    factory = 'hypergan.plugin_functions:FunctionLoss'
                    args['fn'] = fn
                term = {'id': loss['id'], 'factory': factory, 'inputs': inputs,
                        'weight': loss.get('weight', 1.0), 'args': args}
                if 'detach' in loss:
                    term['detach'] = list(loss['detach'])
                if 'owner' in loss or 'network' in loss:
                    term['owner'] = loss.get('owner') or loss.get('network')
                recipe['objectives'].append(term)
            elif kind == 'prior':
                _table(loss, location, {'type', 'weight', 'target_std', 'eps', 'rows'})
                recipe['prior_regularizer'] = {key: value for key, value in loss.items() if key != 'type'}
                if self.raw.get('defaults') != 'particlegan':
                    recipe['prior_regularizer'].setdefault('weight', 1.0)

            else:
                raise ModelFileError(f'{location}.type must be adversarial, objective, reconstruction or prior')

    def train(self, spec, recipe):
        _table(spec, 'train', OPTIMIZER_FIELDS | TRAINING_FIELDS | RUN_CADENCE_FIELDS)
        optimizer = {key: deepcopy(value) for key, value in spec.items() if key in OPTIMIZER_FIELDS}
        training = {key: deepcopy(value) for key, value in spec.items() if key in TRAINING_FIELDS}
        run_cadence = {key: deepcopy(value) for key, value in spec.items() if key in RUN_CADENCE_FIELDS}
        if optimizer:
            recipe['optimizer'] = optimizer
        if training:
            recipe.setdefault('training', {}).update(training)
            if 'every_steps' in recipe['training']:
                recipe.setdefault('metrics', {})['every_steps'] = recipe['training'].pop('every_steps')
        if run_cadence:
            recipe.setdefault('run', {}).update(run_cadence)

    def sampling(self, spec):
        _table(spec, 'sampling', {'count', 'seed', 'output', 'particle_ids', 'views', 'comparison'})
        result = {key: spec[key] for key in ('count', 'seed') if key in spec}
        if 'output' in spec:
            result['generated'] = self.bind(spec['output'], 'sampling.output')
        if 'particle_ids' in spec:
            result['particle_ids'] = self.bind(spec['particle_ids'], 'sampling.particle_ids')
        if 'views' in spec:
            result['views'] = {name: self.bind(path, f'sampling.views.{name}')
                               for name, path in _table(spec['views'], 'sampling.views').items()}
        if 'comparison' in spec:
            result['comparison'] = [{'label': column.get('label'), 'binding': self.bind(column.get('binding'), f'sampling.comparison[{i}].binding')}
                                    for i, column in enumerate(spec['comparison'])]
        return result

    def samplers(self, spec):
        result = {}
        for name, sampler in _table(spec, 'samplers').items():
            location = f'samplers.{name}'
            _table(sampler, location, {'fn', 'factory', 'args', 'inputs', 'count', 'seed'})
            fn = sampler.get('fn') or sampler.get('factory')
            if callable(fn):
                from .api import import_path
                fn = import_path(fn, f'{location}.fn')
            if not isinstance(fn, str):
                raise ModelFileError(f'{location}.fn must name a function by module:object')
            entry = {'fn': fn, 'args': deepcopy(sampler.get('args', {}))}
            if 'inputs' in sampler:
                entry['inputs'] = self.bind_all(sampler['inputs'], f'{location}.inputs')
            if 'count' in sampler:
                entry['count'] = sampler['count']
            if 'seed' in sampler:
                entry['seed'] = sampler['seed']
            result[name] = entry
        return result

    def metrics(self, spec):
        spec = _table(deepcopy(spec), 'metrics')
        if 'every' in spec:
            every_val = spec.pop('every')
            spec.setdefault('every_steps', every_val)
        raw_custom = []
        for name, metric in _table(spec.pop('custom', {}), 'metrics.custom').items():
            raw_custom.append((name, metric, f'metrics.custom.{name}'))
        for name in list(spec.keys()):
            if name not in ('preset', 'disable', 'every_steps', 'every', 'overrides') and isinstance(spec[name], dict):
                raw_custom.append((name, spec.pop(name), f'metrics.{name}'))
        _table(spec, 'metrics', {'preset', 'disable', 'every_steps', 'every', 'overrides'})
        custom = {}
        for name, metric, location in raw_custom:
            metric = deepcopy(_table(metric, location, {'fn', 'factory', 'args', 'inputs', 'every_steps', 'every', 'timeout', 'on_error',
                                      'label', 'unit', 'direction', 'description'}))
            if 'every' in metric:
                ev = metric.pop('every')
                metric.setdefault('every_steps', ev)
            inputs = {key: value if '.' in value else 'update.' + value
                      for key, value in _table(metric.get('inputs'), f'{location}.inputs').items()}
            lowered = {'mode': 'scalar', 'inputs': inputs}
            lowered.update({key: metric[key] for key in ('every_steps', 'timeout', 'on_error') if key in metric})
            lowered.update(self._plugin(metric, location, SCALAR_FUNCTION))
            custom[name] = lowered
        return spec, custom

    def _plugin(self, spec, location, adapter):
        kind = _one_of(spec, ('fn', 'factory'), location)
        presentation = {key: spec[key] for key in ('label', 'unit', 'direction', 'description', 'reduce', 'kind') if key in spec}
        if kind == 'factory':
            if presentation:
                raise ModelFileError(f'{location}: {", ".join(presentation)} belong to fn metrics; a factory describes itself')
            return {'factory': spec['factory'], 'args': deepcopy(spec.get('args', {}))}
        args = {'fn': spec['fn'], **presentation}
        if spec.get('args'):
            args['args'] = deepcopy(spec['args'])
        return {'factory': adapter, 'args': args}

    def evaluations(self, spec):
        raw_spec = _table(deepcopy(spec), 'evaluations')
        shared_defaults = {}
        subtables = {}
        for k, v in raw_spec.items():
            if k in _EVAL_FIELDS and (not isinstance(v, dict) or k in ('data', 'args', 'inputs')):
                shared_defaults[k] = v
            else:
                subtables[k] = v
        custom = {}
        for name, evaluation in subtables.items():
            location = f'evaluations.{name}'
            _table(evaluation, location, _EVAL_FIELDS)
            merged = deepcopy(shared_defaults)
            merged.update(deepcopy(evaluation))
            evaluation = merged
            if 'every_steps' in evaluation:
                ev = evaluation.pop('every_steps')
                evaluation.setdefault('every', ev)
            if 'target' in evaluation:
                tgt = evaluation.pop('target')
                evaluation.setdefault('output', tgt)
            inputs = evaluation.get('inputs', {'generated': 'generated', 'reference': 'reference'})
            inputs = {key: value if value.startswith('evaluation.') else 'evaluation.' + value
                      for key, value in _table(inputs, f'{location}.inputs').items()}
            if 'data' not in evaluation:
                raise ModelFileError(f'{location}.data is required: evaluations never fall back to training data')
            protocol = {'data': self.data(evaluation['data'], f'{location}.data'),
                        'sample_count': evaluation.get('samples'), 'batch_size': evaluation.get('batch_size'),
                        'seed': evaluation.get('seed')}
            if any(value is None for value in protocol.values()):
                raise ModelFileError(f'{location} requires samples, batch_size and seed')
            if 'device' in evaluation:
                protocol['device'] = evaluation['device']
            if 'output' in evaluation:
                output = evaluation['output']
                protocol['generated'] = 'generated' if output == 'generated' else self.bind(output, f'{location}.output')
            lowered = {'mode': 'snapshot', 'inputs': inputs, 'evaluation': protocol}
            if 'every' in evaluation:
                lowered.update(trigger='interval', every_steps=evaluation['every'])
            else:
                lowered['trigger'] = 'manual'
            lowered.update({key: evaluation[key] for key in ('timeout', 'on_error', 'on_busy') if key in evaluation})
            lowered.update(self._plugin(evaluation, location, SNAPSHOT_FUNCTION))
            custom[name] = lowered
        return custom

    def _validate_loss_graph(self, recipe, losses):
        components = recipe['components']
        reach = {name: set() for name in components}
        critics = {self.names[name] for name, s in self.networks.items() if s.get('role') == 'critic'}

        for term in recipe.get('objectives', []):
            detached = set(term.get('detach', ['target'] if 'target' in term['inputs'] else []))
            seen = set()
            for key, path in term['inputs'].items():
                if key not in detached:
                    upstream(path, components, seen=seen)
            owner = term.get('owner')
            if owner is not None:
                lowered_owner = self.names.get(owner, owner)
                if lowered_owner not in seen:
                    raise ModelFileError(
                        f"networks.{owner}: loss {term.get('id', term['factory'])!r} reads "
                        f"{sorted(term['inputs'].values())}, which do not depend on {owner}"
                    )
            else:
                trainable_reached = {name for name in seen if components[name].get('trainable', True) and name not in critics}
                if not trainable_reached:
                    raise ModelFileError(
                        f"Objective loss {term['id']!r} reads {sorted(term['inputs'].values())}, "
                        f"which do not depend on any trainable generator or encoder"
                    )
            for name in seen:
                reach[name].add(f"loss {term['id']}")

        for term in recipe.get('adversarial_terms', []):
            fake = term['fake']
            for name in upstream(fake, components):
                reach[name].add(f"fooling {term['component']}")
        if 'adversarial' in recipe:
            disc_inputs = components.get('discriminator', {}).get('inputs', {})
            fake_path = recipe['adversarial'].get('fake', 'generated')
            if any(p == 'candidate' or p.startswith('candidate.') or producer(p) == 'generator' for p in disc_inputs.values()):
                for name in upstream(fake_path, components):
                    reach[name].add('fooling discriminator')

        for orig_name, lowered_name in self.names.items():
            role = self.networks[orig_name].get('role')
            if role == 'generator' and 'reuse' not in self.networks[orig_name] and not reach.get(lowered_name):
                raise ModelFileError(
                    f"Network {orig_name!r} has role = 'generator' but no adversarial or objective loss trains it"
                )

    def checkpoints(self, table):
        _table(table, 'checkpoints', CHECKPOINT_FIELDS)
        if 'keep_last' in table:
            v = table['keep_last']
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ModelFileError('checkpoints.keep_last must be a non-negative integer')
        if 'keep_every' in table:
            v = table['keep_every']
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ModelFileError('checkpoints.keep_every must be a non-negative integer')
        if 'export_bundle' in table:
            v = table['export_bundle']
            if not isinstance(v, str):
                raise ModelFileError('checkpoints.export_bundle must be a string')
        if 'keep_best' in table:
            v = table['keep_best']
            if v is False or v is None:
                pass
            elif isinstance(v, int) and not isinstance(v, bool):
                if v < 0:
                    raise ModelFileError('checkpoints.keep_best count must be a non-negative integer')
            elif isinstance(v, dict):
                _table(v, 'checkpoints.keep_best', {'metric', 'mode', 'count'})
                if 'metric' in v and not isinstance(v['metric'], str):
                    raise ModelFileError('checkpoints.keep_best.metric must be a string')
                if 'mode' in v and v['mode'] not in ('min', 'max'):
                    raise ModelFileError("checkpoints.keep_best.mode must be 'min' or 'max'")
                if 'count' in v:
                    cnt = v['count']
                    if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 0:
                        raise ModelFileError('checkpoints.keep_best.count must be a non-negative integer')
            else:
                raise ModelFileError('checkpoints.keep_best must be an integer count or table with {metric, mode, count}')

    def run(self):
        raw = self.raw
        _table(raw, 'model file', TOP_LEVEL)
        if raw.get('format', FORMAT) != FORMAT:
            raise ModelFileError(f'format must be {FORMAT!r}')
        losses = raw.get('losses')
        if not isinstance(losses, list) or not all(isinstance(loss, dict) for loss in losses):
            raise ModelFileError('losses must be an array of tables ([[losses]])')
        self._plan_names(losses)
        recipe = {'schema_version': 1, 'name': raw.get('name', 'custom/model')}
        if 'defaults' in raw:
            recipe['defaults'] = raw['defaults']
        if 'recipe' in raw:
            recipe['recipe'] = deepcopy(_table(raw['recipe'], 'recipe'))
        # [dataset] is an alias for [data]; read it without writing into the caller's spec.
        data = raw['data'] if 'data' in raw else raw.get('dataset')
        if data is None:
            raise ModelFileError('data is required (builtin, factory or dataset)')
        recipe['data'] = self.data(data, 'data')
        if isinstance(data, dict) and 'batch_size' in data:
            recipe.setdefault('training', {}).setdefault('batch_size', data['batch_size'])
        if 'prior' in raw:
            recipe['prior'] = self.prior(raw['prior'])
        recipe['components'] = {self.names[name]: self.component(name, spec) for name, spec in self.networks.items()}
        self.losses(losses, recipe)
        if 'penalty' in raw:
            penalty = _table(raw['penalty'], 'penalty')
            if 'coeff' in penalty:
                raise ModelFileError('penalty.coeff: set the critic regularization coefficient on each adversarial loss (penalty = ...)')
            _table(penalty, 'penalty', PENALTY_FIELDS)
            recipe.setdefault('gradient_penalty', {}).update(deepcopy(penalty))
        if 'train' in raw:
            self.train(raw['train'], recipe)
        if 'sampling' in raw:
            recipe['sampling'] = self.sampling(raw['sampling'])
        if 'samplers' in raw:
            recipe['samplers'] = self.samplers(raw['samplers'])
        metrics, custom = self.metrics(raw.get('metrics', {}))
        custom.update(self.evaluations(raw.get('evaluations', {})))
        if custom:
            metrics['custom'] = custom
        if metrics:
            recipe['metrics'] = metrics
        if 'every_steps' not in recipe.get('metrics', {}):
            recipe.setdefault('metrics', {})['every_steps'] = 1
        if 'checkpoints' in raw:
            self.checkpoints(raw['checkpoints'])
            recipe['checkpoints'] = normalize_checkpoint_policy(raw['checkpoints'])
        if 'run' in raw:
            _table(raw['run'], 'run', RUN_FIELDS)
            recipe.setdefault('run', {}).update(deepcopy(raw['run']))
        recipe['format'] = FORMAT

        recipe['name_map'] = dict(self.names)
        self._validate_loss_graph(recipe, losses)
        return recipe


def lower(raw, base=None):
    """Lower a model file mapping to an engine recipe (``resolve_config`` input).

    ``base`` is the directory relative ``hndl`` paths are read from. HNDL text
    is read into the recipe, as the engine does for ``file``; nothing else is
    read and no Python is imported.
    """
    if not isinstance(raw, dict) or not is_model_file(raw):
        res = deepcopy(raw) if isinstance(raw, dict) else raw
        if isinstance(res, dict) and 'checkpoints' in res:
            res['checkpoints'] = normalize_checkpoint_policy(res['checkpoints'])
        return res
    return _Lowering(raw, base).run()


def model_warnings(raw):
    """Advice that is not an error: explicit loss lists make omissions visible."""
    warnings = []
    losses = raw.get('losses', [])
    prior = raw.get('prior', {})
    if prior.get('kind', 'particles') != 'gaussian' and prior.get('learnable', True) and not any(
            loss.get('type') == 'prior' for loss in losses):
        warnings.append('No type = "prior" loss is listed, so the learnable prior table has no spread regularizer (weight 0).')
    return warnings


def _deep_merge(base, override):
    """Deep merge dictionaries; scalars and lists are replaced by override."""
    if isinstance(base, dict) and isinstance(override, dict):
        result = deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = deepcopy(v)
        return result
    return deepcopy(override)


def _anchor_relative_paths(spec, base_dir, _in_path_container=False):
    """Anchor relative file paths (not starting with '${') to base_dir."""
    if isinstance(spec, dict):
        res = {}
        for k, v in spec.items():
            if isinstance(v, str):
                is_path_field = (
                    k in _PATH_KEYS
                    or _in_path_container
                    or v.startswith("./")
                    or v.startswith("../")
                )
                if is_path_field and not v.startswith("${") and not Path(v).is_absolute() and (base_dir / v).exists():
                    res[k] = str((base_dir / v).resolve())
                else:
                    res[k] = v
            else:
                res[k] = _anchor_relative_paths(v, base_dir, _in_path_container=(k == 'network_files'))
        return res
    elif isinstance(spec, list):
        return [_anchor_relative_paths(item, base_dir, _in_path_container=_in_path_container) for item in spec]
    return deepcopy(spec)


def resolve_inheritance(
    model_dict: dict,
    config_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    _chain: tuple = (),
) -> dict:
    """Recursively resolve 'extends' inheritance for model files.

    Parent configurations are merged recursively (tables deep-merged by name,
    scalars and arrays replaced). Relative paths are anchored to the declaring file.
    Circular inheritance chains raise ValueError.
    """
    model_dict = deepcopy(model_dict)
    curr_canon = str(Path(config_path).resolve()) if config_path is not None else None
    if curr_canon is not None:
        if curr_canon in _chain:
            chain_str = " -> ".join(list(_chain) + [curr_canon])
            raise ValueError(f"Circular inheritance detected: {chain_str}")
        new_chain = _chain + (curr_canon,)
    else:
        new_chain = _chain

    if "extends" not in model_dict:
        if "name" not in model_dict and config_path is not None:
            model_dict["name"] = Path(config_path).stem
        return model_dict

    parent_ref = model_dict.pop("extends")
    parent_path = None
    ref_as_path = Path(parent_ref)

    if ref_as_path.is_absolute() and ref_as_path.is_file():
        parent_path = ref_as_path.resolve()
    else:
        # Search candidates in order:
        # 1. relative to config_path.parent
        if config_path is not None:
            cand = (Path(config_path).resolve().parent / parent_ref).resolve()
            if cand.is_file():
                parent_path = cand
        # 2. relative to workspace_root
        if parent_path is None and workspace_root is not None:
            cand = (Path(workspace_root).resolve() / parent_ref).resolve()
            if cand.is_file():
                parent_path = cand
        # 3. relative to current working directory
        if parent_path is None:
            cand = (Path.cwd() / parent_ref).resolve()
            if cand.is_file():
                parent_path = cand
        # 4. packaged model in hypergan/models
        if parent_path is None:
            packaged_dir = Path(__file__).resolve().parent / "models"
            cand = (packaged_dir / parent_ref).resolve()
            if cand.is_file():
                parent_path = cand
            elif not parent_ref.endswith(".toml"):
                cand = (packaged_dir / f"{parent_ref}.toml").resolve()
                if cand.is_file():
                    parent_path = cand

    if parent_path is None:
        raise FileNotFoundError(
            f"Inherited configuration '{parent_ref}' not found (searched relative to config directory, "
            f"workspace root, current directory, and packaged models)"
        )

    parent_canon = str(parent_path)
    if parent_canon in new_chain:
        chain_str = " -> ".join(list(new_chain) + [parent_canon])
        raise ValueError(f"Circular inheritance detected: {chain_str}")

    with parent_path.open("rb") as f:
        parent_raw = tomllib.load(f)

    if not isinstance(parent_raw, dict):
        raise ValueError(f"Inherited parent {parent_path} must be a TOML table")

    # Recursively resolve any extends in parent
    parent_resolved = resolve_inheritance(
        parent_raw,
        config_path=parent_path,
        workspace_root=workspace_root,
        _chain=new_chain,
    )

    # Anchor any relative paths in parent to parent_path.parent
    parent_anchored = _anchor_relative_paths(parent_resolved, parent_path.parent)

    # Deep merge parent (base) and model_dict (override)
    merged = _deep_merge(parent_anchored, model_dict)
    merged.pop("extends", None)

    if "name" not in model_dict:
        if config_path is not None:
            merged["name"] = Path(config_path).stem
        elif "name" in parent_resolved:
            merged["name"] = parent_resolved["name"]
        else:
            merged["name"] = "custom/model"

    return merged


def collect_source_files(config_path: Path | str, workspace_root: Path | str | None = None) -> list[Path]:
    """Collect the chain of source configuration files for provenance snapshotting."""
    sources = []
    seen = set()

    def _walk(path_val: Path | str):
        p = Path(path_val).resolve()
        if p in seen or not p.is_file():
            return
        seen.add(p)
        try:
            with p.open('rb') as f:
                data = tomllib.load(f)
            if isinstance(data, dict) and 'extends' in data:
                parent_ref = data['extends']
                cand = (p.parent / parent_ref).resolve()
                if not cand.is_file() and workspace_root is not None:
                    cand = (Path(workspace_root).resolve() / parent_ref).resolve()
                if not cand.is_file():
                    cand = (Path.cwd() / parent_ref).resolve()
                if not cand.is_file():
                    cand = (Path(__file__).resolve().parent / "models" / parent_ref).resolve()
                    if not cand.is_file() and not parent_ref.endswith(".toml"):
                        cand = (Path(__file__).resolve().parent / "models" / f"{parent_ref}.toml").resolve()
                if cand.is_file():
                    _walk(cand)
        except Exception:
            pass
        sources.append(p)

    ws_file = None
    if workspace_root is not None:
        cand_ws = Path(workspace_root) / "hypergan.toml"
        if cand_ws.is_file():
            ws_file = cand_ws.resolve()
    else:
        from .workspace import find_workspace
        ws_file = find_workspace(config_path)

    if ws_file and ws_file.is_file():
        sources.append(ws_file)
        seen.add(ws_file)

    _walk(config_path)
    return sources
