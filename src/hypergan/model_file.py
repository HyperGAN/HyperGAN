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
    type = "adversarial"        # RpGAN logistic, K3P critic penalty
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

from .network_config import read_source

FORMAT = 'hypergan-model/1'
TOP_LEVEL = {'format', 'name', 'defaults', 'data', 'prior', 'networks', 'losses', 'penalty', 'train',
             'sampling', 'samplers', 'metrics', 'evaluations'}
ROLES = ('generator', 'critic', 'encoder', 'auxiliary')
RESERVED = {'latent', 'batch', 'generated', 'candidate', 'prior', 'components', 'update', 'evaluation',
            'reference'}
NETWORK_FIELDS = {'role', 'hndl', 'source', 'module', 'reuse', 'args', 'inputs', 'input_shape', 'output_shape',
                  'trainable', 'freeze_parameters'}
OPTIMIZER_FIELDS = {'lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas', 'implementation',
                    'd_guard_ratio', 'd_guard_min_steps', 'latent_damping_max_rate'}
TRAINING_FIELDS = {'steps', 'batch_size', 'seed', 'device', 'ema', 'lr_anneal_start', 'lr_floor',
                   'network_lr_floor', 'network_lr_horizon_cap', 'input_noise_std', 'input_noise_anneal_end',
                   'output_noise_std', 'output_noise_warmup', 'phase_draws', 'data_rng_device',
                   'data_seed_offset', 'prior_seed_offset', 'backend'}
PENALTY_FIELDS = {'kappa', 'lazy_k', 'anchor_weight', 'anchor_decay'}
PRIOR_FIELDS = {'kind', 'initialization_device', 'initialization_seed', 'fixed_sigma'}
BUILTIN_DATA = ('gaussian_grid', 'paired_linear', 'image_folder')
ITEM_DATA = 'hypergan.item_data:ItemData'
SCALAR_FUNCTION = 'hypergan.plugin_functions:ScalarFunction'
SNAPSHOT_FUNCTION = 'hypergan.plugin_functions:SnapshotFunction'
UPDATE_SCALARS = ('d_loss', 'g_loss', 'd_adversarial', 'g_adversarial', 'd_adversarial_weighted',
                  'g_adversarial_weighted', 'gradient_penalty', 'prior_loss', 'lr_scale', 'step', 'step_seconds')
_ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}')


class ModelFileError(ValueError):
    """A model file field is missing, unknown or inconsistent; the message names it."""


def is_model_file(raw):
    """Model files have ``networks``; engine recipes have ``components``."""
    return isinstance(raw, dict) and 'networks' in raw and 'components' not in raw


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
            role = spec.get('role')
            if role not in ROLES:
                raise ModelFileError(f'networks.{name}.role must be one of {", ".join(ROLES)}')
            roles.setdefault(role, []).append(name)
        generators = roles.get('generator', [])
        if len(generators) != 1:
            raise ModelFileError(f'Exactly one network may have role = "generator" today (found {len(generators)}); '
                                 'more generators need an engine with several generator outputs')
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
        self.names[generators[0]] = 'generator'
        self.names[primary] = 'discriminator'
        for name, lowered in self.names.items():
            if lowered in ('generator', 'discriminator') and name not in (generators[0], primary):
                raise ModelFileError(f'networks.{name}: the names generator/discriminator are reserved for the '
                                     'generator and the first adversarial loss critic; rename this network')
        self.generator, self.primary = generators[0], primary

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
        if 'reuse' in spec:
            target = spec['reuse']
            if target not in self.networks:
                raise ModelFileError(f'{location}.reuse must name another network')
            result = {'reuse': self.names[target], 'inputs': self.bind_all(spec.get('inputs'), f'{location}.inputs')}
            if 'freeze_parameters' in spec:
                result['freeze_parameters'] = spec['freeze_parameters']
            return result
        kind = _one_of(spec, ('hndl', 'source', 'module'), location)
        args = deepcopy(_table(spec.get('args', {}), f'{location}.args'))
        if kind == 'module':
            if 'input_shape' in spec or 'output_shape' in spec:
                raise ModelFileError(f'{location}: input_shape/output_shape describe HNDL networks; pass module settings in args')
            factory = spec['module']
        else:
            factory = 'hndl'
            if kind == 'hndl':
                file = Path(spec['hndl'])
                if not file.is_absolute():
                    if self.base is None:
                        raise ModelFileError(f'{location}.hndl is relative but the model has no file location; use an absolute path or source')
                    file = self.base / file
                try:
                    args['source'] = read_source(file)
                except OSError as exc:
                    raise ModelFileError(f'{location}.hndl: cannot read {file}: {exc}') from exc
            else:
                args['source'] = spec['source']
            for key in ('input_shape', 'output_shape'):
                if key not in spec:
                    raise ModelFileError(f'{location}.{key} is required for an HNDL network')
                args[key] = deepcopy(spec[key])
        result = {'factory': factory, 'args': args, 'inputs': self.bind_all(spec.get('inputs'), f'{location}.inputs')}
        if 'trainable' in spec:
            result['trainable'] = spec['trainable']
        return result

    def data(self, spec, location):
        _table(spec, location, {'builtin', 'factory', 'dataset', 'args', 'shuffle', 'workers'})
        kind = _one_of(spec, ('builtin', 'factory', 'dataset'), location)
        args = deepcopy(_table(spec.get('args', {}), f'{location}.args'))
        if kind != 'dataset' and ({'shuffle', 'workers'} & set(spec)):
            raise ModelFileError(f'{location}: shuffle/workers apply to item-level datasets (dataset = "module:Class")')
        if kind == 'builtin':
            if spec['builtin'] not in BUILTIN_DATA:
                raise ModelFileError(f'{location}.builtin must be one of {", ".join(BUILTIN_DATA)}')
            return {'factory': spec['builtin'], 'args': args}
        if kind == 'factory':
            return {'factory': spec['factory'], 'args': args}
        item = {'dataset': spec['dataset'], 'args': args}
        for key in ('shuffle', 'workers'):
            if key in spec:
                item[key] = spec[key]
        return {'factory': ITEM_DATA, 'args': item}

    def prior(self, spec):
        spec = _table(deepcopy(spec), 'prior')
        result = {key: spec.pop(key) for key in PRIOR_FIELDS if key in spec}
        result['args'] = spec
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
                _table(loss, location, {'type', 'id', 'critic', 'weight', 'penalty', 'real', 'fake', 'inputs'})
                penalty = loss.get('penalty', 1.0)
                if type(penalty) not in (int, float) or penalty < 0:
                    raise ModelFileError(f'{location}.penalty is the K3P coefficient: a nonnegative number (0 disables it)')
                if loss['critic'] == self.primary and 'adversarial' not in recipe:
                    real, fake = loss.get('real', 'batch.real'), loss.get('fake', 'generated')
                    if (real, fake) != ('batch.real', 'generated') or 'inputs' in loss:
                        raise ModelFileError(f'{location}: the first adversarial loss compares batch.real with generated '
                                             'through its critic\'s own inputs (engine limit); put other comparisons in later adversarial losses')
                    if 'id' in loss:
                        raise ModelFileError(f'{location}: the first adversarial loss publishes as loss/d_adversarial and loss/g_adversarial; remove id')
                    recipe['adversarial'] = {'weight': loss.get('weight', 1.0)}
                    recipe.setdefault('gradient_penalty', {})['coeff'] = penalty
                    continue
                term = {'id': loss.get('id', loss['critic']), 'component': self.names[loss['critic']],
                        'weight': loss.get('weight', 1.0), 'penalty': penalty > 0,
                        'real': self.bind(loss.get('real', 'batch.real'), f'{location}.real'),
                        'fake': self.bind(loss.get('fake', 'generated'), f'{location}.fake')}
                if penalty > 0:
                    term['penalty_coeff'] = penalty
                if 'inputs' in loss:
                    term['inputs'] = self.bind_all(loss['inputs'], f'{location}.inputs')
                if term['id'] in seen_ids:
                    raise ModelFileError(f'{location}: loss id {term["id"]!r} is repeated; give each loss an id')
                seen_ids.add(term['id'])
                recipe.setdefault('adversarial_terms', []).append(term)
            elif kind in ('objective', 'reconstruction'):
                _table(loss, location, {'type', 'id', 'fn', 'args', 'inputs', 'input', 'target', 'weight', 'detach'})
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
                term = {'id': loss['id'], 'factory': loss.get('fn', 'mse'), 'inputs': inputs,
                        'weight': loss.get('weight', 1.0), 'args': deepcopy(loss.get('args', {}))}
                if 'detach' in loss:
                    term['detach'] = list(loss['detach'])
                recipe['objectives'].append(term)
            elif kind == 'prior':
                _table(loss, location, {'type', 'weight', 'target_std', 'eps', 'rows'})
                recipe['prior_regularizer'] = {key: value for key, value in loss.items() if key != 'type'}
                recipe['prior_regularizer'].setdefault('weight', 1.0)
            else:
                raise ModelFileError(f'{location}.type must be adversarial, objective, reconstruction or prior')

    def train(self, spec, recipe):
        _table(spec, 'train', OPTIMIZER_FIELDS | TRAINING_FIELDS)
        optimizer = {key: deepcopy(value) for key, value in spec.items() if key in OPTIMIZER_FIELDS}
        training = {key: deepcopy(value) for key, value in spec.items() if key in TRAINING_FIELDS}
        if optimizer:
            recipe['optimizer'] = optimizer
        if training:
            recipe['training'] = training

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
            _table(sampler, location, {'fn', 'args'})
            if not isinstance(sampler.get('fn'), str):
                raise ModelFileError(f'{location}.fn must name a function by module:object')
            result[name] = {'fn': sampler['fn'], 'args': deepcopy(sampler.get('args', {}))}
        return result

    def metrics(self, spec):
        spec = _table(deepcopy(spec), 'metrics', {'preset', 'disable', 'every_steps', 'overrides', 'custom'})
        custom = {}
        for name, metric in _table(spec.pop('custom', {}), 'metrics.custom').items():
            location = f'metrics.custom.{name}'
            _table(metric, location, {'fn', 'factory', 'args', 'inputs', 'every_steps', 'timeout', 'on_error',
                                      'label', 'unit', 'direction', 'description'})
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
        custom = {}
        for name, evaluation in _table(spec, 'evaluations').items():
            location = f'evaluations.{name}'
            _table(evaluation, location, {'fn', 'factory', 'args', 'inputs', 'every', 'device', 'samples', 'batch_size',
                                          'seed', 'data', 'output', 'timeout', 'on_error', 'label', 'unit', 'direction',
                                          'description', 'reduce', 'kind'})
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
            lowered.update({key: evaluation[key] for key in ('timeout', 'on_error') if key in evaluation})
            lowered.update(self._plugin(evaluation, location, SNAPSHOT_FUNCTION))
            custom[name] = lowered
        return custom

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
        if 'data' not in raw:
            raise ModelFileError('data is required (builtin, factory or dataset)')
        recipe['data'] = self.data(raw['data'], 'data')
        if 'prior' in raw:
            recipe['prior'] = self.prior(raw['prior'])
        recipe['components'] = {self.names[name]: self.component(name, spec) for name, spec in self.networks.items()}
        self.losses(losses, recipe)
        if 'penalty' in raw:
            penalty = _table(raw['penalty'], 'penalty')
            if 'coeff' in penalty:
                raise ModelFileError('penalty.coeff: set the K3P coefficient on each adversarial loss (penalty = ...)')
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
        return recipe


def lower(raw, base=None):
    """Lower a model file mapping to an engine recipe (``resolve_config`` input).

    ``base`` is the directory relative ``hndl`` paths are read from. HNDL text
    is read into the recipe, as the engine does for ``file``; nothing else is
    read and no Python is imported.
    """
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
