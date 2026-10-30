"""Torch-free loading of the architecture text recorded in run configurations."""
from pathlib import Path
from string import Template
import math

NETWORK_DIRECTORY = Path(__file__).with_name('networks')
MAX_SOURCE_BYTES = 1_048_576


def read_source(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('HNDL source exceeds 1 MiB')
    # Match TOML multiline-string newline normalization on every platform.
    return data.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


def packaged_source(name):
    path = Path(name)
    if path.name != str(path) or path.suffix != '.hndl':
        raise ValueError('Packaged network must be a .hndl filename')
    return read_source(NETWORK_DIRECTORY / path)


class SourceFragment(str):
    """Trusted application-selected HNDL fragment, never accepted from config parameters."""


def _literal(value):
    if isinstance(value, SourceFragment):
        return str(value)
    if value is None or type(value) in (str, bool, int):
        return repr(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_literal(item) for item in value) + ']'
    raise ValueError('HNDL template parameters must be finite scalar literals or lists')


def render_source(source, parameters=None):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('HNDL source must be nonempty text')
    if len(source.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise ValueError('HNDL source exceeds 1 MiB')
    try:
        return Template(source).substitute({key: _literal(value) for key, value in (parameters or {}).items()})
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Invalid HNDL template parameter: {exc}') from exc


def validate_pretrained_providers(providers):
    """Validate trusted provider options without importing model libraries."""
    if not isinstance(providers, dict):
        raise ValueError('pretrained_providers must be a table')
    for name, options in providers.items():
        if name != 'dinov3_vits16':
            raise ValueError(f'Unknown configurable pretrained provider: {name}')
        if not isinstance(options, dict) or set(options) != {'source_path', 'source_commit'}:
            raise ValueError('dinov3_vits16 requires source_path and source_commit')
        if not isinstance(options['source_path'], str) or not options['source_path'].strip():
            raise ValueError('DINOv3 source_path must be nonempty text')
        commit = options['source_commit']
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('DINOv3 source_commit must be a full lowercase Git SHA')


def validate_network_args(args, location=None, registry=None):
    """Statically parse and validate HNDL network architecture text via hndl.resolve().

    If location is a non-empty string, raises ValueError on error (for backward compatibility).
    If location is None, returns a list of error diagnostic strings (empty if valid).
    """
    if location is not None and not isinstance(location, str):
        registry = location
        location = None
    loc_str = location or "network_args"
    errors = []

    def _fail(msg: str):
        if location:
            raise ValueError(msg)
        errors.append(msg)

    if not isinstance(args, dict):
        _fail(f"{loc_str} must be a table")
        return errors

    allowed = {'source', 'file', 'input_shape', 'output_shape', 'parameters', 'input_dtype', 'pretrained_providers'}
    unknown = set(args) - allowed
    if unknown:
        _fail(f"{loc_str}: unknown HNDL arguments {sorted(unknown)}")
        return errors

    if ('source' in args) == ('file' in args):
        _fail(f"{loc_str}: specify exactly one of source or file")
        return errors

    for key in ('input_shape', 'output_shape'):
        contract = args.get(key)
        shapes = contract if isinstance(contract, dict) else {key: contract}
        if not shapes or any(not isinstance(name, str) or not name.isidentifier() for name in shapes):
            _fail(f"{loc_str}.{key} must declare valid port names")
            return errors
        for name, shape in shapes.items():
            if (not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3, 4)
                    or shape[0] != 'B' or any(type(size) is not int or size < 1 for size in shape[1:])):
                _fail(f"{loc_str}.{key}.{name} must be ['B', positive dimensions...]")
                return errors

    if not isinstance(args.get('parameters', {}), dict):
        _fail(f"{loc_str}.parameters must be a table")
        return errors

    try:
        validate_pretrained_providers(args.get('pretrained_providers', {}))
    except ValueError as exc:
        _fail(str(exc))
        return errors

    dtype = args.get('input_dtype')
    if dtype is not None:
        ports = args['input_shape'] if isinstance(args['input_shape'], dict) else {'x': None}
        dtypes = dtype if isinstance(dtype, dict) else {name: dtype for name in ports}
        if (not dtypes or set(dtypes) - set(ports)
                or any(value not in ('float32', 'int32', 'int64', 'bool') for value in dtypes.values())):
            _fail(f"{loc_str}.input_dtype must declare supported dtypes for input ports")
            return errors

    source_text = None
    if 'source' in args:
        try:
            source_text = render_source(args['source'], args.get('parameters'))
        except Exception as exc:
            _fail(f"{loc_str}: {exc}")
            return errors
    elif 'file' in args:
        f = args['file']
        if not isinstance(f, str) or not f or Path(f).suffix != '.hndl':
            _fail(f"{loc_str}.file must name a .hndl file")
            return errors
        if location is None:
            try:
                if Path(f).name == str(f) and (NETWORK_DIRECTORY / f).is_file():
                    source_text = packaged_source(f)
                elif Path(f).is_file():
                    source_text = read_source(Path(f))
            except Exception as exc:
                _fail(f"{loc_str}.file: {exc}")
                return errors

    if location is None and source_text is not None:
        try:
            from hndl import Registry, HNDLError, resolve as hndl_resolve
            from .hndl_augmentation import register_augmentation
            from .pretrained_providers import register_providers
            active_reg = register_augmentation(Registry.builtins() if registry is None else registry)
            active_reg = register_providers(active_reg, args.get('pretrained_providers'))
            hndl_resolve(
                source_text,
                input_shape=args['input_shape'],
                output_shape=args['output_shape'],
                registry=active_reg,
                input_dtype=args.get('input_dtype'),
            )
        except HNDLError as exc:
            _fail(f"{loc_str}: {exc}")
            return errors
        except Exception as exc:
            _fail(f"{loc_str}: HNDL validation error: {exc}")
            return errors

    return errors


# Templates are loaded without importing their numerical adapters. Resolved
# configs include their exact text, including reusable attention fragments.
_IMAGE_CRITIC = ('image_pixel', 'image_pixel_block', 'image_attention', 'image_feature_head',
                 'image_resnet_stage1', 'image_resnet_stage2', 'image_resnet_stage3', 'image_critic_score')
COMPONENT_TEMPLATES = {
    'hypergan.image_components:CIFARGenerator': ('image_generator', 'image_attention'),
    'hypergan.image_components:CIFARRoutingEncoder': ('image_encoder',),
    'hypergan.image_components:CIFARDiscriminator': _IMAGE_CRITIC,
    'hypergan.autoencoder_components:ParticleAEEncoder256': ('colorization_encoder_features', 'colorization_query', 'autoencoder_offset'),
    'hypergan.colorization_components:ColorizationGenerator': ('colorization_generator16', 'colorization_generator32', 'image_attention'),
    'hypergan.colorization_components:GrayscaleRoutingEncoder': ('colorization_encoder_features', 'colorization_query'),
    'hypergan.colorization_components:DINOv3Discriminator': ('colorization_dinov3_patch_tokens', 'colorization_pixel_head', 'colorization_feature_project', 'colorization_linear_head', 'colorization_joint_score', 'image_attention'),
    'hypergan.colorization_components:DINOv3ProjectedDiscriminator': ('colorization_dinov3_patch_tokens', 'colorization_random_project', 'colorization_linear_head', 'colorization_conv_head', 'colorization_pixel_features', 'colorization_feature_concat', 'image_attention'),
    'hypergan.colorization_components:DINOv3MultiScaleDiscriminator': ('colorization_dinov3_multidepth', 'colorization_multidepth_projection', 'colorization_scale_head32', 'colorization_scale_head16', 'colorization_scale_head8', 'colorization_scale_head4', 'colorization_multiscale_score', 'image_attention'),
    'hypergan.colorization_components:DCGANDiscriminator256': ('colorization_dcgan256',),
}


def materialize_networks(spec):
    """Snapshot adapter templates into args so edits participate in run identity."""
    names = COMPONENT_TEMPLATES.get(spec.get('factory'))
    if names is None:
        return
    args = spec.setdefault('args', {})
    overrides = args.get('networks', {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               or not v.strip() for k, v in overrides.items()):
        raise ValueError('component args.networks must map template names to nonempty HNDL source')
    if set(overrides) - set(names):
        raise ValueError(f'Unknown network templates: {sorted(set(overrides) - set(names))}')
    args['networks'] = {name: overrides[name] if name in overrides else packaged_source(name + '.hndl')
                        for name in names}
