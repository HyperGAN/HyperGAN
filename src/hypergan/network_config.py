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


def validate_network_args(args, location):
    allowed = {'source', 'file', 'input_shape', 'output_shape', 'parameters', 'input_dtype', 'pretrained_providers'}
    if set(args) - allowed:
        raise ValueError(f'{location}: unknown HNDL arguments {sorted(set(args) - allowed)}')
    if ('source' in args) == ('file' in args):
        raise ValueError(f'{location}: specify exactly one of source or file')
    for key in ('input_shape', 'output_shape'):
        contract = args.get(key)
        shapes = contract if isinstance(contract, dict) else {key: contract}
        if not shapes or any(not isinstance(name, str) or not name.isidentifier() for name in shapes):
            raise ValueError(f'{location}.{key} must declare valid port names')
        for name, shape in shapes.items():
            if (not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3, 4)
                    or shape[0] != 'B' or any(type(size) is not int or size < 1 for size in shape[1:])):
                raise ValueError(f'{location}.{key}.{name} must be ["B", positive dimensions...]')
    if not isinstance(args.get('parameters', {}), dict):
        raise ValueError(f'{location}.parameters must be a table')
    validate_pretrained_providers(args.get('pretrained_providers', {}))
    if 'source' in args:
        render_source(args['source'], args.get('parameters'))
    elif not isinstance(args['file'], str) or not args['file']:
        raise ValueError(f'{location}.file must name a .hndl file')
    dtype = args.get('input_dtype')
    if dtype is not None:
        ports = args['input_shape'] if isinstance(args['input_shape'], dict) else {'x': None}
        dtypes = dtype if isinstance(dtype, dict) else {name: dtype for name in ports}
        if (not dtypes or set(dtypes) - set(ports)
                or any(value not in ('float32', 'int32', 'int64', 'bool') for value in dtypes.values())):
            raise ValueError(f'{location}.input_dtype must declare supported dtypes for input ports')


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
