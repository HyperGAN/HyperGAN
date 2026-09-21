"""Load fixture architectures from configuration; adapters only inject test behavior."""
from pathlib import Path
from string import Template

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from hypergan.hndl_networks import build_network


_CONFIG = tomllib.loads((Path(__file__).parent / 'fixtures' / 'networks.toml').read_text())


def fixture_network(name, input_shape, output_shape, *, device='cpu', **parameters):
    """Construct an HNDL graph, preserving caller-owned RNG and device choices."""
    config = _CONFIG[name]
    if name == 'mlp':
        source = '\n'.join([config['input'], *([config['hidden']] * (parameters.pop('depth', 2) - 1)), config['output']])
        source = source.replace('\\n', '\n')
    else:
        source = config['source']
    source = Template(source).substitute({key: repr(value) for key, value in parameters.items()})
    return build_network(source, input_shape=('B', *input_shape), output_shape=('B', *output_shape)).to(device)


def fixture_linear(in_features, out_features, bias=True, *, device='cpu'):
    """Expose the HNDL-owned layer for tests that inspect its exact weight/state keys."""
    return fixture_network('linear', (in_features,), (out_features,), bias=bias, device=device).nodes.n_layer


def fixture_norm(channels, *, image=False):
    shape = (channels, 2, 2) if image else (channels,)
    return fixture_network('batch_norm', shape, shape).nodes.n_layer


def fixture_dropout(probability, width):
    return fixture_network('dropout', (width,), (width,), probability=probability).nodes.n_layer
