"""Engine-recipe TOML fixtures for tests.

These write the internal resolved-recipe schema (``[components.*]``,
``[training]``), not the user-facing ``hypergan-model/1`` file, so they live
with the tests rather than in the package.
"""
from pathlib import Path

from hypergan.config import validate_device

LEGACY_REFERENCE_TOML = '''# A five-step numerical reference, not a converged image model.
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
weight = 1.0

[gradient_penalty]
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

# The released-trainer variant of the reference above.
DEFAULT_TOML = '''# Five-step task smoke test; increase steps for a training run.
schema_version = 1
name = "reference/100gaussians"
defaults = "particlegan"

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
source = "linear(64)\\nleaky_relu(0.2)\\nlinear(64)\\nleaky_relu(0.2)\\nlinear()"

[components.discriminator]
factory = "hndl"
inputs = { x = "candidate" }
[components.discriminator.args]
input_shape = ["B", 2]
output_shape = ["B", 1]
source = "linear(64)\\nleaky_relu(0.2)\\nlinear(64)\\nleaky_relu(0.2)\\nlinear()"

[prior]
kind = "particles"
initialization_seed = 2
[prior.args]
num_particles = 20000
z_dim = 4

[training]
steps = 5
batch_size = 16
seed = 0
device = "cpu"
initialization = "deterministic_orthogonal"

[sampling]
count = 256
seed = 10000
'''


def write_default(path, *, device="cuda", defaults="particlegan"):
    """Create a GPU-first project; CPU numerical fixtures opt in explicitly."""
    validate_device(device)
    if defaults not in ('particlegan', 'hypergan'):
        raise ValueError('defaults must be particlegan or hypergan')
    path = Path(path)
    if path.suffix != ".toml":
        path = path / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        template = DEFAULT_TOML if defaults == 'particlegan' else LEGACY_REFERENCE_TOML
        stream.write(template.replace('device = "cpu"', f'device = "{device}"'))
    return path
