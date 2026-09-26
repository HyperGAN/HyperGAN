"""The three example models, shared by the example scripts and the tests."""
from pathlib import Path

from hypergan import api_per_network as hg

from .data import PairedPoints, RingPoints
from .nets import Encoder, PairCritic, PointGenerator
from .observe import code_norm, g_over_d, radius_gap, scatter

HERE = Path(__file__).resolve().parent

CRITIC = '''
linear(64)
leaky_relu(0.2)
linear(64)
leaky_relu(0.2)
linear()
'''

CONDITIONAL_GENERATOR = '''
concat(z, code)
linear(32)
leaky_relu(0.2)
linear(32)
leaky_relu(0.2)
linear()
'''


def simple(steps=20):
    """One generator (your own nn.Module), one critic (HNDL), 2-D Gaussian-grid data."""
    model = {
        'generator': hg.generator(
            hg.net(PointGenerator, z_dim=4, width=64),
            inputs={'z': hg.LATENT},
            losses=[hg.adversarial('discriminator')]),
        'discriminator': hg.critic(
            hg.hndl(CRITIC, input_shape=['B', 2], output_shape=['B', 1]),
            inputs={'x': hg.CANDIDATE}),
    }
    return hg.recipe(model, name='api/simple',
                     data=hg.data('gaussian_grid', side=5, noise=0.05),
                     prior=hg.particles(z_dim=4, count=256),
                     training=hg.training(steps=steps, batch_size=32, device='cpu', seed=1))


def multi_network(steps=10):
    """Encoder + generator + two critics + reconstruction, each network owning its losses."""
    model = {
        'encoder': hg.encoder(
            hg.net(Encoder, dim=2, code=2),
            inputs={'condition': 'batch.condition'},
            losses=[hg.loss(code_norm, inputs={'code': 'encoder'}, weight=0.01, id='code_norm')]),
        'generator': hg.generator(
            hg.hndl(CONDITIONAL_GENERATOR, input_shape={'z': ['B', 4], 'code': ['B', 2]}, output_shape=['B', 2]),
            inputs={'z': hg.LATENT, 'code': 'encoder'},
            losses=[hg.adversarial('discriminator'),
                    hg.adversarial('marginal'),
                    hg.mse('generated', 'batch.real', weight=1.0, id='reconstruction')],
            optimizer=hg.adam(lr=6e-4, betas=(0.0, 0.999))),
        'discriminator': hg.critic(
            hg.net(PairCritic, dim=2, condition=2),
            inputs={'x': hg.CANDIDATE, 'condition': 'batch.condition'},
            penalty=hg.k3p(coeff=1.0),
            optimizer=hg.adam(lr_mult=1.5)),
        'marginal': hg.critic(
            hg.hndl(file=HERE / 'marginal_critic.hndl', input_shape=['B', 2], output_shape=['B', 1]),
            inputs={'x': hg.CANDIDATE},
            judges=[hg.judge('batch.real', 'generated', weight=0.5)],
            penalty=hg.k3p(coeff=0.5)),
    }
    return hg.recipe(model, name='api/multi-network',
                     data=hg.items(PairedPoints, count=1024),
                     prior=hg.particles(z_dim=4, count=64, optimizer=hg.adam(lr_mult=10.0)),
                     training=hg.training(steps=steps, batch_size=16, device='cpu', seed=2))


def extensions(steps=20):
    """Item-level data, a custom metric, a custom evaluation and a custom sampler."""
    model = {
        'generator': hg.generator(
            hg.net(PointGenerator, z_dim=4, width=64),
            inputs={'z': hg.LATENT},
            losses=[hg.adversarial('discriminator')]),
        'discriminator': hg.critic(
            hg.hndl(CRITIC, input_shape=['B', 2], output_shape=['B', 1]),
            inputs={'x': hg.CANDIDATE}),
    }
    observe = [
        hg.metric('g_over_d', g_over_d, inputs={'g_loss': 'update.g_loss', 'd_loss': 'update.d_loss'},
                  every=5, label='G/D loss ratio'),
        hg.evaluation('radius_gap', radius_gap, data=hg.items(RingPoints, count=256, seed=1),
                      samples=128, batch_size=32, seed=5, every=10, device='cpu', direction='minimize'),
        hg.sampler('scatter', scatter),
    ]
    return hg.recipe(model, name='api/extensions',
                     data=hg.items(RingPoints, count=2048, seed=0),
                     prior=hg.particles(z_dim=4, count=256),
                     training=hg.training(steps=steps, batch_size=32, device='cpu', seed=3),
                     sampling={'count': 64, 'seed': 3},
                     observe=observe)
