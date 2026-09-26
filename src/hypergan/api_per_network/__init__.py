"""HyperGAN 2 Python API proof of concept: networks own their losses.

    from hypergan import api_per_network as hg

    model = {
        'generator': hg.generator(hg.net(MyG), inputs={'z': hg.LATENT},
                                  losses=[hg.adversarial('discriminator')]),
        'discriminator': hg.critic(hg.net(MyD), inputs={'x': hg.CANDIDATE}),
    }
    recipe = hg.recipe(model, data=hg.items(MyDataset), prior=hg.particles(z_dim=4, count=256),
                       training=hg.training(steps=100, batch_size=32, device='cpu'))
    hg.save(recipe, 'model.toml')         # a plain HyperGAN config file
    run = hg.train(recipe, 'runs/demo')   # or hg.train('model.toml', ...)
    hg.metrics(run)['loss/g_total']

Each network is declared with its role; everything that trains it is in that
declaration. The recipe lowers to the existing resolved configuration, so the
viewer, resume, fingerprints and replicated execution are unchanged.
"""
from .declare import *  # noqa: F401,F403
from .declare import __all__ as _declared
from .api import (  # noqa: F401
    Preview, Run, catalog, evaluate, evaluations, explain, fingerprint, load, metrics, previews, resume,
    run, samples, save, train, validate)
from .lowering import lift, lower  # noqa: F401

__all__ = list(_declared) + ['Preview', 'Run', 'catalog', 'evaluate', 'evaluations', 'explain', 'fingerprint',
                             'lift', 'load', 'lower', 'metrics', 'previews', 'resume', 'run', 'samples', 'save',
                             'train', 'validate']
