"""HyperGAN 2 functional graph API (proof of concept).

Build a model from references and a list of losses; roles come from the losses::

    import hypergan.graph as hg

    x = hg.data("real")
    z = hg.latent(4)
    fake = hg.net(Generator, z=z)
    d = hg.net(Critic, x=hg.candidate)
    run = hg.train(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake)], run="runs/demo", steps=100)
    print(hg.last(run))

The model lowers to the existing configuration and is saved as a TOML file
(``<run>.toml`` by default) that ``hypergan train`` can run unchanged. User
code is referenced by import path and never copied.
"""
from . import view
from .model import Model, describe, fingerprint, load, model, resolve, roles, save
from .nodes import (DataSource, Field, Latent, Net, Ref, adversarial, batches, candidate, dataset,
                    evaluation, gaussian_grid, hndl, l1, loss, metric, mse, net, paired_linear, sampler)
from .runs import Run, evaluate, evaluations, last, metrics, previews, resume, samples, train


def data(field="real", *, shape=None):
    """A field of the training batch (``real`` by default)."""
    return Field(field, shape)


def latent(dim, *, kind="particles", particles=None, **args):
    """The prior sample, ``[B, dim]``. ParticleGAN's particle prior by default."""
    return Latent(dim, kind, particles, **args)


__all__ = [
    "Model", "Run", "DataSource", "Field", "Latent", "Net", "Ref", "view",
    "data", "latent", "candidate", "net", "hndl",
    "adversarial", "l1", "mse", "loss",
    "metric", "evaluation", "sampler",
    "dataset", "batches", "gaussian_grid", "paired_linear",
    "model", "save", "load", "resolve", "fingerprint", "describe", "roles",
    "train", "resume", "metrics", "last", "evaluations", "evaluate", "samples", "previews",
]
