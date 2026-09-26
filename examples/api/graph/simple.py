"""Demo 1: the simplest case - one generator, one discriminator, your own networks.

The generator is a PyTorch class in my_project.py; the discriminator is HNDL
source in critic.hndl. Data is the builtin 2-D Gaussian grid (not images).

    cd examples/api/graph && python simple.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from my_project import Generator


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    here = Path(__file__).parent

    # Shorthand: one pair, roles by keyword.
    run = hg.train(hg.gaussian_grid(), generator=Generator, discriminator=hg.hndl(file=here / "critic.hndl", shape=["B", 1], x=hg.candidate),
                   latent=4, run=runs / "simple", steps=40, batch_size=64, device="cpu")

    # The same model spelled out: references plus one list of losses.
    x = hg.data()                       # batch field "real"
    z = hg.latent(4)
    fake = hg.net(Generator, z=z)       # non-reference kwargs would be constructor args
    d = hg.hndl(file=here / "critic.hndl", shape=["B", 1], x=hg.candidate)
    explicit = hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake)], steps=40, batch_size=64, device="cpu")
    assert hg.fingerprint(explicit) == run.manifest["config_sha256"], "shorthand and explicit forms are the same model"

    print(hg.describe(explicit))
    print("config file:", run.config_path)
    print("steps:", run.step)
    for name, value in sorted(hg.last(run, ["loss/d_total", "loss/g_total", "loss/gradient_penalty", "loss/prior_regularizer"]).items()):
        print(f"  {name:<24} {value:.4f}")


# Training starts worker processes that import this file; the guard keeps them from training too.
if __name__ == "__main__":
    main()
