"""Demo 2: encoder + generator + two discriminators + reconstruction losses.

Roles come from the losses: the two networks scored by hg.adversarial are
critics; the encoder, generator and decoder are upstream of a fake sample or
a generator-side loss, so they train with the generator. Synthetic paired
vectors (condition -> target), tiny dimensions, CPU.

    cd examples/api/graph && python multi_network.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from my_project import ConditionalGenerator, Decoder, Encoder, PairCritic


def build(steps=10):
    """The model as a value; demos 4 and 5 reuse it."""
    here = Path(__file__).parent
    x = hg.data()                                   # the real target
    c = hg.data("condition")                        # paired input
    z = hg.latent(4)

    code = hg.net(Encoder, x=c, code=4)             # returns {"mu": ..., "scale": ...}
    fake = hg.net(ConditionalGenerator, z=z, code=code["mu"])
    recon = hg.net(Decoder, code=code["mu"])        # autoencoder path back to the condition

    pair = hg.net(PairCritic, x=hg.candidate, condition=c)                  # conditional critic
    marginal = hg.hndl(file=here / "critic.hndl", shape=["B", 1], x=hg.candidate, name="marginal")

    losses = [
        hg.adversarial(pair, real=x, fake=fake),                            # main pair
        hg.adversarial(marginal, real=x, fake=fake, weight=0.5, penalty=1.0),
        hg.l1(recon, c, name="reconstruction"),
        hg.mse(fake, x, weight=0.1, name="paired"),
    ]
    return hg.model(hg.paired_linear(), losses, name="demo/multi-network", steps=steps, batch_size=16, device="cpu")


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    here = Path(__file__).parent
    model = build()
    print(hg.describe(model))

    run = hg.train(model, run=runs / "multi_network", config=here / "multi_network.toml")
    print("config file:", run.config_path)
    print("steps:", run.step)
    for name, value in sorted(hg.last(run).items()):
        if name.startswith("loss/"):
            print(f"  {name:<32} {value:.4f}")


# Training starts worker processes that import this file; the guard keeps them from training too.
if __name__ == "__main__":
    main()
