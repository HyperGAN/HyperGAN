"""Demo 3: your own data loader, loss, metric, evaluation and sampler.

- data:       my_project.RingItems implements only __len__/__getitem__
- loss:       my_project.radius_loss, a plain function of generator output
- metric:     my_project.g_over_d(g_loss, d_loss), cheap, from update values
- evaluation: my_project.ring_error(generated, reference) on holdout items
- sampler:    my_project.scatter(points) -> a 2-D scatter view

    cd examples/api/graph && python extensions.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from my_project import Generator, RingItems, g_over_d, radius_loss, ring_error, scatter


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    here = Path(__file__).parent

    x = hg.data("x", shape=["B", 2])      # item field "x"; the shape feeds the HNDL critic
    z = hg.latent(4)
    fake = hg.net(Generator, z=z, hidden=32)
    d = hg.hndl(file=here / "critic.hndl", shape=["B", 1], x=hg.candidate)

    model = hg.model(
        hg.dataset(RingItems, count=512),
        [
            hg.adversarial(d, real=x, fake=fake),
            hg.loss(radius_loss, input=fake, weight=0.1, radius=1.0),
        ],
        metrics=[hg.metric(g_over_d, every=10, label="G/D loss ratio")],
        evaluations=[hg.evaluation(ring_error, hg.dataset(RingItems, count=256, seed=1), every=50,
                                   samples=128, batch_size=32, device="cpu")],
        samplers=[hg.sampler(scatter, points=fake, count=128)],
        name="demo/extensions", steps=200, batch_size=32, device="cpu",
    )
    print(hg.describe(model))
    run = hg.train(model, run=runs / "extensions", preview_every=50)

    print("steps:", run.step)
    print("custom metric g_over_d:", hg.metrics(run, ["g_over_d"]).get("g_over_d"))
    for metric_id, rows in hg.evaluations(run).items():
        print(f"evaluation {metric_id}:", [(row["step"], row["status"], row["value"]) for row in rows])
    for sampler_id, views in hg.samples(run).items():
        for view_id, paths in views.items():
            print(f"sampler {sampler_id}/{view_id}:", [str(p) for p in paths])
    print("previews:", [str(p) for p in hg.previews(run)])


# Training starts worker processes that import this file; the guard keeps them from training too.
if __name__ == "__main__":
    main()
