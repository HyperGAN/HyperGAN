"""Demo 6: the same config file drives several processes.

Two CPU ranks (gloo) train the item-level ring model. Each rank rebuilds the
networks and the dataset from the saved config and the importable
my_project module. Every rank draws the same global index plan from the shared
data seed and keeps its slice (today each rank still loads the whole global
batch before slicing; loading only its slice needs ItemDataset.plan/load wired
into the replicated loop). Nothing is pickled from this script.

    cd examples/api/graph && python replicated.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from my_project import Generator, RingItems


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    here = Path(__file__).parent
    x = hg.data("x", shape=["B", 2])
    fake = hg.net(Generator, z=hg.latent(4), hidden=32)
    d = hg.hndl(file=here / "critic.hndl", shape=["B", 1], x=hg.candidate)
    run = hg.train(hg.dataset(RingItems, count=512), [hg.adversarial(d, real=x, fake=fake)],
                   name="demo/replicated", steps=20, batch_size=32, device="cpu",
                   run=runs / "replicated", profile="cpu-replicated-gloo")
    manifest = run.manifest
    print("execution:", manifest.get("execution"))
    print("steps:", run.step, "samples seen:", manifest.get("samples_seen"))
    for name, value in sorted(hg.last(run, ["loss/d_total", "loss/g_total"]).items()):
        print(f"  {name:<14} {value:.4f}")


if __name__ == "__main__":
    main()
