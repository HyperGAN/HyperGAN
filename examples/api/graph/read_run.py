"""Demo 5: read a run back - metrics, evaluations, samples and previews.

Reads the run written by extensions.py (run that first).

    cd examples/api/graph && python read_run.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from hypergan.metrics import read_catalog


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    run = hg.Run(runs / "extensions")
    print("run", run.path.name, "status", run.manifest["status"], "step", run.step)

    catalog = read_catalog(run.path)["metrics"]           # labels, units, owners
    series = hg.metrics(run)
    for name in ("loss/d_total", "loss/g_total", "loss/objectives/radius_loss", "g_over_d"):
        steps = [step for step, _ in series.get(name, [])]
        label = catalog.get(name, {}).get("label", "")
        print(f"  {name:<28} {label!r:<34} {len(steps)} points, last {series[name][-1][1]:.4f}")

    receipt = hg.evaluate(run, "ring_error")              # an explicit evaluation, now
    print("explicit evaluation:", receipt["status"], receipt["result"].get("value"))
    for metric_id, rows in hg.evaluations(run).items():
        print("evaluation", metric_id, [(row["step"], row["status"], row["value"]) for row in rows])

    for sampler_id, views in hg.samples(run, count=32, seed=7).items():
        for view_id, paths in views.items():
            print("sampler", sampler_id, view_id, [p.name for p in paths])
    print("previews:", len(hg.previews(run)))


if __name__ == "__main__":
    main()
