"""Demo 4: the config file is the durable artifact.

Save the multi-network model, load it back (as hg.load and as the CLI's
loader), compare fingerprints, show that an edited file is never silently
overwritten, and train straight from the file.

    cd examples/api/graph && python round_trip.py [runs-dir]
"""
import sys
from pathlib import Path

import hypergan.graph as hg
from hypergan.config import fingerprint, load_config
from multi_network import build


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    runs.mkdir(parents=True, exist_ok=True)
    model = build(steps=3)
    path = hg.save(model, runs / "round_trip.toml", overwrite=True)

    loaded = hg.load(path)
    print("built   ", hg.fingerprint(model))
    print("hg.load ", hg.fingerprint(loaded))
    print("CLI load", fingerprint(load_config(path)))
    assert hg.fingerprint(model) == hg.fingerprint(loaded) == fingerprint(load_config(path))
    assert loaded.roles == model.roles, (loaded.roles, model.roles)
    print("roles   ", loaded.roles)

    hg.save(model, path)                       # unchanged file: saving again is fine
    path.write_text(path.read_text().replace("weight = 0.1", "weight = 0.2"))
    try:
        hg.save(model, path)                   # hand-edited: refuse to clobber it
    except FileExistsError as error:
        print("refused:", error)
    edited = hg.load(path)
    print("edited  ", hg.fingerprint(edited), "(a different model)")

    run = hg.train(path, run=runs / "round_trip")  # trains the file as written
    print("trained from file:", run.config_path, "steps", run.step,
          "manifest fingerprint matches edited file:", run.manifest["config_sha256"] == hg.fingerprint(edited))


if __name__ == "__main__":
    main()
