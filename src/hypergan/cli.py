"""Small command interface shared by installed console and module entrypoints."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser():
    parser = argparse.ArgumentParser(
        prog="hypergan", description="Create, validate, and run reproducible GAN projects."
    )
    parser.add_argument("--version", action="version", version=f"hypergan {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="Show the installed version")
    commands.add_parser("recipes", help="List available recipes and qualification status")
    new = commands.add_parser("new", help="Create a reference project configuration")
    new.add_argument("path", type=Path)
    validate = commands.add_parser("validate", help="Validate a project without loading training dependencies")
    validate.add_argument("path", type=Path)
    inspect = commands.add_parser("inspect", help="Read a run manifest without loading model weights")
    inspect.add_argument("path", type=Path)
    train = commands.add_parser("train", help="Run a bounded CPU numerical reference (requires the train extra)")
    train.add_argument("config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--steps", type=_positive_int)
    sample = commands.add_parser("sample", help="Sample a saved reference model (requires the train extra)")
    sample.add_argument("run_dir", type=Path)
    sample.add_argument("--count", type=_positive_int, default=16)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", type=Path)
    return parser


def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _warnings(config):
    for warning in config.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "version":
            print(f"hypergan {__version__}")
        elif args.command == "inspect":
            path = args.path / "manifest.json" if args.path.is_dir() else args.path
            with path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            if not isinstance(manifest, dict):
                raise ValueError("run manifest must contain a JSON object")
            _print_json(manifest)
        elif args.command in {"new", "validate", "recipes"}:
            from . import config

            if args.command == "new":
                print(config.write_default(args.path))
            elif args.command == "recipes":
                _print_json(config.list_recipes())
            else:
                resolved = config.load_config(args.path)
                _warnings(resolved)
                _print_json(resolved)
        elif args.command == "train":
            from .config import load_config

            _warnings(load_config(args.config))
            from .training import train

            _print_json(train(args.config, args.run_dir, steps=args.steps))
        elif args.command == "sample":
            from .artifacts import sample

            print(sample(args.run_dir, count=args.count, seed=args.seed, output=args.output))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "particlegan"}:
            print("error: training dependencies are missing; install 'hypergan[train]' in this environment", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
