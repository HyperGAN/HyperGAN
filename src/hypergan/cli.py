"""Small command interface shared by installed console and module entrypoints."""

import argparse
import json
import math
from pathlib import Path
import sys

from . import __version__


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return number


def _run_options(parser, *, resume=False):
    parser.add_argument("--checkpoint-every", type=_positive_int, default=None if resume else 100,
                        help="save a complete checkpoint every N updates (default: 100; resume inherits)")
    parser.add_argument("--max-seconds", type=_positive_seconds,
                        help="stop at an update boundary after this attempt's wall-time budget")
    parser.add_argument("--stop-after-steps", type=_positive_int,
                        help="stop this attempt after N updates, preserving the total learning-rate schedule")
    parser.add_argument("--progress-json", action="store_true",
                        help="stream flushed JSONL events and a final result to stdout")
    previews = parser.add_mutually_exclusive_group()
    previews.add_argument("--preview-every", type=_positive_int,
                          help="publish an isolated EMA preview every N complete updates")
    previews.add_argument("--no-previews", dest="preview_every", action="store_const", const=0,
                          help="disable periodic previews for this attempt")
    parser.set_defaults(preview_every=None if resume else 0)
    parser.add_argument("--preview-keep", type=_positive_int, default=None if resume else 3,
                        help="retain at most N periodic previews (default: 3; resume inherits)")


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
    preflight = commands.add_parser("preflight", help="Check a CPU execution profile before training")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--profile", type=Path, required=True,
                           help="separate execution-profile TOML file")
    preflight.add_argument("--runtime", action="store_true",
                           help="also construct the recipe in bounded CPU workers (requires train extra)")
    data_check = commands.add_parser("data-check", help="Validate an image_folder inventory and preprocessing")
    data_check.add_argument("config", type=Path)
    data_check.add_argument("--output", type=Path, help="write the data manifest to a new file")
    inspect = commands.add_parser("inspect", help="Read a run manifest without loading model weights")
    inspect.add_argument("path", type=Path)
    events = commands.add_parser("events", help="Read a bounded page of run events without loading training")
    events.add_argument("run_dir", type=Path)
    events.add_argument("--cursor", help="opaque cursor returned by the previous page")
    events.add_argument("--limit", type=_positive_int, default=100)
    events.add_argument("--max-bytes", type=_positive_int, default=1048576)
    checkpoint = commands.add_parser("checkpoint", help="Request a checkpoint at the trainer's next safe boundary")
    checkpoint.add_argument("run_dir", type=Path)
    operation = checkpoint.add_mutually_exclusive_group()
    operation.add_argument("--request-id", help="reuse this ID to retry the same request safely")
    operation.add_argument("--status", metavar="REQUEST_ID", help="read a request receipt without submitting")
    checkpoint.add_argument("--attempt-id", help="target this attempt (default: current manifest attempt)")
    train = commands.add_parser("train", help="Run a bounded CPU numerical reference (requires the train extra)")
    train.add_argument("config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--steps", type=_positive_int)
    _run_options(train)
    resume = commands.add_parser("resume", help="Continue a complete CPU training checkpoint")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--checkpoint", type=Path, help="choose an older checkpoint within this run")
    resume.add_argument("--config", type=Path, help="verify exact compatibility with this configuration")
    _run_options(resume, resume=True)
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


def _progress(args):
    def emit(event):
        if args.progress_json:
            print(json.dumps(event, allow_nan=False), flush=True)
        elif event.get("event") == "train":
            print(f"step {event['step']}: D={event['d_loss']:.6g} G={event['g_loss']:.6g}",
                  file=sys.stderr, flush=True)
    return emit


def _run_result(args, result):
    if args.progress_json:
        print(json.dumps({"event": "result", "manifest": result}, allow_nan=False), flush=True)
    else:
        _print_json(result)


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
        elif args.command == "events":
            from .run_events import read_event_page

            _print_json(read_event_page(args.run_dir, args.cursor, limit=args.limit, max_bytes=args.max_bytes))
        elif args.command == "checkpoint":
            from .run_requests import checkpoint_request_status, submit_checkpoint_request

            if args.status is not None:
                if args.attempt_id is not None:
                    raise ValueError("--attempt-id cannot be combined with --status")
                _print_json(checkpoint_request_status(args.run_dir, args.status))
            else:
                with (args.run_dir / "manifest.json").open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                if not isinstance(manifest, dict) or not manifest.get("run_id") or not manifest.get("attempt_id"):
                    raise ValueError("Run manifest has no run/attempt identity for a checkpoint request")
                if manifest.get("status") != "running":
                    if args.request_id is None:
                        raise ValueError("Run is not running; use resume to continue a stopped run")
                    try:
                        checkpoint_request_status(args.run_dir, args.request_id)
                    except FileNotFoundError as exc:
                        raise ValueError("Run is not running; use resume to continue a stopped run") from exc
                _print_json(submit_checkpoint_request(args.run_dir, run_id=manifest["run_id"],
                                                      attempt_id=args.attempt_id or manifest["attempt_id"],
                                                      request_id=args.request_id))
        elif args.command == "data-check":
            from .config import load_config
            from .data import ImageFolder

            config = load_config(args.config)
            if config["data"]["factory"] != "image_folder":
                raise ValueError("data-check currently supports data.factory='image_folder'")
            data = ImageFolder(**config["data"]["args"])
            manifest = {"schema_version": 1, "data": data.resume_identity(), "inventory": data.inventory}
            if args.output is not None:
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            _print_json(manifest)
        elif args.command == "preflight":
            from .config import load_config
            from .execution_profiles import load_execution_profile

            config = load_config(args.config)
            _warnings(config)
            profile = load_execution_profile(args.profile, config)
            if args.runtime:
                from .execution_preflight import preflight

                _print_json(preflight(config, profile))
            else:
                _print_json({"schema_version": 1, "stage": "structural", "profile": profile,
                             "runtime_checked": False})
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
            from .config import config_values, load_config, resolve_config

            resolved = load_config(args.config)
            if args.steps is not None:
                values = config_values(resolved)
                values["training"]["steps"] = args.steps
                resolved = resolve_config(values)
            _warnings(resolved)
            from .training import train

            _run_result(args, train(args.config, args.run_dir, steps=args.steps,
                                   checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
                                   stop_after_steps=args.stop_after_steps, on_event=_progress(args),
                                   preview_every=args.preview_every, preview_keep=args.preview_keep))
        elif args.command == "resume":
            from .training import resume

            _run_result(args, resume(args.run_dir, checkpoint=args.checkpoint, config_path=args.config,
                                    checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
                                    stop_after_steps=args.stop_after_steps, on_event=_progress(args),
                                    preview_every=args.preview_every, preview_keep=args.preview_keep))
        elif args.command == "sample":
            from .artifacts import sample

            print(sample(args.run_dir, count=args.count, seed=args.seed, output=args.output))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "particlegan", "numpy"}:
            print("error: training dependencies are missing; install 'hypergan[train]' in this environment", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
