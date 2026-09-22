"""Small command interface shared by installed console and module entrypoints."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

from . import __version__


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _preview_keep(value):
    """Bound the retained preview history; 'all' keeps every published sample."""
    from .previews import KEEP_ALL
    if isinstance(value, str) and value.strip().lower() == "all":
        return KEEP_ALL
    return _positive_int(value)


def _public_origin(value):
    """An absolute http(s) origin for a TLS proxy in front of the viewer."""
    from .web_session import normalize_public_origin
    try:
        return normalize_public_origin(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _sample_name(value):
    # Validated without importing training dependencies.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,15}", value) is None:
        raise argparse.ArgumentTypeError(
            "must be 1-16 letters, digits, dots, colons, underscores or hyphens")
    return value


def _positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return number


def _run_options(parser):
    from .ports import DEFAULT_VIEWER_PORT
    parser.add_argument("--profile", help="execution profile name or TOML path; new runs default to native device, existing runs inherit")
    from .execution import SERVICE_TIMEOUTS
    for name in SERVICE_TIMEOUTS:
        parser.add_argument("--" + name.replace("_", "-"), type=_positive_seconds,
                            help="replicated service deadline in seconds (attempt policy)")
    server = parser.add_mutually_exclusive_group()
    server.add_argument("--server", action="store_true", help="require the local viewer before training starts")
    server.add_argument("--no-server", action="store_true", help="train without web imports or listening sockets")
    parser.add_argument("--port", "--server-port", dest="server_port", type=int,
                        help=f"require a specific viewer port (default: {DEFAULT_VIEWER_PORT}, "
                             "or the next free port above it; 0 selects any free port)")
    parser.add_argument("--server-host", help="listen address (default: 0.0.0.0)")
    parser.add_argument("--auth", choices=("none", "token"), help="viewer authentication (default: none)")
    parser.add_argument("--public-origin", type=_public_origin, metavar="URL",
                        help="absolute origin of a TLS proxy in front of the viewer, "
                             "e.g. https://machine.tailnet.ts.net (scheme, host and optional port only)")
    parser.add_argument("--open", action="store_true", help="require the viewer and open its sign-in page")
    parser.add_argument("--dev", "--viewer-dev", dest="viewer_dev", action="store_true",
                        help="serve browser assets fresh from this checkout (same as HYPERGAN_VIEWER_DEV=1)")
    parser.add_argument("--checkpoint-every", type=_positive_int,
                        help="save a complete checkpoint every N updates (default: 100; existing runs inherit)")
    parser.add_argument("--max-seconds", type=_positive_seconds,
                        help="stop at an update boundary after this attempt's wall-time budget")
    parser.add_argument("--stop-after-steps", type=_positive_int,
                        help="stop this attempt after N updates, preserving the total learning-rate schedule")
    parser.add_argument("--progress-every", type=_positive_int,
                        help="print routine progress every N updates (default: 100; saved UI setting inherits)")
    parser.add_argument("--progress-json", action="store_true",
                        help="stream cadence-filtered JSONL progress, lifecycle events and a final result to stdout")
    previews = parser.add_mutually_exclusive_group()
    previews.add_argument("--preview-every", type=_positive_int,
                          help="publish an isolated EMA preview every N complete updates")
    previews.add_argument("--no-previews", dest="preview_every", action="store_const", const=0,
                          help="disable periodic previews for this attempt")
    parser.set_defaults(preview_every=None)
    parser.add_argument("--preview-keep", type=_preview_keep, metavar="N",
                        help="retain at most N periodic previews; when the run outgrows N "
                             "the older samples are thinned by doubling their spacing, so "
                             "the slider still spans the whole run (default: 128; pass "
                             "'all' to keep every sample; a resume inherits only a bound "
                             "that was asked for explicitly)")
    parser.add_argument("--preview-name", type=_sample_name,
                        help="short stable name indexing this run's generated samples "
                             "(default: g; the real batch is published as x)")


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
    new.add_argument("--device", default="cuda", help="training device: cuda (default), cuda:N, or explicit cpu")
    validate = commands.add_parser("validate", help="Validate a project without loading training dependencies")
    validate.add_argument("path", type=Path)
    preflight = commands.add_parser("preflight", help="Check an execution profile before training")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--profile", type=Path,
                           help="execution-profile TOML file; omitted uses the configured native device")
    preflight.add_argument("--runtime", action="store_true",
                           help="also construct the recipe in bounded CPU/CUDA workers (requires train extra)")
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
    metrics = commands.add_parser("metrics", help="Read immutable metric definitions without loading training")
    metrics.add_argument("run_dir", type=Path)
    metrics.add_argument("--revision", help="historical catalog SHA256 (default: active catalog)")
    project = commands.add_parser("project", help="Build Python event-map contributions independently of serving")
    project.add_argument("run_dir", type=Path)
    project.add_argument("--map-spec", type=Path, help="JSON MapSpec; default selects published scalar metrics")
    project.add_argument("--follow", action="store_true", help="continue projecting newly completed events")
    project.add_argument("--limit", type=_positive_int, default=100, help="documents per bounded work page")
    contributions = commands.add_parser("contributions", help="Read a bounded page of mapped event frames")
    contributions.add_argument("run_dir", type=Path)
    contributions.add_argument("--map-revision", help="map SHA256 (default: built-in scalar map)")
    contributions.add_argument("--cursor", help="last applied projection cursor")
    contributions.add_argument("--limit", type=_positive_int, default=100)
    contributions.add_argument("--max-bytes", type=_positive_int, default=1048576)
    serve = commands.add_parser("serve", help="Serve an existing run over the local API and browser UI (web extra)")
    serve.add_argument("run_dir", type=Path)
    from .ports import DEFAULT_VIEWER_PORT
    serve.add_argument("--port", type=int,
                       help=f"listen port (default: {DEFAULT_VIEWER_PORT}, or the next free port "
                            "above it; 0 selects any free port)")
    serve.add_argument("--host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    serve.add_argument("--auth", choices=("none", "token"), default="none", help="authentication mode (default: none)")
    serve.add_argument("--public-origin", type=_public_origin, metavar="URL",
                       help="absolute origin of a TLS proxy in front of the viewer, "
                            "e.g. https://machine.tailnet.ts.net (scheme, host and optional port only)")
    serve.add_argument("--session-file", type=Path, help="new private credential file outside the run")
    serve.add_argument("--open", action="store_true", help="open the local viewer in a browser")
    serve.add_argument("--dev", action="store_true",
                       help="serve browser assets fresh from this checkout (same as HYPERGAN_VIEWER_DEV=1)")
    server_status = commands.add_parser("server-status", help="Read the automatic viewer URL, status and log location")
    server_status.add_argument("run_dir", type=Path)
    stop_server = commands.add_parser("stop-server", help="Stop the persistent automatic viewer for a run")
    stop_server.add_argument("run_dir", type=Path)
    from .evaluation_cli import add_evaluate_parser
    add_evaluate_parser(commands)
    from .signal_diagnostic_cli import add_signal_parser
    add_signal_parser(commands, _positive_int)
    checkpoint = commands.add_parser("checkpoint", help="Request a checkpoint at the trainer's next safe boundary")
    checkpoint.add_argument("run_dir", type=Path)
    operation = checkpoint.add_mutually_exclusive_group()
    operation.add_argument("--request-id", help="reuse this ID to retry the same request safely")
    operation.add_argument("--status", metavar="REQUEST_ID", help="read a request receipt without submitting")
    checkpoint.add_argument("--attempt-id", help="target this attempt (default: current manifest attempt)")
    train = commands.add_parser("train", help="Create a run or resume its latest checkpoint with the same configuration")
    train.add_argument("config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True, help="new run directory, or an existing run to resume")
    train.add_argument("--steps", type=_positive_int, help="total target steps; may increase on resume when lr_floor=1 (constant learning rate)")
    tuning = train.add_mutually_exclusive_group()
    tuning.add_argument("--tune", dest="tune", action="store_true",
                        help="calibrate owned generator initialization once before the first update; resumes keep saved weights")
    tuning.add_argument("--no-tune", dest="tune", action="store_false",
                        help="use the configured initialization without calibration (default)")
    train.set_defaults(tune=False)
    _run_options(train)
    resume = commands.add_parser("resume", help="Continue a complete training checkpoint on its recorded device")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--checkpoint", type=Path, help="choose an older checkpoint within this run")
    resume.add_argument("--config", type=Path, help="verify exact compatibility with this configuration")
    _run_options(resume)
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


def _training_viewer(args):
    if args.no_server:
        if (args.open or args.server_port is not None or args.server_host is not None
                or args.auth is not None or args.public_origin is not None or args.viewer_dev):
            raise ValueError("--no-server cannot be combined with --open, --port/--server-port, "
                             "--server-host, --auth, --public-origin or --dev")
        from contextlib import nullcontext
        return nullcontext()
    # The supervisor is a detached subprocess; the environment is what reaches it.
    if args.viewer_dev:
        from .web_dev import enable

        enable()
    from .web_autostart import training_viewer
    return training_viewer(args.run_dir,
                           required=args.server or args.open or args.server_port is not None
                           or args.server_host is not None or args.auth is not None
                           or args.public_origin is not None,
                           port=args.server_port, host=args.server_host, auth=args.auth,
                           public_origin=args.public_origin, open_browser=args.open)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command in {"train", "resume"}:
        from .bounded_cli_output import training_output
        with training_output(progress_json=args.progress_json) as output:
            return _dispatch(args, output=output)
    return _dispatch(args)


def _dispatch(args, *, output=None):
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
        elif args.command == "metrics":
            from .metrics import read_catalog

            _print_json(read_catalog(args.run_dir, args.revision))
        elif args.command == "evaluate":
            from .evaluation_cli import run_evaluate

            _print_json(run_evaluate(args))
        elif args.command == "diagnose-signal":
            from .signal_diagnostic_cli import run_signal

            _print_json(run_signal(args))
        elif args.command == "serve":
            from .web_dev import enable
            from .web_launch import serve

            if args.dev:
                enable()
            serve(args.run_dir, port=args.port, host=args.host, auth=args.auth,
                  session_file=args.session_file, open_browser=args.open,
                  public_origin=args.public_origin)
        elif args.command == "server-status":
            from .web_autostart import viewer_status

            _print_json(viewer_status(args.run_dir))
        elif args.command == "stop-server":
            from .web_autostart import stop_viewer

            _print_json(stop_viewer(args.run_dir))
        elif args.command == "contributions":
            from .event_views import MapSpec, read_projection_page

            _print_json(read_projection_page(args.run_dir, args.map_revision or MapSpec().revision,
                                            args.cursor, limit=args.limit, max_bytes=args.max_bytes))
        elif args.command == "project":
            from .event_views import MapSpec, Projector
            import time

            spec = MapSpec(**json.loads(args.map_spec.read_text(encoding="utf-8"))) if args.map_spec else MapSpec()
            # Reopen before the custom worker's five-minute lifetime expires.
            # Its durable projection frame is the only restart watermark.
            following = True
            while following:
                opened = time.monotonic()
                with Projector(args.run_dir, spec) as projector:
                    while True:
                        progress = projector.project(limit=args.limit)
                        if args.follow and progress["documents"]:
                            print(json.dumps(dict(progress, map_revision=spec.revision)), flush=True)
                        if not args.follow and not progress["has_more"]:
                            _print_json(dict(progress, map_revision=spec.revision))
                            following = False
                            break
                        if time.monotonic() - opened >= 240:
                            break
                        if not progress["has_more"]:
                            time.sleep(0.25)
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
            from .execution_preflight import _resolve_profile
            profile = (load_execution_profile(args.profile, config) if args.profile is not None
                       else _resolve_profile(None, config))
            if args.runtime:
                from .execution_preflight import preflight

                _print_json(preflight(config, profile))
            else:
                _print_json({"schema_version": 1, "stage": "structural", "profile": profile,
                             "runtime_checked": False})
        elif args.command in {"new", "validate", "recipes"}:
            from . import config

            if args.command == "new":
                print(config.write_default(args.path, device=args.device))
            elif args.command == "recipes":
                _print_json(config.list_recipes())
            else:
                resolved = config.load_config(args.path)
                _warnings(resolved)
                _print_json(resolved)
        elif args.command in {"train", "resume"}:
            from .execution import SERVICE_TIMEOUTS, prepare_train, prepare_resume

            options = dict(profile=args.profile,
                           service_policy={name: getattr(args, name) for name in SERVICE_TIMEOUTS
                                           if getattr(args, name) is not None},
                           checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
                           stop_after_steps=args.stop_after_steps,
                           preview_every=args.preview_every, preview_keep=args.preview_keep,
                           preview_name=args.preview_name)
            if args.command == "train":
                prepared = prepare_train(args.config, args.run_dir, args.steps, tune=args.tune, **options)
            else:
                prepared = prepare_resume(args.run_dir, args.checkpoint, args.config, **options)
            _warnings(prepared.config)
            from .metrics import manual_evaluation_hint, manual_evaluation_reminder

            # The generic warnings block is easy to skim past, so an all-manual
            # evaluation setup also gets one distinct line naming the exact edit.
            labels = {"run": str(args.run_dir),
                      "config_path": str(args.config) if getattr(args, "config", None) else "CONFIG"}
            hint = manual_evaluation_hint(prepared.config, **labels)
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
            output.configure(args.run_dir, progress_every=args.progress_every,
                             evaluation_reminder=manual_evaluation_reminder(prepared.config, **labels))
            with _training_viewer(args):
                output.result(prepared.run(on_event=output.progress), run_dir=args.run_dir)
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
