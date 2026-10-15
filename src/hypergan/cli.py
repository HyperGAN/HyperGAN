"""Small command interface shared by installed console and module entrypoints."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
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
    parser.add_argument("--host", "--server-host", dest="server_host", default=None, help="listen address (default: 0.0.0.0)")
    parser.add_argument("--auth", choices=("none", "token"), help="viewer authentication (default: none)")
    parser.add_argument("--public-origin", type=_public_origin, metavar="URL",
                        help="absolute origin of a TLS proxy in front of the viewer, "
                             "e.g. https://machine.tailnet.ts.net (scheme, host and optional port only)")
    parser.add_argument("--open", action="store_true", help="require the viewer and open its sign-in page")
    parser.add_argument("--dev", "--viewer-dev", dest="viewer_dev", action="store_true",
                        help="serve browser assets fresh from this checkout (same as HYPERGAN_VIEWER_DEV=1)")
    parser.add_argument("--checkpoint-every", type=_positive_int,
                        help="save a complete checkpoint every N updates (default: 1000; existing runs inherit)")
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


class _CustomParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        if len(args) >= 2 and args[0] == "checkpoint" and args[1] == "prune":
            args = ["checkpoints", *args[1:]]
        return super().parse_args(args, namespace)

    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        if len(args) >= 2 and args[0] == "checkpoint" and args[1] == "prune":
            args = ["checkpoints", *args[1:]]
        return super().parse_known_args(args, namespace)


def _parser():
    parser = _CustomParser(
        prog="hypergan", description="Create, validate, and run reproducible GAN projects."
    )
    parser.add_argument("--version", action="version", version=f"hypergan {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="Show the installed version")
    commands.add_parser("recipes", help="List available recipes and qualification status")
    new = commands.add_parser("new", help="Create a reference project configuration")
    new.add_argument("path", type=Path)
    new.add_argument("--device", default="cuda", help="training device: cuda (default), cuda:N, or explicit cpu")
    new.add_argument("--dataset", type=Path, default=None, help="directory of images to size and generate recipe for")
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
    status_cmd = commands.add_parser("status", help="Show a concise, human-friendly status summary of a run")
    status_cmd.add_argument("run_dir", type=Path, help="path to run directory")
    status_cmd.add_argument("--json", action="store_true", help="output structured status as JSON")
    runs = commands.add_parser("runs", help="Inspect and compare experiment runs")
    runs_subs = runs.add_subparsers(dest="runs_command", required=True)
    runs_ls = runs_subs.add_parser("ls", help="List runs under root directory")
    runs_ls.add_argument("root", nargs="?", type=Path, default=None, help="runs root directory")
    runs_ls.add_argument("--sort", choices=("fid", "best", "step", "updated", "name"), default="updated", help="sort order")
    runs_ls.add_argument("--status", choices=("running", "stopped", "completed", "failed"), default=None, help="filter by status")
    runs_ls.add_argument("--json", action="store_true", help="output runs table as JSON array")
    runs_compare = runs_subs.add_parser("compare", help="Compare multiple runs side-by-side")
    runs_compare.add_argument("runs", nargs="+", type=Path, help="two or more run directories")
    runs_compare.add_argument("--metric", required=True, help="metric name to compare (e.g. fid50k_train, loss_g)")
    runs_compare.add_argument("--at", required=True, help="steps to compare at (e.g. 50k,100k,best,latest)")
    runs_compare.add_argument("--csv", action="store_true", help="export comparison as CSV")
    gc_cmd = commands.add_parser("gc", help="Clean rebuildable views and compact events in finished runs")
    gc_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory to clean")
    gc_cmd.add_argument("--all", action="store_true", help="clean all runs in workspace runs root")
    gc_cmd.add_argument("--dry-run", action="store_true", help="calculate reclaimable space without modifying files")
    gc_cmd.add_argument("--stride", type=_positive_int, default=10, help="compaction stride for routine train events (default: 10)")
    model = commands.add_parser("model", help="Describe a run's formulation, losses, optimizers and networks")
    model.add_argument("path", type=Path, help="run directory, manifest.json or project .toml")
    model.add_argument("--networks", action="store_true",
                       help="also build each component on the meta device for per-layer detail (loads torch)")
    model.add_argument("--write", action="store_true",
                       help="record the per-layer detail as RUN/model.json for the viewer (implies --networks)")
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
    serve.add_argument("--port", "--server-port", dest="port", type=int,
                       help=f"listen port (default: {DEFAULT_VIEWER_PORT}, or the next free port "
                            "above it; 0 selects any free port)")
    serve.add_argument("--host", "--server-host", dest="host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    serve.add_argument("--auth", choices=("none", "token"), default="none", help="authentication mode (default: none)")
    serve.add_argument("--public-origin", type=_public_origin, metavar="URL",
                       help="absolute origin of a TLS proxy in front of the viewer, "
                            "e.g. https://machine.tailnet.ts.net (scheme, host and optional port only)")
    serve.add_argument("--session-file", type=Path, help="new private credential file outside the run")
    serve.add_argument("--open", action="store_true", help="open the local viewer in a browser")
    serve.add_argument("--dev", "--viewer-dev", dest="dev", action="store_true",
                       help="serve browser assets fresh from this checkout (same as HYPERGAN_VIEWER_DEV=1)")
    from .ports import DEFAULT_VIEWER_PORT
    view = commands.add_parser("view", help="Start, inspect, or stop the root dashboard for a runs directory")
    view.add_argument("root", nargs="?", type=Path, default=None,
                      help="runs root directory (default: workspace runs_root or current directory)")
    view_mode = view.add_mutually_exclusive_group()
    view_mode.add_argument("--status", action="store_true", help="report dashboard status, URL, version, and PID")
    view_mode.add_argument("--stop", action="store_true", help="cleanly stop the running dashboard")
    view.add_argument("--port", "--server-port", dest="port", type=int, default=DEFAULT_VIEWER_PORT, help="port to bind (default: 8765)")
    view.add_argument("--host", "--server-host", dest="host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    view.add_argument("--auth", choices=("none", "token"), default="none", help="authentication mode (default: none)")
    view.add_argument("--open", action="store_true", help="open the local viewer in a browser")
    view.add_argument("--dev", "--viewer-dev", dest="dev", action="store_true",
                      help="serve browser assets fresh from this checkout (same as HYPERGAN_VIEWER_DEV=1)")
    view.add_argument("--public-origin", type=_public_origin, metavar="URL",
                      help="absolute origin of a TLS proxy in front of the viewer")
    server_status = commands.add_parser("server-status", help="Read the automatic viewer URL, status and log location")
    server_status.add_argument("run_dir", type=Path)
    stop_server = commands.add_parser("stop-server", help="Stop the persistent automatic viewer or all viewers across the host")
    stop_server.add_argument("run_dir", type=Path, nargs="?", default=None, help="run or root directory to stop")
    stop_server.add_argument("--all", action="store_true", help="safely kill all legacy leaked viewers and projectors across the host")
    from .evaluation_cli import add_evaluate_parser
    add_evaluate_parser(commands)
    checkpoint = commands.add_parser("checkpoint", help="Request a checkpoint at the trainer's next safe boundary")
    checkpoint.add_argument("run_dir", type=Path)
    operation = checkpoint.add_mutually_exclusive_group()
    operation.add_argument("--request-id", help="reuse this ID to retry the same request safely")
    operation.add_argument("--status", metavar="REQUEST_ID", help="read a request receipt without submitting")
    checkpoint.add_argument("--attempt-id", help="target this attempt (default: current manifest attempt)")
    checkpoints = commands.add_parser("checkpoints", help="Inspect and manage run checkpoints")
    checkpoints_subs = checkpoints.add_subparsers(dest="checkpoints_command", required=True)
    prune = checkpoints_subs.add_parser("prune", help="Prune obsolete checkpoints under the retention policy")
    prune.add_argument("run_dir", type=Path, help="path to run directory")
    prune.add_argument("--dry-run", action="store_true", help="evaluate retention without deleting any files")
    export = commands.add_parser("export", help="Export a standalone EMA inference model bundle from a run checkpoint")
    export.add_argument("run_dir", type=Path, help="run directory")
    export.add_argument("--checkpoint", default="latest", help="target checkpoint: best, latest, step number, or path")
    export.add_argument("-o", "--output", type=Path, help="output file or directory (default: <run_dir>/model.pt)")
    convert = commands.add_parser("convert", help="Convert legacy engine recipe to hypergan-model/1 TOML")
    convert.add_argument("recipe", type=Path, help="path to legacy recipe TOML")
    convert.add_argument("--output", type=Path, required=True, help="output model.toml destination")
    def _add_train_and_resume_arguments(p, *, is_resume=False):
        if is_resume:
            p.add_argument("run_dir", type=Path, help="existing run directory to resume")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="explicit run directory override")
            p.add_argument("--config", type=Path, default=None, help="verify exact compatibility with this configuration")
        else:
            p.add_argument("config", type=Path, help="configuration file or existing run directory to resume")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="new run directory, or an existing run to resume")
            p.add_argument("--config", type=Path, default=None, help="explicit configuration file override")
        p.add_argument("--checkpoint", default=None, help="target checkpoint: best, latest, step number, or path")
        p.add_argument("--steps", type=_positive_int, help="total target steps; may increase on resume when lr_floor=1 (constant learning rate)")
        p.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                       help="override configuration value (e.g. --set train.device=cuda:1 or --set losses.reconstruction.weight=0.5)")
        _run_options(p)

    train = commands.add_parser("train", help="Create a run or resume its latest checkpoint with the same configuration")
    _add_train_and_resume_arguments(train, is_resume=False)

    resume = commands.add_parser("resume", help="Continue a complete training checkpoint on its recorded device")
    _add_train_and_resume_arguments(resume, is_resume=True)
    sample = commands.add_parser("sample", help="Sample a saved reference model (requires the train extra)")
    sample.add_argument("run_dir", type=Path)
    sample.add_argument("--count", type=_positive_int, default=16)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", type=Path)
    sample.add_argument("--sampler", help="sampler name declared in [samplers], or module:function import path")
    sample.add_argument("--checkpoint", default=None, help="target checkpoint: best, latest, step number, or path")
    return parser


def _parse_set_value(raw: str) -> object:
    val = raw.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "none"):
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def _normalize_set_key(key: str) -> str:
    key = re.sub(r'\[([^\]]+)\]', r'.\1', key)
    return key.strip().strip('.')


def _parse_set_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}; expected KEY=VALUE")
        key, raw_val = item.split("=", 1)
        norm_key = _normalize_set_key(key)
        val = _parse_set_value(raw_val)
        overrides[norm_key] = val
    return overrides


def _check_reexec(run_spec, ws_data):
    """Determine if re-exec is required based on [run].env or workspace env profiles.
    
    Returns:
        (needs_reexec, target_python, target_src_paths, target_env_vars)
    """
    if not isinstance(run_spec, dict):
        return False, None, [], {}

    ws_data = ws_data or {}
    envs = ws_data.get("envs", {})
    ws_root = Path(ws_data.get("workspace", {}).get("root", ".")).resolve()

    target_python = None
    target_src_paths = []
    target_env_vars = {}

    env_ref = run_spec.get("env")
    if isinstance(env_ref, str):
        if env_ref in envs:
            env_conf = envs[env_ref]
            py_val = env_conf.get("interpreter") or env_conf.get("python")
            if py_val:
                py_str = str(py_val).strip()
                cand = Path(py_str)
                if cand.is_absolute():
                    target_python = str(cand)
                elif "/" in py_str or (ws_root / cand).exists():
                    target_python = str((ws_root / cand).absolute())
                else:
                    w = shutil.which(py_str)
                    if w:
                        target_python = str(Path(w).absolute())
                    else:
                        target_python = str((ws_root / cand).absolute())

            src_val = env_conf.get("worktree") or env_conf.get("src")
            if src_val is not None:
                if isinstance(src_val, (list, tuple)):
                    raw_list = list(src_val)
                elif isinstance(src_val, (str, Path)):
                    raw_list = [src_val]
                else:
                    raw_list = []
                for s in raw_list:
                    sp = Path(s)
                    if not sp.is_absolute():
                        sp = ws_root / sp
                    sp = sp.resolve()
                    if (sp / "src").is_dir():
                        p_str = str((sp / "src").resolve())
                    else:
                        p_str = str(sp)
                    if p_str not in target_src_paths:
                        target_src_paths.append(p_str)

            if "env" in env_conf and isinstance(env_conf["env"], dict):
                target_env_vars.update({k: str(v) for k, v in env_conf["env"].items()})
        elif Path(env_ref).is_file():
            target_python = str(Path(env_ref).absolute())
    elif isinstance(env_ref, dict):
        target_env_vars.update({k: str(v) for k, v in env_ref.items()})

    # [run.env] inline table if present
    if "env" in run_spec and isinstance(run_spec["env"], dict):
        target_env_vars.update({k: str(v) for k, v in run_spec["env"].items()})

    needs_reexec = False
    if target_python:
        tp = Path(target_python)
        venv_cfg = tp.parent.parent / "pyvenv.cfg"
        if venv_cfg.is_file():
            if Path(sys.prefix).resolve() != tp.parent.parent.resolve():
                needs_reexec = True
        else:
            if Path(sys.executable).resolve() != tp.resolve():
                needs_reexec = True
    resolved_head = [str(Path(x).resolve()) for x in sys.path[:len(target_src_paths) + 3] if x]
    for p in target_src_paths:
        if str(Path(p).resolve()) not in resolved_head:
            needs_reexec = True
            break
    for k, v in target_env_vars.items():
        if os.environ.get(k) != str(v):
            needs_reexec = True
            break

    return needs_reexec, target_python, target_src_paths, target_env_vars


def _determine_attempt(run_dir: Path) -> int:
    reexec_att = os.environ.get("_HYPERGAN_REEXEC_ATTEMPT")
    if reexec_att:
        try:
            return int(reexec_att)
        except ValueError:
            pass
    launch_dir = run_dir / "launch"
    if launch_dir.is_dir():
        existing = [p for p in launch_dir.glob("*.json") if re.fullmatch(r"\d{4}\.json", p.name)]
        if existing:
            return max(int(p.stem) for p in existing) + 1
    return 1


def _reconstruct_argv_for_reexec(argv, old_config_str, new_config_str, run_dir_str):
    new_args = []
    idx = 0
    if len(argv) > 0:
        if "hypergan" not in Path(argv[0]).name:
            if len(argv) >= 3 and argv[1] == "-m" and argv[2] == "hypergan":
                idx = 3
            else:
                idx = 1
        else:
            idx = 1

    has_run_dir = False
    old_p = Path(old_config_str).resolve()
    while idx < len(argv):
        arg = argv[idx]
        if arg in ("--set", "-s"):
            idx += 2 if idx + 1 < len(argv) else 1
            continue
        elif arg.startswith("--set=") or arg.startswith("-s="):
            idx += 1
            continue

        if arg == "--run-dir":
            has_run_dir = True
            new_args.extend([arg, argv[idx + 1]])
            idx += 2
            continue
        elif arg.startswith("--run-dir="):
            has_run_dir = True
            new_args.append(arg)
            idx += 1
            continue

        try:
            if Path(arg).resolve() == old_p:
                new_args.append(new_config_str)
                idx += 1
                continue
        except Exception:
            pass
        new_args.append(arg)
        idx += 1

    if not has_run_dir:
        new_args.extend(["--run-dir", run_dir_str])

    return new_args



def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _warnings(config):
    warnings = config.get("warnings", [])
    has_trust_notice = False
    for warning in warnings:
        w_lower = warning.lower()
        if ("unqualified until separately evaluated" in w_lower
                or "execute trusted python" in w_lower
                or "configuration is not a sandbox" in w_lower
                or "bounded workers" in w_lower):
            has_trust_notice = True
        else:
            print(f"warning: {warning}", file=sys.stderr)
    if has_trust_notice:
        print("warning: custom recipe executes trusted Python code; combinations are unqualified until evaluated", file=sys.stderr)


def _training_viewer(args, *, config=None):
    if args.no_server:
        if (args.open or args.server_port is not None or args.server_host is not None
                or args.auth is not None or args.public_origin is not None or args.viewer_dev):
            raise ValueError("--no-server cannot be combined with --open, --port/--server-port, "
                             "--server-host, --auth, --public-origin or --dev")
        from contextlib import nullcontext
        return nullcontext()
    if config and isinstance(config, dict) and config.get('run', {}).get('viewer') is False:
        from contextlib import nullcontext
        return nullcontext()
    # The supervisor is a detached subprocess; the environment is what reaches it.
    if args.viewer_dev:
        from .web_dev import enable

        enable()
    from .ports import DEFAULT_VIEWER_PORT
    from .web_autostart import training_viewer, ensure_dashboard_running, find_runs_root

    run_dir = Path(args.run_dir).resolve()
    runs_root = find_runs_root(run_dir)
    required = bool(args.server or args.open or args.server_port is not None
                    or args.server_host is not None or args.auth is not None
                    or args.public_origin is not None)
    try:
        url = ensure_dashboard_running(runs_root, port=args.server_port or DEFAULT_VIEWER_PORT,
                                       host=args.server_host or '0.0.0.0',
                                       auth=args.auth or 'none',
                                       public_origin=args.public_origin,
                                       check_version=True,
                                       run_id=run_dir.name if run_dir != runs_root else None)
        args._dashboard_url = url
    except Exception as exc:
        args._dashboard_url = None
        if required:
            raise
        print(f"warning: viewer unavailable: {exc}; training continues headless", file=sys.stderr)

    return training_viewer(args.run_dir,
                           required=required,
                           port=args.server_port, host=args.server_host, auth=args.auth,
                           public_origin=args.public_origin, open_browser=args.open,
                           config=config)


def _print_training_header(run_dir: Path, experiment_name: str, device: str, dashboard_url: Optional[str] = None):
    print("=== HyperGAN ===", file=sys.stderr)
    print(f"Run directory:  {run_dir}", file=sys.stderr)
    print(f"Experiment:     {experiment_name}", file=sys.stderr)

    device_str = str(device)
    if "cuda" in device_str.lower():
        try:
            import torch
            dev_idx = 0
            if ":" in device_str:
                dev_idx = int(device_str.split(":")[1])
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(dev_idx)
                device_str = f"gpu{dev_idx} (CUDA {dev_idx}: {prop.name})"
            else:
                device_str = f"{device_str} (CUDA)"
        except Exception:
            device_str = f"{device_str} (CUDA)"
    elif device_str.lower() == "cpu":
        device_str = "cpu (CPU)"
    print(f"Devices:        {device_str}", file=sys.stderr)

    import platform
    print(f"Environment:    Python {platform.python_version()} ({sys.executable})", file=sys.stderr)

    if dashboard_url:
        lan_url = None
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
            if lan_ip and lan_ip != "127.0.0.1" and "localhost" in dashboard_url:
                lan_url = dashboard_url.replace("localhost", lan_ip).replace("127.0.0.1", lan_ip)
        except Exception:
            pass
        if lan_url:
            print(f"Dashboard:      {dashboard_url} (or {lan_url})", file=sys.stderr)
        else:
            print(f"Dashboard:      {dashboard_url}", file=sys.stderr)
    else:
        print("Dashboard:      headless (viewer disabled)", file=sys.stderr)



def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command in {"train", "resume"}:
        from .bounded_cli_output import training_output
        with training_output(progress_json=args.progress_json) as output:
            return _dispatch(args, output=output)
    return _dispatch(args)


def _dispatch(args, *, output=None):
    try:
        from . import api as hg

        if args.command == "version":
            print(f"hypergan {__version__}")
        elif args.command == "inspect":
            run = hg.open_run(args.path)
            _print_json(run.manifest)
        elif args.command == "status":
            res = hg.status(args.run_dir)
            if args.json:
                _print_json(res)
            else:
                from .inspection import format_run_status
                print(format_run_status(res))
        elif args.command == "runs":
            if args.runs_command == "ls":
                runs_list = hg.runs_ls(args.root, sort=args.sort, status_filter=args.status)
                if args.json:
                    _print_json(runs_list)
                else:
                    from .inspection import format_runs_table
                    print(format_runs_table(runs_list))
            elif args.runs_command == "compare":
                comp = hg.runs_compare(args.runs, metric=args.metric, at_steps=args.at)
                if args.csv:
                    from .inspection import format_comparison_csv
                    print(format_comparison_csv(comp), end="")
                else:
                    from .inspection import format_comparison_table
                    print(format_comparison_table(comp))
        elif args.command == "gc":
            if args.all:
                res = hg.gc_all(root=args.run_dir, dry_run=args.dry_run, stride=args.stride)
                from .inspection import format_bytes
                cleaned = [r for r in res if not r.get("skipped")]
                skipped = [r for r in res if r.get("skipped")]
                cleaned_count = len(cleaned)
                skipped_count = len(skipped)
                reclaimed = sum(r.get("total_reclaimed", 0) for r in cleaned)
                print(f"GC complete across {cleaned_count + skipped_count} runs:")
                print(f"  Cleaned:  {cleaned_count} runs (reclaimed {format_bytes(reclaimed)})")
                if skipped_count:
                    print(f"  Skipped:  {skipped_count} active runs")
            else:
                if args.run_dir is None:
                    raise ValueError("run_dir is required unless --all is specified")
                res = hg.gc(args.run_dir, dry_run=args.dry_run, stride=args.stride)
                from .inspection import format_bytes
                views_str = format_bytes(res.get("views_reclaimed_bytes", 0))
                events_str = format_bytes(res.get("events_reclaimed_bytes", 0))
                total_str = format_bytes(res.get("reclaimed_bytes", 0))
                prefix = "Would clean" if args.dry_run else "Cleaned"
                print(f"{prefix} run: {args.run_dir}")
                print(f"  Views removed:     {views_str}")
                print(f"  Events compacted:  {events_str}")
                print(f"  Total reclaimed:   {total_str}")
        elif args.command == "model":
            desc = hg.describe(args.path, networks=args.networks, write=args.write)
            if args.write:
                from .model_description import MODEL_FILE
                print(f"wrote {args.path / MODEL_FILE}", file=sys.stderr)
            _print_json(desc)
        elif args.command == "events":
            _print_json(hg.events(args.run_dir, cursor=args.cursor, limit=args.limit, max_bytes=args.max_bytes))
        elif args.command == "metrics":
            _print_json(hg.catalog(args.run_dir, revision=args.revision))
        elif args.command in {"evaluate", "eval"}:
            from .evaluation_cli import run_evaluate
            _print_json(run_evaluate(args))
        elif args.command == "serve":
            hg.serve(args.run_dir, port=args.port, host=args.host, auth=args.auth,
                     session_file=args.session_file, open_browser=args.open,
                     public_origin=args.public_origin, dev=args.dev)
        elif args.command == "view":
            root = args.root or hg.find_runs_root(Path.cwd())
            if args.status:
                _print_json(hg.dashboard_status(root))
            elif args.stop:
                _print_json(hg.stop_dashboard(root))
            else:
                hg.serve(root, port=args.port, host=args.host, auth=args.auth,
                         open_browser=args.open, public_origin=args.public_origin, dev=args.dev)
        elif args.command == "server-status":
            _print_json(hg.viewer_status(args.run_dir))
        elif args.command == "stop-server":
            if getattr(args, "all", False):
                killed = hg.stop_all_servers()
                _print_json({"status": "stopped", "killed_pids": killed, "count": len(killed)})
            else:
                if args.run_dir is None:
                    raise ValueError("run_dir is required unless --all is specified")
                _print_json(hg.stop_viewer(args.run_dir))
        elif args.command == "contributions":
            _print_json(hg.contributions(args.run_dir, map_revision=args.map_revision,
                                        cursor=args.cursor, limit=args.limit, max_bytes=args.max_bytes))
        elif args.command == "project":
            res = hg.project(args.run_dir, map_spec=args.map_spec, follow=args.follow, limit=args.limit,
                             on_progress=lambda p: (print(json.dumps(p), flush=True) if args.follow else _print_json(p)))
            if not args.follow and res is not None:
                _print_json(res)
        elif args.command == "checkpoint":
            if args.status is not None:
                if args.attempt_id is not None:
                    raise ValueError("--attempt-id cannot be combined with --status")
                _print_json(hg.checkpoint_status(args.run_dir, args.status))
            else:
                _print_json(hg.request_checkpoint(args.run_dir, attempt_id=args.attempt_id, request_id=args.request_id))
        elif args.command == "checkpoints":
            if getattr(args, "checkpoints_command", None) == "prune":
                _print_json(hg.prune_checkpoints(args.run_dir, dry_run=args.dry_run))
        elif args.command == "export":
            out = hg.export(args.run_dir, checkpoint=args.checkpoint, output=args.output)
            print(out)
        elif args.command == "data-check":
            _print_json(hg.data_check(args.config, output=args.output))
        elif args.command == "preflight":
            cfg = hg.load(args.config)
            _warnings(hg.resolve(cfg))
            _print_json(hg.preflight(cfg, profile=args.profile, runtime=args.runtime))
        elif args.command in {"new", "validate", "recipes", "convert"}:
            if args.command == "new":
                if args.dataset is not None:
                    from .recipe_generator import generate_recipe
                    dest = args.path if args.path.suffix == ".toml" else (args.path / "model.toml" if args.path.is_dir() else args.path.with_suffix(".toml"))
                    name = args.path.stem
                    saved = generate_recipe(name, args.dataset, dest, device=args.device)
                    print(saved)
                else:
                    from .config import validate_device
                    validate_device(args.device)
                    m = hg.packaged("gaussian-grid")
                    if args.device:
                        m = hg.override(m, {"train.device": args.device})
                    dest = args.path / "model.toml" if (args.path.is_dir() or args.path.suffix != ".toml") else args.path
                    if dest.exists():
                        raise FileExistsError(f"{dest} already exists; choose a new path")
                    saved = hg.save(m, dest)
                    print(saved)
            elif args.command == "recipes":
                _print_json(hg.packaged_list())
            elif args.command == "convert":
                raw = hg.load_recipe(args.recipe)
                m = hg.lift(raw, base=args.recipe.parent)
                saved = hg.save(m, args.output)
                print(saved)
            else:
                m = hg.load(args.path)
                warnings = hg.validate(m)
                for warning in warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                _print_json(hg.resolve(m))
        elif args.command in {"train", "resume"}:
            from .execution import SERVICE_TIMEOUTS
            options = dict(profile=args.profile,
                           service_policy={name: getattr(args, name) for name in SERVICE_TIMEOUTS
                                           if getattr(args, name) is not None},
                           checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
                           stop_after_steps=args.stop_after_steps,
                           preview_every=args.preview_every, preview_keep=args.preview_keep,
                           preview_name=args.preview_name)
            sets = _parse_set_overrides(args.sets) if getattr(args, "sets", None) else {}
            is_resume_cmd = args.command == "resume"
            target_arg = getattr(args, "run_dir", None) if is_resume_cmd else getattr(args, "config", None)
            if target_arg is None:
                target_arg = getattr(args, "config", None) or getattr(args, "run_dir", None)
            target_path = target_arg.resolve() if target_arg else None

            is_run_dir = is_resume_cmd or (target_path and target_path.is_dir() and (
                (target_path / "manifest.json").is_file() or (target_path / "recipe.toml").is_file() or (target_path / "summary.json").is_file()
            ))

            if is_run_dir:
                args.run_dir = target_path
                config_override = getattr(args, "config", None) if is_resume_cmd else None
                if config_override is None:
                    if (args.run_dir / "recipe.toml").is_file():
                        config_override = args.run_dir / "recipe.toml"
                    elif (args.run_dir / "model.toml").is_file():
                        config_override = args.run_dir / "model.toml"
                m = hg.load(config_override) if config_override else None
                if m and sets:
                    m = hg.override(m, sets)
                prepared = hg.prepare_resume(args.run_dir, getattr(args, "checkpoint", None), m, steps=getattr(args, "steps", None), sets=sets, **options)
            else:
                config_path = target_path
                if args.run_dir is None:
                    ws_path = hg.find_workspace(config_path)
                    if ws_path:
                        ws_data = hg.load_workspace(ws_path)
                        ws_sec = ws_data.get("workspace", {})
                        if "run_dir" in ws_sec:
                            run_dir_str = hg.interpolate_variables(ws_sec["run_dir"], ws_data, {"stem": config_path.stem, "name": config_path.stem})
                            args.run_dir = Path(run_dir_str).resolve()
                        elif "runs_root" in ws_sec:
                            args.run_dir = (Path(ws_sec["runs_root"]) / f"train-{config_path.stem}").resolve()
                        else:
                            args.run_dir = (Path.cwd() / f"train-{config_path.stem}").resolve()
                    else:
                        args.run_dir = (Path.cwd() / f"train-{config_path.stem}").resolve()
                else:
                    args.run_dir = args.run_dir.resolve()

                m = hg.load(config_path)
                if sets:
                    m = hg.override(m, sets)

                run_spec = m.spec.get("run", {}) if isinstance(m.spec, dict) else {}
                ws_path = hg.find_workspace(config_path)
                ws_data = hg.load_workspace(ws_path) if ws_path else {}

                needs_reexec, target_py, target_srcs, target_envs = _check_reexec(run_spec, ws_data)
                if needs_reexec:
                    if os.environ.get("_HYPERGAN_REEXEC") == "1":
                        unsatisfied = []
                        if target_py:
                            tp = Path(target_py)
                            venv_cfg = tp.parent.parent / "pyvenv.cfg"
                            if venv_cfg.is_file():
                                if Path(sys.prefix).resolve() != tp.parent.parent.resolve():
                                    unsatisfied.append(f"python venv (current: {sys.prefix}, target: {tp.parent.parent})")
                            else:
                                if Path(sys.executable).resolve() != tp.resolve():
                                    unsatisfied.append(f"python (current: {sys.executable}, target: {target_py})")
                        for k, v in target_envs.items():
                            if os.environ.get(k) != str(v):
                                unsatisfied.append(f"env var {k} (current: {os.environ.get(k)!r}, target: {v!r})")
                        resolved_head = [str(Path(x).resolve()) for x in sys.path[:len(target_srcs) + 3] if x]
                        for p in target_srcs:
                            if str(Path(p).resolve()) not in resolved_head:
                                unsatisfied.append(f"src path {p} not in sys.path")
                        raise RuntimeError(
                            f"Re-exec loop detected: process was already re-executed with _HYPERGAN_REEXEC=1, "
                            f"but target environment conditions remain unsatisfied: {', '.join(unsatisfied)}"
                        )

                    args.run_dir.mkdir(parents=True, exist_ok=True)
                    from .provenance import write_launch_provenance, snapshot_config_sources
                    from .model_file import collect_source_files

                    sources = collect_source_files(config_path, ws_path.parent if ws_path else None)
                    snapshot_config_sources(args.run_dir, sources)

                    attempt = _determine_attempt(args.run_dir)
                    meta = {
                        "argv": list(sys.argv),
                        "executable": target_py or sys.executable,
                        "run": run_spec,
                        "environment": target_envs,
                    }
                    if target_srcs:
                        meta["src_paths"] = target_srcs
                        meta["worktree"] = target_srcs[0]
                    write_launch_provenance(args.run_dir, attempt=attempt, metadata=meta)

                    lowered = hg.lower(m)
                    from .toml_writer import dumps as _dumps
                    recipe_text = _dumps(lowered, table_depth=3)
                    (args.run_dir / "recipe.toml").write_text(recipe_text, encoding="utf-8")

                    new_env = os.environ.copy()
                    new_env["_HYPERGAN_REEXEC"] = "1"
                    new_env["_HYPERGAN_REEXEC_ATTEMPT"] = f"{attempt:04d}"
                    for k, v in target_envs.items():
                        new_env[k] = str(v)
                    if target_srcs:
                        cur_pp = new_env.get("PYTHONPATH", "")
                        prefix = ":".join(target_srcs)
                        new_env["PYTHONPATH"] = f"{prefix}:{cur_pp}" if cur_pp else prefix

                    target_executable = target_py or sys.executable
                    new_argv = _reconstruct_argv_for_reexec(sys.argv, str(config_path), str(args.run_dir / "recipe.toml"), str(args.run_dir))
                    os.execve(target_executable, [target_executable, "-m", "hypergan", *new_argv], new_env)

                prepared = hg.prepare(m, args.run_dir, args.steps, **options)

            _warnings(prepared.config)
            from .metrics import manual_evaluation_hint, manual_evaluation_reminder

            labels = {"run": str(args.run_dir),
                      "config_path": str(getattr(args, "config", None) or args.run_dir)}
            hint = manual_evaluation_hint(prepared.config, **labels)
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
            output.configure(args.run_dir, progress_every=args.progress_every,
                             evaluation_reminder=manual_evaluation_reminder(prepared.config, **labels))
            config_dict = getattr(prepared.config, 'spec', prepared.config)
            with _training_viewer(args, config=config_dict):
                if not getattr(args, "progress_json", False):
                    exp_name = config_dict.get("name") if isinstance(config_dict, dict) else (target_path.stem if target_path else args.run_dir.name)
                    train_sec = config_dict.get("train", {}) if isinstance(config_dict, dict) else {}
                    dev = train_sec.get("device", "cuda")
                    dashboard_url = getattr(args, "_dashboard_url", None)
                    _print_training_header(args.run_dir, exp_name or str(args.run_dir.name), dev, dashboard_url)
                human_footer = (not is_resume_cmd) and not getattr(args, "progress_json", False)
                output.result(prepared.run(on_event=output.progress), run_dir=args.run_dir, human=human_footer)
        elif args.command == "sample":
            s = hg.sample(args.run_dir, count=args.count, seed=args.seed, output=args.output,
                          checkpoint=getattr(args, "checkpoint", None))
            if getattr(args, "sampler", None):
                view_obj = hg.view(s, args.sampler, run=args.run_dir)
                if hasattr(view_obj, "save") and args.output:
                    view_obj.save(args.output)
            print(s.path)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "particlegan", "numpy"}:
            print("error: training dependencies are missing; install 'hypergan[train]' in this environment", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
