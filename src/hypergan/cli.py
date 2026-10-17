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
                            help=argparse.SUPPRESS)
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


_add_run_options = _run_options


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
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{new,train,resume,ps,status,logs,stop,runs,sample,evaluate,eval,export,view,queue,model,validate,recipes,tag,rm,gc,prune}",
    )
    commands.add_parser("version")
    commands.add_parser("recipes", help="List available recipes and qualification status")
    new = commands.add_parser("new", help="Create a starter project or model .toml configuration (does not start training)")
    new.add_argument("path", nargs="?", type=Path, default=None, help="destination .toml file, project directory, or dataset directory (when -o is given)")
    new.add_argument("-l", "--list", dest="list_templates", action="store_true", help="list available starter templates and examples")
    new.add_argument("-t", "--template", default=None, choices=("dcgan", "tiny-transformer", "resnet", "gaussian-grid"),
                     help="starter architecture template (default: dcgan with --dataset/--resolution, else gaussian-grid)")
    new.add_argument("-r", "--resolution", type=int, default=None, choices=(32, 64, 128, 256),
                     help="target image resolution (32, 64, 128, 256)")
    new.add_argument("-o", "--output", type=Path, default=None, help="output .toml file path")
    new.add_argument("-p", "--project", type=Path, default=None, help="initialize a full HyperGAN project directory with hypergan.toml, runs/, and configs/")
    new.add_argument("--device", default="cuda", help="training device: cuda (default), cuda:N, or explicit cpu")
    new.add_argument("--dataset", type=Path, default=None, help="directory of images to size and generate recipe for")
    validate = commands.add_parser("validate", help="Validate a project without loading training dependencies")
    validate.add_argument("path", type=Path)
    validate.add_argument("--runtime", action="store_true",
                          help="also construct the recipe in bounded CPU/CUDA workers (requires train extra)")
    validate.add_argument("--profile", type=Path, default=None,
                          help="execution-profile TOML file for --runtime preflight")
    validate.add_argument("--data", action="store_true",
                          help="verify dataset directory and emit data inventory manifest")
    validate.add_argument("--output", type=Path, default=None,
                          help="write the data manifest to a new file (when --data is passed)")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--profile", type=Path,
                           help="execution-profile TOML file; omitted uses the configured native device")
    preflight.add_argument("--runtime", action="store_true",
                           help="also construct the recipe in bounded CPU/CUDA workers (requires train extra)")
    data_check = commands.add_parser("data-check")
    data_check.add_argument("config", type=Path)
    data_check.add_argument("--output", type=Path, help="write the data manifest to a new file")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or manifest path")
    ps_cmd = commands.add_parser("ps", help="List active training runs, GPU VRAM, and queued jobs (or -a for all runs)")
    ps_cmd.add_argument("-a", "--all", dest="all_runs", action="store_true", help="show all runs (default shows only active runs)")
    ps_cmd.add_argument("-q", "--quiet", action="store_true", help="only display short run IDs")
    ps_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    status_cmd = commands.add_parser("status", help="Show a concise, human-friendly status summary of a run")
    status_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name (default: active or latest run)")
    status_cmd.add_argument("--json", action="store_true", help="output structured status as JSON")
    logs_cmd = commands.add_parser("logs", help="View or follow (-f) live formatted training logs for a run")
    logs_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name (default: active or latest run)")
    logs_cmd.add_argument("-f", "--follow", action="store_true", help="follow live log output (Ctrl+C detaches without stopping training)")
    logs_cmd.add_argument("-n", "--tail", type=int, default=50, help="number of recent log lines to show (default: 50)")
    stop_cmd = commands.add_parser("stop", help="Gracefully stop an active training run (saves checkpoint and halts)")
    stop_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help="active run directory, slug, short ID, or name (default: active run)")
    stop_cmd.add_argument("--all", action="store_true", help="gracefully stop all active training runs")
    stop_cmd.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for graceful checkpoint shutdown (default: 15)")
    stop_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    tag_cmd = commands.add_parser("tag", help="Rename or tag a training run in its manifest and global registry")
    tag_cmd.add_argument("run_dir", type=Path, help="run directory, slug, short ID, or name")
    tag_cmd.add_argument("name", help="new name or tag for the run")
    tag_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    rm_cmd = commands.add_parser("rm", help="Remove a training run directory and its registry entry")
    rm_cmd.add_argument("run_dir", type=Path, help="run directory, slug, short ID, or name to remove")
    rm_cmd.add_argument("-f", "--force", action="store_true", help="force removal (stops run first if active)")
    rm_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    runs = commands.add_parser("runs", help="Inspect and compare experiment runs")
    runs.add_argument("--sort", choices=("fid", "best", "step", "updated", "name"), default="updated", help="sort order")
    runs.add_argument("--status", choices=("running", "stopped", "completed", "failed"), default=None, help="filter by status")
    runs.add_argument("--json", action="store_true", help="output runs table as JSON array")
    runs_subs = runs.add_subparsers(dest="runs_command", required=False)
    runs_ls = runs_subs.add_parser("ls", help="List runs under root directory")
    runs_ls.add_argument("root", nargs="?", type=Path, default=None, help="runs root directory")
    runs_ls.add_argument("--sort", choices=("fid", "best", "step", "updated", "name"), default="updated", help="sort order")
    runs_ls.add_argument("--status", choices=("running", "stopped", "completed", "failed"), default=None, help="filter by status")
    runs_ls.add_argument("--json", action="store_true", help="output runs table as JSON array")
    runs_compare = runs_subs.add_parser("compare", help="Compare multiple runs side-by-side")
    runs_compare.add_argument("runs", nargs="+", type=Path, help="two or more run directories, slugs, or IDs")
    runs_compare.add_argument("--metric", required=True, help="metric name to compare (e.g. fid50k_train, loss_g)")
    runs_compare.add_argument("--at", required=True, help="steps to compare at (e.g. 50k,100k,best,latest)")
    runs_compare.add_argument("--csv", action="store_true", help="export comparison as CSV")
    gc_cmd = commands.add_parser("gc", aliases=["prune"], help="Clean rebuildable views and compact events in finished runs")
    gc_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, or ID to clean")
    gc_cmd.add_argument("--all", action="store_true", help="clean all runs in workspace runs root")
    gc_cmd.add_argument("--dry-run", action="store_true", help="calculate reclaimable space without modifying files")
    gc_cmd.add_argument("--stride", type=_positive_int, default=10, help="compaction stride for routine train events (default: 10)")
    model = commands.add_parser("model", help="Describe a run's formulation, losses, optimizers and networks")
    model.add_argument("path", nargs="?", type=Path, default=None, help="run directory, slug, short ID, manifest.json or project .toml")
    model.add_argument("--networks", action="store_true",
                       help="also build each component on the meta device for per-layer detail (loads torch)")
    model.add_argument("--write", action="store_true",
                       help="record the per-layer detail as RUN/model.json for the viewer (implies --networks)")
    events = commands.add_parser("events")
    events.add_argument("run_dir", nargs="?", type=Path, default=None)
    events.add_argument("--cursor", help="opaque cursor returned by the previous page")
    events.add_argument("--limit", type=_positive_int, default=100)
    events.add_argument("--max-bytes", type=_positive_int, default=1048576)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("run_dir", nargs="?", type=Path, default=None)
    metrics.add_argument("--revision", help="historical catalog SHA256 (default: active catalog)")
    project = commands.add_parser("project")
    project.add_argument("run_dir", nargs="?", type=Path, default=None)
    project.add_argument("--map-spec", type=Path, help="JSON MapSpec; default selects published scalar metrics")
    project.add_argument("--follow", action="store_true", help="continue projecting newly completed events")
    project.add_argument("--limit", type=_positive_int, default=100, help="documents per bounded work page")
    contributions = commands.add_parser("contributions")
    contributions.add_argument("run_dir", nargs="?", type=Path, default=None)
    contributions.add_argument("--map-revision", help="map SHA256 (default: built-in scalar map)")
    contributions.add_argument("--cursor", help="last applied projection cursor")
    contributions.add_argument("--limit", type=_positive_int, default=100)
    contributions.add_argument("--max-bytes", type=_positive_int, default=1048576)
    serve = commands.add_parser("serve")
    serve.add_argument("run_dir", nargs="?", type=Path, default=None)
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
    server_status = commands.add_parser("server-status")
    server_status.add_argument("run_dir", nargs="?", type=Path, default=None)
    stop_server = commands.add_parser("stop-server")
    stop_server.add_argument("run_dir", type=Path, nargs="?", default=None, help="run or root directory to stop")
    stop_server.add_argument("--all", action="store_true", help="safely kill all legacy leaked viewers and projectors across the host")
    from .evaluation_cli import add_evaluate_parser
    add_evaluate_parser(commands)
    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name (default: active run)")
    operation = checkpoint.add_mutually_exclusive_group()
    operation.add_argument("--request-id", help="reuse this ID to retry the same request safely")
    operation.add_argument("--status", metavar="REQUEST_ID", help="read a request receipt without submitting")
    checkpoint.add_argument("--attempt-id", help="target this attempt (default: current manifest attempt)")
    checkpoints = commands.add_parser("checkpoints")
    checkpoints_subs = checkpoints.add_subparsers(dest="checkpoints_command", required=True)
    prune = checkpoints_subs.add_parser("prune", help="Prune obsolete checkpoints under the retention policy")
    prune.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name")
    prune.add_argument("--dry-run", action="store_true", help="evaluate retention without deleting any files")
    export = commands.add_parser("export", help="Export a standalone EMA inference model bundle from a run checkpoint")
    export.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name")
    export.add_argument("--checkpoint", default="latest", help="target checkpoint: best, latest, step number, or path")
    export.add_argument("-o", "--output", type=Path, help="output file or directory (default: <run_dir>/model.pt)")
    convert = commands.add_parser("convert")
    convert.add_argument("recipe", type=Path, help="path to legacy recipe TOML")
    convert.add_argument("--output", type=Path, required=True, help="output model.toml destination")
    def _add_train_and_resume_arguments(p, *, is_resume=False):
        if is_resume:
            p.add_argument("run_dir", nargs="?", type=Path, default=None, help="existing run directory, slug, short ID, or name to resume")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="explicit run directory override")
            p.add_argument("--config", type=Path, default=None, help="verify exact compatibility with this configuration")
        else:
            p.add_argument("config", type=Path, help="configuration file or existing run directory/slug/ID to resume")
            p.add_argument("-n", "--name", default=None, help="custom experiment name / slug for the run")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="new run directory, or an existing run to resume")
            p.add_argument("--config", type=Path, default=None, help="explicit configuration file override")
        p.add_argument("-d", "--devices", default=None,
                       help="device(s) to train on: 0, 0,1, gpu0,gpu1, auto, cpu")
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
    sample.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name (default: active or latest run)")
    sample.add_argument("--count", type=_positive_int, default=16)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", type=Path)
    sample.add_argument("--sampler", help="sampler name declared in [samplers], or module:function import path")
    sample.add_argument("--checkpoint", default=None, help="target checkpoint: best, latest, step number, or path")

    queue_cmd = commands.add_parser("queue", help="Multi-GPU job queue scheduler, runner daemon, and service generator")
    queue_cmd.add_argument("--json", action="store_true", help="output JSON format")
    queue_cmd.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")
    queue_subs = queue_cmd.add_subparsers(dest="queue_command", required=False)

    q_add = queue_subs.add_parser("add", help="Enqueue a training experiment or wait directive")
    q_add.add_argument("model_file", type=Path, help="path to experiment model TOML (or run directory if --wait)")
    q_add.add_argument("-d", "--devices", default=None, help="target devices (e.g. gpu0, gpu0,gpu1, 0, 0,1)")
    q_add.add_argument("--run-dir", type=Path, default=None, help="explicit run directory")
    q_add.add_argument("--name", default=None, help="optional custom job name")
    q_add.add_argument("--wait", action="store_true", help="wait for target run directory to complete before proceeding")
    q_add.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")

    q_ls = queue_subs.add_parser("ls", help="List jobs from the queue")
    q_ls.add_argument("--status", choices=("queued", "running", "completed", "failed", "cancelled"), default=None, help="filter by status")
    q_ls.add_argument("--json", action="store_true", help="output JSON format")
    q_ls.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")

    q_status = queue_subs.add_parser("status", help="Inspect job details, progress, and live log tail")
    q_status.add_argument("job_id", nargs="?", default=None, help="job ID to inspect (default: active/latest job)")
    q_status.add_argument("--lines", type=int, default=20, help="number of log tail lines (default: 20)")
    q_status.add_argument("--json", action="store_true", help="output JSON format")
    q_status.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")

    q_stop = queue_subs.add_parser("stop", help="Cancel queued jobs or signal running job to halt")
    q_stop.add_argument("job_id", nargs="?", default=None, help="job ID to stop")
    q_stop.add_argument("--all", action="store_true", help="cancel all queued jobs and halt queue runner")
    q_stop.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")

    q_run = queue_subs.add_parser("run", help="Start the sequential queue daemon runner")
    q_run.add_argument("-d", "--devices", default=None, help="restrict runner to specific device(s)")
    q_run.add_argument("--poll-interval", type=float, default=5.0, help="seconds between queue checks (default: 5.0)")
    q_run.add_argument("--once", action="store_true", help="execute available jobs and exit instead of continuous loop")
    q_run.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")

    q_svc = queue_subs.add_parser("install-service", help="Generate systemd user service unit")
    q_svc.add_argument("--user", action="store_true", default=True, help="install as user service in ~/.config/systemd/user (default: true)")
    q_svc.add_argument("-d", "--devices", default=None, help="bind service runner to specific device(s)")
    q_svc.add_argument("--name", default=None, help="custom service name")
    q_svc.add_argument("--dry-run", action="store_true", help="print generated unit file without writing to disk")

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


def _print_training_header(run_dir: Path, experiment_name: str, device: str, dashboard_url: str | None = None):
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
            target_p = hg.resolve_run_target(None) if args.path is None else args.path
            run = hg.open_run(target_p)
            _print_json(run.manifest)
        elif args.command == "ps":
            ps_data = hg.ps(all_runs=args.all_runs)
            if args.json:
                _print_json(ps_data)
            else:
                from .inspection import format_ps_table
                out_str = format_ps_table(ps_data, all_runs=args.all_runs, quiet=args.quiet)
                if out_str:
                    print(out_str)
        elif args.command == "status":
            res = hg.status(args.run_dir)
            if args.json:
                _print_json(res)
            else:
                from .inspection import format_run_status
                print(format_run_status(res))
        elif args.command == "logs":
            hg.logs(args.run_dir, tail=args.tail, follow=args.follow)
        elif args.command == "stop":
            res = hg.stop(args.run_dir, stop_all=args.all, timeout=args.timeout)
            if args.json:
                _print_json(res)
            else:
                print(res.get("message", f"Stopped {res.get('short_id', '')}"))
        elif args.command == "tag":
            from . import inspection
            if hasattr(inspection, "tag_run"):
                res = dict(inspection.tag_run(args.run_dir, args.name))
            else:
                res = hg.tag(args.run_dir, args.name)
            run_id_short = str(res.get("short_id") or res.get("run_id", ""))[:8]
            r_name = res.get("name", args.name)
            r_dir = res.get("run_dir", str(args.run_dir))
            res.setdefault("short_id", run_id_short)
            res.setdefault("message", f"Tagged {run_id_short} as '{r_name}' ({r_dir})")
            if getattr(args, "json", False) or not sys.stdout.isatty():
                _print_json(res)
            else:
                print(f"Tagged {run_id_short} as '{r_name}' ({r_dir})")
        elif args.command == "rm":
            from . import inspection
            if hasattr(inspection, "remove_run"):
                res = dict(inspection.remove_run(args.run_dir, force=args.force))
            else:
                res = hg.rm(args.run_dir, force=args.force)
            run_id_short = str(res.get("short_id") or res.get("run_id", ""))[:8]
            r_dir = res.get("run_dir", str(args.run_dir))
            r_name = res.get("name", Path(r_dir).name if r_dir else "")
            res.setdefault("short_id", run_id_short)
            res.setdefault("message", f"Removed run '{r_name}' ({run_id_short}) at {r_dir}")
            if getattr(args, "json", False) or not sys.stdout.isatty():
                _print_json(res)
            else:
                print(f"Removed run '{r_name}' ({run_id_short}) at {r_dir}")
        elif args.command == "runs":
            if getattr(args, "runs_command", None) is None:
                args.runs_command = "ls"
                if not hasattr(args, "root"):
                    args.root = None
                if not hasattr(args, "sort") or args.sort is None:
                    args.sort = "updated"
                if not hasattr(args, "status"):
                    args.status = None
                if not hasattr(args, "json"):
                    args.json = False
            if args.runs_command == "ls":
                root_arg = getattr(args, "root", None)
                runs_list = hg.runs_ls(root_arg, sort=args.sort, status_filter=args.status)
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
        elif args.command in {"gc", "prune"}:
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
                target_run = hg.resolve_run_target(args.run_dir)
                res = hg.gc(target_run, dry_run=args.dry_run, stride=args.stride)
                from .inspection import format_bytes
                views_str = format_bytes(res.get("views_reclaimed_bytes", 0))
                events_str = format_bytes(res.get("events_reclaimed_bytes", 0))
                total_str = format_bytes(res.get("reclaimed_bytes", 0))
                prefix = "Would clean" if args.dry_run else "Cleaned"
                print(f"{prefix} run: {res.get('run_dir', target_run)}")
                print(f"  Views removed:     {views_str}")
                print(f"  Events compacted:  {events_str}")
                print(f"  Total reclaimed:   {total_str}")
        elif args.command == "model":
            resolved_model_path = args.path
            if resolved_model_path is None:
                resolved_model_path = hg.resolve_run_target(None)
            elif not Path(resolved_model_path).exists() and not str(resolved_model_path).endswith(".toml"):
                try:
                    resolved_model_path = hg.resolve_run_target(resolved_model_path)
                except Exception:
                    pass
            desc = hg.describe(resolved_model_path, networks=args.networks, write=args.write)
            if args.write:
                from .model_description import MODEL_FILE
                print(f"wrote {Path(resolved_model_path) / MODEL_FILE}", file=sys.stderr)
            _print_json(desc)
        elif args.command == "events":
            _print_json(hg.events(args.run_dir, cursor=args.cursor, limit=args.limit, max_bytes=args.max_bytes))
        elif args.command == "metrics":
            _print_json(hg.catalog(args.run_dir, revision=args.revision))
        elif args.command in {"evaluate", "eval"}:
            from .evaluation_cli import run_evaluate
            _print_json(run_evaluate(args))
        elif args.command == "serve":
            hg.serve(hg.resolve_run_target(args.run_dir), port=args.port, host=args.host, auth=args.auth,
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
                if getattr(args, "list_templates", False):
                    from .recipe_generator import format_starter_templates
                    print(format_starter_templates())
                    return 0
                proj_dir = getattr(args, "project", None)
                tmpl = getattr(args, "template", None)
                res_opt = getattr(args, "resolution", None)
                out_opt = getattr(args, "output", None)
                if proj_dir is not None:
                    dataset_arg = args.dataset if args.dataset is not None else args.path
                    info = hg.init_project(
                        proj_dir,
                        template=tmpl or "dcgan",
                        resolution=res_opt,
                        data_dir=dataset_arg,
                        output_config=out_opt,
                        device=args.device,
                    )
                    print(f"Initialized HyperGAN project in {info['project_dir']}")
                    print(f"  Workspace config: {info['workspace_config']}")
                    print(f"  Starter recipe:   {info['recipe_path']}")
                    print("Next steps:")
                    for step_cmd in info["next_steps"]:
                        print(f"  {step_cmd}")
                    return 0
                if out_opt is not None and args.path is not None and args.dataset is None:
                    dataset_arg = args.path
                    target_dest = out_opt
                else:
                    dataset_arg = args.dataset
                    target_dest = out_opt if out_opt is not None else args.path
                if target_dest is None:
                    raise ValueError("path is required (or pass `hypergan new --list` to see starter templates)")
                if dataset_arg is not None or (tmpl is not None and tmpl != "gaussian-grid") or res_opt is not None:
                    from .recipe_generator import generate_recipe
                    dest = target_dest if target_dest.suffix == ".toml" else (target_dest / "model.toml" if target_dest.is_dir() else target_dest.with_suffix(".toml"))
                    name = target_dest.stem
                    saved = generate_recipe(
                        name,
                        dataset_arg,
                        dest,
                        device=args.device,
                        template=tmpl or "dcgan",
                        resolution=res_opt,
                    )
                    print(saved)
                else:
                    from .config import validate_device
                    validate_device(args.device)
                    m = hg.packaged("gaussian-grid")
                    if args.device:
                        m = hg.override(m, {"train.device": args.device})
                    dest = target_dest / "model.toml" if (target_dest.is_dir() or target_dest.suffix != ".toml") else target_dest
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
                do_runtime = getattr(args, "runtime", False)
                do_data = getattr(args, "data", False)
                if do_runtime and do_data:
                    resolved_cfg = dict(hg.resolve(m))
                    resolved_cfg["preflight"] = hg.preflight_check(m, profile=getattr(args, "profile", None), runtime=True)
                    resolved_cfg["data_check"] = hg.check_data(m, output=getattr(args, "output", None))
                    _print_json(resolved_cfg)
                elif do_runtime:
                    _print_json(hg.preflight_check(m, profile=getattr(args, "profile", None), runtime=True))
                elif do_data:
                    _print_json(hg.check_data(m, output=getattr(args, "output", None)))
                else:
                    _print_json(hg.resolve(m))
        elif args.command in {"train", "resume"}:
            from .execution import SERVICE_TIMEOUTS
            from .registry import is_run_dir as _is_valid_run_dir
            options = dict(profile=args.profile,
                           service_policy={name: getattr(args, name) for name in SERVICE_TIMEOUTS
                                           if getattr(args, name) is not None},
                           devices=getattr(args, "devices", None),
                           checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
                           stop_after_steps=args.stop_after_steps,
                           preview_every=args.preview_every, preview_keep=args.preview_keep,
                           preview_name=args.preview_name)
            sets = _parse_set_overrides(args.sets) if getattr(args, "sets", None) else {}
            is_resume_cmd = args.command == "resume"
            target_arg = getattr(args, "run_dir", None) if is_resume_cmd else getattr(args, "config", None)
            if target_arg is None:
                target_arg = getattr(args, "config", None) or getattr(args, "run_dir", None)

            if is_resume_cmd:
                target_path = hg.resolve_run_target(target_arg)
                is_run_dir = True
            else:
                target_path = target_arg.resolve() if target_arg else None
                is_run_dir = bool(target_path and target_path.is_dir() and (
                    (target_path / "manifest.json").is_file() or (target_path / "recipe.toml").is_file() or (target_path / "summary.json").is_file()
                ))
                if not is_run_dir and target_arg is not None and not target_path.exists() and not str(target_arg).endswith(".toml"):
                    try:
                        cand_resolved = hg.resolve_run_target(target_arg)
                        if _is_valid_run_dir(cand_resolved):
                            target_path = cand_resolved
                            is_run_dir = True
                    except Exception:
                        pass

            if is_run_dir:
                args.run_dir = target_path
                config_override = getattr(args, "config", None) if is_resume_cmd else None
                manifest_p = args.run_dir / "manifest.json"
                manifest_data = {}
                if manifest_p.is_file():
                    try:
                        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                needs_stored_cfg = bool(
                    getattr(args, "steps", None) is not None
                    or sets
                    or getattr(args, "eval_every", None) is not None
                    or getattr(args, "eval_metrics", None) is not None
                    or not manifest_data
                )
                if config_override:
                    m = hg.load(config_override)
                elif needs_stored_cfg:
                    try:
                        m = hg._load_stored_run_model(args.run_dir)
                    except Exception:
                        m = None
                else:
                    m = None
                if m and sets:
                    m = hg.override(m, sets)

                run_spec = {}
                if m and isinstance(m.spec, dict):
                    run_spec = m.spec.get("run", {})
                elif manifest_data and isinstance(manifest_data.get("config"), dict):
                    run_spec = manifest_data["config"].get("run", {})
                if not run_spec:
                    recipe_p = args.run_dir / "recipe.toml"
                    if recipe_p.is_file():
                        try:
                            import tomllib
                            r_data = tomllib.loads(recipe_p.read_text(encoding="utf-8"))
                            if isinstance(r_data, dict):
                                run_spec = r_data.get("run", {})
                        except Exception:
                            pass
                if not run_spec:
                    launch_dir = args.run_dir / "launch"
                    if launch_dir.is_dir():
                        l_files = sorted(launch_dir.glob("*.json"))
                        if l_files:
                            try:
                                l_obj = json.loads(l_files[-1].read_text(encoding="utf-8"))
                                run_spec = l_obj.get("run", {})
                            except Exception:
                                pass

                ws_path = hg.find_workspace(args.run_dir)
                if not ws_path:
                    launch_dir = args.run_dir / "launch"
                    if launch_dir.is_dir():
                        l_files = sorted(launch_dir.glob("*.json"))
                        for lf in reversed(l_files):
                            try:
                                l_obj = json.loads(lf.read_text(encoding="utf-8"))
                                for a in l_obj.get("argv", []):
                                    if a.endswith(".toml") and Path(a).is_file():
                                        cand_ws = hg.find_workspace(Path(a))
                                        if cand_ws:
                                            ws_path = cand_ws
                                            break
                                if ws_path:
                                    break
                            except Exception:
                                pass
                if not ws_path and os.environ.get("HYPERGAN_WORKSPACE"):
                    ws_path = Path(os.environ["HYPERGAN_WORKSPACE"])
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

                    attempt = _determine_attempt(args.run_dir)
                    new_env = os.environ.copy()
                    new_env["_HYPERGAN_REEXEC"] = "1"
                    new_env["_HYPERGAN_REEXEC_ATTEMPT"] = f"{attempt:04d}"
                    if ws_path:
                        new_env["HYPERGAN_WORKSPACE"] = str(ws_path)
                    for k, v in target_envs.items():
                        new_env[k] = str(v)
                    if target_srcs:
                        cur_pp = new_env.get("PYTHONPATH", "")
                        prefix = ":".join(target_srcs)
                        new_env["PYTHONPATH"] = f"{prefix}:{cur_pp}" if cur_pp else prefix

                    target_executable = target_py or sys.executable
                    raw_args = list(sys.argv)
                    if len(raw_args) >= 3 and raw_args[1] == "-m" and raw_args[2] == "hypergan":
                        exec_argv = [target_executable, "-m", "hypergan", *raw_args[3:]]
                    elif "hypergan" in Path(raw_args[0]).name:
                        exec_argv = [target_executable, "-m", "hypergan", *raw_args[1:]]
                    else:
                        exec_argv = [target_executable, *raw_args[1:]]
                    os.execve(target_executable, exec_argv, new_env)

                prepared = hg.prepare_resume(args.run_dir, getattr(args, "checkpoint", None), m, steps=getattr(args, "steps", None), sets=sets, **options)
            else:
                config_path = target_path
                custom_name = getattr(args, "name", None)
                if args.run_dir is None:
                    ws_path = hg.find_workspace(config_path)
                    if ws_path:
                        ws_data = hg.load_workspace(ws_path)
                        ws_sec = ws_data.get("workspace", {})
                        runs_root = Path(ws_sec["runs_root"]).resolve() if "runs_root" in ws_sec else Path.cwd().resolve()
                        slug_dir_name, short_slug = (custom_name, custom_name) if custom_name else hg.generate_run_slug(config_path.stem, parent_dir=runs_root)
                        if "run_dir" in ws_sec:
                            run_dir_str = hg.interpolate_variables(
                                ws_sec["run_dir"],
                                ws_data,
                                {"stem": config_path.stem, "name": slug_dir_name, "slug": short_slug},
                            )
                            args.run_dir = Path(run_dir_str).resolve()
                        else:
                            args.run_dir = (runs_root / slug_dir_name).resolve()
                    else:
                        slug_dir_name, short_slug = (custom_name, custom_name) if custom_name else hg.generate_run_slug(config_path.stem, parent_dir=Path.cwd())
                        args.run_dir = (Path.cwd() / slug_dir_name).resolve()
                else:
                    args.run_dir = args.run_dir.resolve()

                m = hg.load(config_path)
                if custom_name:
                    m = hg.override(m, {"name": custom_name})
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
                    if ws_path:
                        new_env["HYPERGAN_WORKSPACE"] = str(ws_path)
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
                is_tty = bool(getattr(output.stdout.original, "isatty", lambda: False)()) if output and hasattr(output, "stdout") else False
                human_footer = (not is_resume_cmd or is_tty) and not getattr(args, "progress_json", False)
                output.result(prepared.run(on_event=output.progress), run_dir=args.run_dir, human=human_footer)
        elif args.command == "sample":
            s = hg.sample(args.run_dir, count=args.count, seed=args.seed, output=args.output,
                          checkpoint=getattr(args, "checkpoint", None), default_image_png=True)
            if getattr(args, "sampler", None):
                view_obj = hg.view(s, args.sampler, run=args.run_dir)
                if hasattr(view_obj, "save") and args.output:
                    view_obj.save(args.output)
            print(s.path)
        elif args.command == "queue":
            from .queue import format_queue_table, format_queue_status
            if getattr(args, "queue_command", None) is None:
                args.queue_command = "status"
                if not hasattr(args, "job_id"):
                    args.job_id = None
                if not hasattr(args, "lines"):
                    args.lines = 20
                if not hasattr(args, "json"):
                    args.json = False
                if not hasattr(args, "queue_file"):
                    args.queue_file = None

            if args.queue_command == "add":
                res = hg.queue_add(
                    args.model_file,
                    devices=args.devices,
                    run_dir=args.run_dir,
                    name=args.name,
                    wait=args.wait,
                    queue_file=getattr(args, "queue_file", None),
                )
                print(f"Enqueued {res['id']}: {res['name']} (devices: {res.get('devices') or 'default'}, status: {res['status']})")
            elif args.queue_command == "ls":
                jobs = hg.queue_ls(status=args.status, queue_file=getattr(args, "queue_file", None))
                if getattr(args, "json", False):
                    _print_json(jobs)
                else:
                    print(format_queue_table(jobs))
            elif args.queue_command == "status":
                st = hg.queue_status(job_id=getattr(args, "job_id", None), queue_file=getattr(args, "queue_file", None), lines=getattr(args, "lines", 20))
                if getattr(args, "json", False):
                    _print_json(st)
                else:
                    print(format_queue_status(st))
            elif args.queue_command == "stop":
                res = hg.queue_stop(job_id=args.job_id, stop_all=args.all, queue_file=getattr(args, "queue_file", None))
                print(res["message"])
            elif args.queue_command == "run":
                return hg.queue_run(devices=args.devices, poll_interval=args.poll_interval, once=args.once, queue_file=getattr(args, "queue_file", None))
            elif args.queue_command == "install-service":
                path, instructions = hg.queue_install_service(user=args.user, devices=args.devices, name=args.name, dry_run=args.dry_run)
                print(instructions)
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
