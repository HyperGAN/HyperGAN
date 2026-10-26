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
    parser.add_argument("--profile", help="execution profile name or TOML path; new runs default to native device, existing runs inherit "
                                          "(multi-node profiles and the parameter_average/pseudo_gradient aggregation strategies are experimental)")
    from .execution import SERVICE_TIMEOUTS
    for name in SERVICE_TIMEOUTS:
        parser.add_argument("--" + name.replace("_", "-"), type=_positive_seconds,
                            help=argparse.SUPPRESS)
    server = parser.add_mutually_exclusive_group()
    server.add_argument("--server", action="store_true", help="require the local viewer before training starts")
    server.add_argument("--no-server", "--no-web", dest="no_server", action="store_true",
                        help="train without web imports or listening sockets")
    parser.add_argument("--port", "--server-port", dest="server_port", type=int,
                        help=f"require a specific viewer port (default: {DEFAULT_VIEWER_PORT}; an occupied "
                             "port is an error, never a search; 0 selects any free port)")
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


# How the run-management commands name a run (registry.resolve_destructive_target).
_RUN_TARGET = "run ID, short ID, slug, tag, run directory name, or path"

# Deprecated top-level command spellings: old -> (current command, warning).
# They still run for one release, warn on stderr and are not listed in --help.
_DEPRECATED_COMMANDS = {
    "prune": ("gc", "warning: `hypergan prune` is deprecated and will be removed; use `hypergan gc` "
                    "(views and events), or `hypergan checkpoints prune` for checkpoint retention"),
}


class _CustomParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        args = list(sys.argv[1:] if args is None else args)
        # Global options are flags without values, so the first positional is the command.
        index = next((i for i, arg in enumerate(args) if not arg.startswith("-")), None)
        if index is not None:
            command = args[index]
            if command in _DEPRECATED_COMMANDS:
                current, warning = _DEPRECATED_COMMANDS[command]
                print(warning, file=sys.stderr)
                args[index] = current
            elif command == "checkpoint" and args[index + 1:index + 2] == ["prune"]:
                self.error("`hypergan checkpoint prune` is not a command; use `hypergan checkpoints prune`")
        return super().parse_known_args(args, namespace)


def _parser():
    parser = _CustomParser(
        prog="hypergan", description="Create, validate, and run reproducible GAN projects."
    )
    parser.add_argument("--version", action="version", version=f"hypergan {__version__}")
    parser.add_argument("--json", action="store_true", dest="global_json", help="output results as JSON")
    parser.add_argument("--debug", action="store_true",
                        help="print the full traceback when a command fails (same as HYPERGAN_DEBUG=1)")
    # Subcommand parsers are plain: only the top level maps deprecated spellings.
    commands = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)
    commands.add_parser("version", help="Print the HyperGAN version")
    recipes_cmd = commands.add_parser("recipes", help="List the packaged model recipes (JSON)")
    recipes_cmd.add_argument("--json", action="store_true", help="accepted for consistency; the list is always JSON")
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
    new.add_argument("--json", action="store_true", help="output created project info as JSON")
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
    validate.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                          help="override configuration value before validation (e.g. --set train.steps=500; VALUE null removes the key)")
    validate.add_argument("--json", action="store_true", help="output validation result as JSON")
    preflight = commands.add_parser("preflight", help="Check a configuration against an execution profile before training")
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--profile", type=Path,
                           help="execution-profile TOML file; omitted uses the configured native device")
    preflight.add_argument("--runtime", action="store_true",
                           help="also construct the recipe in bounded CPU/CUDA workers (requires train extra)")
    preflight.add_argument("--json", action="store_true", help="output preflight check as JSON")
    data_check = commands.add_parser("data-check", help="Verify a configuration's image dataset and write its inventory manifest")
    data_check.add_argument("config", type=Path)
    data_check.add_argument("--output", type=Path, help="write the data manifest to a new file")
    download_data = commands.add_parser(
        "download-data",
        help="Opt-in acquisition, checksum verification, and manifest preparation for qualified datasets",
    )
    download_data.add_argument("dataset", nargs="?", default="cifar10", help="qualified dataset identifier (default: cifar10)")
    download_data.add_argument("-o", "--output", type=Path, default=None, help="destination directory for extracted dataset and dataset-manifest.json")
    download_data.add_argument("--cache-dir", type=Path, default=None, help="archive download cache directory (default: <output>/.cache)")
    download_data.add_argument("--info", action="store_true", help="show source URL, version, expected bytes, checksums, and usage notice without downloading")
    download_data.add_argument("--list", dest="list_datasets", action="store_true", help="list qualified datasets available for download")
    download_data.add_argument("--verify-only", action="store_true", help="verify existing archive/extracted files without downloading")
    download_data.add_argument("--source-archive", type=Path, default=None, help="verify and extract an already-downloaded local archive file")
    download_data.add_argument("--url", default=None, help="override download URL (archive must still pass pinned size and SHA256 verification)")
    download_data.add_argument("--timeout", type=_positive_seconds, default=60.0, help="network timeout in seconds (default: 60)")
    download_data.add_argument("--json", action="store_true", help="output result as JSON")
    inspect = commands.add_parser("inspect", help="Inspect raw run manifest JSON")
    inspect.add_argument("path", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or manifest path")
    inspect.add_argument("--json", action="store_true", help="output manifest JSON")
    ps_cmd = commands.add_parser("ps", help="List active training runs, GPU VRAM, and queued jobs (or -a for all runs)")
    ps_cmd.add_argument("-a", "--all", dest="all_runs", action="store_true", help="show all runs (default shows only active runs)")
    ps_cmd.add_argument("-q", "--quiet", action="store_true", help="only display short run IDs")
    ps_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    status_cmd = commands.add_parser("status", help="Show a concise, human-friendly status summary of a run")
    status_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help=f"{_RUN_TARGET} (default: active or latest run)")
    status_cmd.add_argument("--json", action="store_true", help="output structured status as JSON")
    logs_cmd = commands.add_parser("logs", help="View or follow (-f) live formatted training logs for a run")
    logs_cmd.add_argument("run_dir", nargs="?", type=str, default=None, help=f"{_RUN_TARGET} (default: active or latest run)")
    logs_cmd.add_argument("-f", "--follow", action="store_true", help="follow live log output (Ctrl+C detaches without stopping training)")
    logs_cmd.add_argument("-n", "--tail", type=_positive_int, default=50, help="number of recent log lines to show (default: 50)")
    logs_cmd.add_argument("--json", action="store_true", help="output raw JSON lines")
    stop_cmd = commands.add_parser("stop", help="Gracefully stop an active training run (saves checkpoint and halts)")
    stop_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help=f"active run: {_RUN_TARGET} (default: active run)")
    stop_cmd.add_argument("--all", action="store_true", help="gracefully stop all active training runs")
    stop_cmd.add_argument("--timeout", type=_positive_seconds, default=15.0, help="seconds to wait for graceful checkpoint shutdown (default: 15)")
    stop_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    tag_cmd = commands.add_parser("tag", help="Rename or tag a training run in its manifest and global registry")
    tag_cmd.add_argument("run_dir", type=Path, help=_RUN_TARGET)
    tag_cmd.add_argument("name", help="new name or tag for the run")
    tag_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    rm_cmd = commands.add_parser("rm", help="Remove a training run directory and its registry entry")
    rm_cmd.add_argument("run_dir", type=Path, help=f"run to remove: {_RUN_TARGET}")
    rm_cmd.add_argument("-f", "--force", action="store_true", help="force removal (stops run first if active)")
    rm_cmd.add_argument("--json", action="store_true", help="output structured JSON")
    runs = commands.add_parser("runs", help="Inspect and compare experiment runs")
    runs.add_argument("--sort", choices=("fid", "best", "step", "updated", "name"), default="updated", help="sort order")
    runs.add_argument("--status", choices=("running", "stopped", "completed", "failed"), default=None, help="filter by status")
    runs.add_argument("--json", action="store_true", default=False, help="output runs table as JSON array")
    runs_subs = runs.add_subparsers(dest="runs_command", required=False)
    runs_ls = runs_subs.add_parser("ls", help="List runs under root directory")
    runs_ls.add_argument("root", nargs="?", type=Path, default=None, help="runs root directory")
    runs_ls.add_argument("--sort", choices=("fid", "best", "step", "updated", "name"), default="updated", help="sort order")
    runs_ls.add_argument("--status", choices=("running", "stopped", "completed", "failed"), default=None, help="filter by status")
    runs_ls.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output runs table as JSON array")
    runs_compare = runs_subs.add_parser("compare", help="Compare multiple runs side-by-side")
    runs_compare.add_argument("runs", nargs="+", type=Path, help="two or more run directories, slugs, or IDs")
    runs_compare.add_argument("--metric", default="loss/g_total", help="metric name to compare (default: loss/g_total, or e.g. fid50k_train, loss_g)")
    runs_compare.add_argument("--at", default="latest", help="steps to compare at (default: latest, or e.g. 50k,100k,best,latest)")
    runs_compare.add_argument("--csv", action="store_true", help="export comparison as CSV")
    runs_compare.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output comparison as JSON")

    compare_cmd = commands.add_parser("compare", help="Compare multiple runs side-by-side (alias for runs compare)")
    compare_cmd.add_argument("runs", nargs="+", type=Path, help="two or more run directories, slugs, or IDs")
    compare_cmd.add_argument("--metric", default="loss/g_total", help="metric name to compare (default: loss/g_total, or e.g. fid50k_train, loss_g)")
    compare_cmd.add_argument("--at", default="latest", help="steps to compare at (default: latest, or e.g. 50k,100k,best,latest)")
    compare_cmd.add_argument("--csv", action="store_true", help="export comparison as CSV")
    compare_cmd.add_argument("--json", action="store_true", default=False, help="output comparison as JSON")

    diff_cmd = commands.add_parser("diff", help="Compare two recipes, runs, or checkpoints")
    diff_cmd.add_argument("target_a", type=str, help="first recipe TOML, run directory/slug/tag, or checkpoint")
    diff_cmd.add_argument("target_b", nargs="?", default=None, type=str, help="second recipe TOML, run directory/slug/tag, or checkpoint (optional when -s/--set is passed)")
    diff_cmd.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                          help="override configuration value on the target before comparing (VALUE null removes the key)")
    diff_cmd.add_argument("--json", action="store_true", help="output diff as JSON")

    explain_cmd = commands.add_parser("explain", help="Explain each network's role, inputs, optimizer group, and gradient sources")
    explain_cmd.add_argument("path", nargs="?", type=Path, default=None, help="model .toml, project directory, or run directory/slug/tag")
    explain_cmd.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                             help="override configuration value before explaining (e.g. --set train.lr=0.0005; VALUE null removes the key)")
    explain_cmd.add_argument("--json", action="store_true", help="output explanation as JSON")

    doctor_cmd = commands.add_parser("doctor", help="Inspect Python, PyTorch, CUDA devices, workspace, and runner health")
    doctor_cmd.add_argument("--json", action="store_true", help="output diagnostic report as JSON")

    report_cmd = commands.add_parser("report", help="Generate Markdown or JSON summary table of run metrics and artifacts")
    report_cmd.add_argument("runs", nargs="*", default=[], help="run directories, slugs, or tags (default: all registered runs)")
    report_cmd.add_argument("--json", action="store_true", help="output report as JSON")

    gc_cmd = commands.add_parser("gc", help="Clean rebuildable views and compact events in finished runs")
    gc_cmd.add_argument("run_dir", nargs="?", type=Path, default=None, help=f"run to clean: {_RUN_TARGET}")
    gc_cmd.add_argument("--all", action="store_true", help="clean all runs in workspace runs root")
    gc_cmd.add_argument("--dry-run", action="store_true", help="calculate reclaimable space without modifying files")
    gc_cmd.add_argument("--stride", type=_positive_int, default=10, help="compaction stride for routine train events (default: 10)")
    gc_cmd.add_argument("--json", action="store_true", help="output cleanup statistics as JSON")
    model = commands.add_parser("model", help="Describe a run's formulation, losses, optimizers and networks")
    model.add_argument("path", nargs="?", type=Path, default=None, help="run directory, slug, short ID, manifest.json or project .toml")
    model.add_argument("--networks", action="store_true",
                       help="also build each component on the meta device for per-layer detail (loads torch)")
    model.add_argument("--write", action="store_true",
                       help="record the per-layer detail as RUN/model.json for the viewer (implies --networks)")
    model.add_argument("--json", action="store_true", help="output model description JSON")
    events = commands.add_parser("events", help="Page through a run's recorded events as JSON")
    events.add_argument("run_dir", nargs="?", type=Path, default=None)
    events.add_argument("--cursor", help="opaque cursor returned by the previous page")
    events.add_argument("--limit", type=_positive_int, default=100)
    events.add_argument("--max-bytes", type=_positive_int, default=1048576)
    events.add_argument("--json", action="store_true", help="output events as JSON")
    metrics = commands.add_parser("metrics", help="Show a run's published metrics catalog")
    metrics.add_argument("run_dir", nargs="?", type=Path, default=None)
    metrics.add_argument("--revision", help="historical catalog SHA256 (default: active catalog)")
    metrics.add_argument("--json", action="store_true", help="output metrics catalog as JSON")
    project = commands.add_parser("project", help="Project a run's completed events into metric views")
    project.add_argument("run_dir", nargs="?", type=Path, default=None)
    project.add_argument("--map-spec", type=Path, help="JSON MapSpec; default selects published scalar metrics")
    project.add_argument("--follow", action="store_true", help="continue projecting newly completed events")
    project.add_argument("--limit", type=_positive_int, default=100, help="documents per bounded work page")
    project.add_argument("--json", action="store_true", help="output projection as JSON")
    contributions = commands.add_parser("contributions", help="Page through a run's projected metric contributions")
    contributions.add_argument("run_dir", nargs="?", type=Path, default=None)
    contributions.add_argument("--map-revision", help="map SHA256 (default: built-in scalar map)")
    contributions.add_argument("--cursor", help="last applied projection cursor")
    contributions.add_argument("--limit", type=_positive_int, default=100)
    contributions.add_argument("--max-bytes", type=_positive_int, default=1048576)
    serve = commands.add_parser("serve")
    serve.add_argument("run_dir", nargs="?", type=Path, default=None)
    from .ports import DEFAULT_VIEWER_PORT
    serve.add_argument("--port", "--server-port", dest="port", type=int,
                       help=f"listen port (default: {DEFAULT_VIEWER_PORT}; an occupied port is an "
                            "error, never a search; 0 selects any free port)")
    serve.add_argument("--host", "--server-host", dest="host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    serve.add_argument("--auth", choices=("none", "token"), default="none", help="authentication mode (default: none)")
    serve.add_argument("--public-origin", type=_public_origin, metavar="URL",
                       help="absolute origin of a TLS proxy in front of the viewer, "
                            "e.g. https://machine.tailnet.ts.net (scheme, host and optional port only)")
    serve.add_argument("--session-file", type=Path, help="new private credential file outside the run")
    serve_browser = serve.add_mutually_exclusive_group()
    serve_browser.add_argument("--open", action="store_true", help="open the local viewer in a browser")
    serve_browser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
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
    view_browser = view.add_mutually_exclusive_group()
    view_browser.add_argument("--open", action="store_true", help="open the local viewer in a browser")
    view_browser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
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
    checkpoint = commands.add_parser("checkpoint", help="Request or inspect an on-demand checkpoint for an active run")
    checkpoint.add_argument("run_dir", nargs="?", type=Path, default=None, help=f"{_RUN_TARGET} (default: active run)")
    operation = checkpoint.add_mutually_exclusive_group()
    operation.add_argument("--request-id", help="reuse this ID to retry the same request safely")
    operation.add_argument("--status", metavar="REQUEST_ID", help="read a request receipt without submitting")
    checkpoint.add_argument("--attempt-id", help="target this attempt (default: current manifest attempt)")
    checkpoints = commands.add_parser("checkpoints", help="Manage a run's saved checkpoints (prune)")
    checkpoints_subs = checkpoints.add_subparsers(dest="checkpoints_command", required=True)
    prune = checkpoints_subs.add_parser("prune", help="Prune obsolete checkpoints under the retention policy")
    prune.add_argument("run_dir", nargs="?", type=Path, default=None, help=_RUN_TARGET)
    prune.add_argument("--dry-run", action="store_true", help="evaluate retention without deleting any files")
    prune.add_argument("--json", action="store_true", help="output prune results as JSON")
    export = commands.add_parser("export", help="Export a standalone EMA inference model bundle from a run checkpoint")
    export.add_argument("run_dir", nargs="?", type=Path, default=None, help="run directory, slug, short ID, or name")
    export.add_argument("--checkpoint", default="latest", help="target checkpoint: best, latest, step number, or path")
    export.add_argument("-o", "--output", type=Path, help="output file or directory (default: <run_dir>/model.pt)")
    export.add_argument("--json", action="store_true", help="output export info as JSON")
    convert = commands.add_parser("convert", help="Convert a legacy recipe TOML into a model file")
    convert.add_argument("recipe", type=Path, help="path to legacy recipe TOML")
    convert.add_argument("--output", type=Path, required=True, help="output model.toml destination")
    def _add_train_and_resume_arguments(p, *, is_resume=False):
        if is_resume:
            p.add_argument("run_dir", nargs="?", type=Path, default=None, help="existing run directory, slug, short ID, or name to resume")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="explicit run directory override")
            p.add_argument("--config", type=Path, default=None, help="verify exact compatibility with this configuration")
        else:
            p.add_argument("config", nargs="?", type=Path, default=None, help="configuration file or existing run directory/slug/ID to resume")
            p.add_argument("-n", "--name", default=None, help="custom experiment name / slug for the run")
            p.add_argument("--run-dir", type=Path, required=False, default=None, help="new run directory, or an existing run to resume")
            p.add_argument("--config", dest="config_opt", type=Path, default=None, help="explicit configuration file override")
        p.add_argument("-d", "--devices", default=None,
                       help="device(s) to train on: 0, 0,1, gpu0,gpu1, auto, cpu")
        p.add_argument("--checkpoint", default=None, help="target checkpoint: best, latest, step number, or path")
        p.add_argument("--steps", type=_positive_int, help="total target steps; may increase on resume when lr_floor=1 (constant learning rate)")
        p.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                       help="override configuration value (e.g. --set train.device=cuda:1 or --set losses.reconstruction.weight=0.5; "
                            "VALUE null removes the key)")
        _run_options(p)

    train = commands.add_parser("train", help="Create a run or resume its latest checkpoint with the same configuration")
    _add_train_and_resume_arguments(train, is_resume=False)

    resume = commands.add_parser("resume", help="Continue a complete training checkpoint on its recorded device")
    _add_train_and_resume_arguments(resume, is_resume=True)
    sample = commands.add_parser("sample", help="Sample a saved reference model (requires the train extra)")
    sample.add_argument("run_dir", nargs="?", type=str, default=None, help="run directory, slug, short ID, name, or standalone .pt bundle file (default: active or latest run)")
    sample.add_argument("--count", type=_positive_int, default=16, help="number of samples to generate (default: 16)")
    sample.add_argument("--seed", type=int, default=None, help="random seed for sampling (default: [sampling].seed in config, or 42)")
    sample.add_argument("--output", type=Path, default=None, help="output file path for samples (e.g. sample.png or sample.json)")
    sample.add_argument("-i", "--input", dest="inputs", action="append", default=[], metavar="PATH_OR_KEY=PATH", help="conditioning input image file path or KEY=PATH binding for conditional models")
    sample.add_argument("--sampler", help="sampler name declared in [samplers], or module:function import path")
    sample.add_argument("--checkpoint", default=None, help="target checkpoint: best, latest, step number, or path")
    sample.add_argument("--json", action="store_true", help="output sample info as JSON")

    queue_cmd = commands.add_parser("queue", help="Multi-GPU job queue scheduler, runner daemon, and service generator")
    queue_cmd.add_argument("--json", action="store_true", default=False, help="output JSON format")
    queue_cmd.add_argument("--queue-file", type=Path, default=None, help="override queue file destination")
    queue_subs = queue_cmd.add_subparsers(dest="queue_command", required=False)

    q_add = queue_subs.add_parser("add", help="Enqueue a training experiment or wait directive")
    q_add.add_argument("model_file", type=Path, help="path to experiment model TOML (or run directory if --wait)")
    q_add.add_argument("-d", "--devices", default=None, help="target devices (e.g. gpu0, gpu0,gpu1, 0, 0,1)")
    q_add.add_argument("--run-dir", type=Path, default=None, help="explicit run directory")
    q_add.add_argument("-n", "--name", default=None, help="optional custom job name")
    q_add.add_argument("--steps", type=_positive_int, default=None, help="override total training steps")
    q_add.add_argument("-s", "--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                       help="override configuration value (e.g. --set train.steps=1000)")
    q_add.add_argument("--no-web", "--no-server", dest="no_web", action="store_true", default=False,
                       help="train jobs without web server / dashboard")
    q_add.add_argument("--wait", action="store_true", help="wait for target run directory to complete before proceeding")
    q_add.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_add.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_ls = queue_subs.add_parser("ls", help="List jobs from the queue")
    q_ls.add_argument("--status", choices=("queued", "running", "completed", "failed", "cancelled", "interrupted"), default=None, help="filter by status")
    q_ls.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_ls.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_status = queue_subs.add_parser("status", help="Inspect job details, progress, and live log tail")
    q_status.add_argument("job_id", nargs="?", default=None, help="job ID to inspect (default: active/latest job)")
    q_status.add_argument("--lines", type=_positive_int, default=20, help="number of log tail lines (default: 20)")
    q_status.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_status.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_stop = queue_subs.add_parser("stop", help="Cancel queued jobs or signal running job to halt")
    q_stop.add_argument("job_id", nargs="?", default=None, help="job ID to stop")
    q_stop.add_argument("--all", action="store_true", help="cancel all queued jobs and halt queue runner")
    q_stop.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_stop.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_clear = queue_subs.add_parser("clear", help="Cancel all queued jobs and halt queue runner (alias for queue stop --all)")
    q_clear.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_clear.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_run = queue_subs.add_parser("run", help="Start the sequential queue daemon runner")
    q_run.add_argument("-d", "--devices", default=None, help="restrict runner to specific device(s)")
    q_run.add_argument("--poll-interval", type=_positive_seconds, default=5.0, help="seconds between queue checks (default: 5.0)")
    q_run.add_argument("--once", action="store_true", help="execute available jobs and exit instead of continuous loop")
    q_run.add_argument("--no-web", "--no-server", dest="no_web", action="store_true", default=False,
                       help="execute jobs without web server / dashboard")
    q_run.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="output JSON format")
    q_run.add_argument("--queue-file", type=Path, default=argparse.SUPPRESS, help="override queue file destination")

    q_svc = queue_subs.add_parser("install-service", help="Generate a systemd user service unit in ~/.config/systemd/user")
    # Units are always user units; --user is still accepted for existing scripts.
    q_svc.add_argument("--user", action="store_true", help=argparse.SUPPRESS)
    q_svc.add_argument("-d", "--devices", default=None, help="bind service runner to specific device(s)")
    q_svc.add_argument("--name", default=None, help="custom service name")
    q_svc.add_argument("--dry-run", action="store_true", help="print generated unit file without writing to disk")

    def _validate_qualify_timeout(val: str) -> float:
        try:
            t = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid timeout value: {val}")
        if not math.isfinite(t) or t < 1.0 or t > 180.0:
            raise argparse.ArgumentTypeError(f"Qualification timeout must be between 1 and 180s, got {t}")
        return t

    qualify_cmd = commands.add_parser(
        "qualify",
        help="Run hardware and distributed qualification smoke suite and emit JSON verification receipt",
    )
    qualify_cmd.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="destination path for JSON verification receipt (default: a per-run file under the HyperGAN state directory, qualification/)",
    )
    qualify_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="emit a receipt whose suites report dry_run without running any of them",
    )
    qualify_cmd.add_argument(
        "--suite",
        default=None,
        help="filter to a specific qualification suite (e.g. single_device, cpu_replicated_gloo, cuda_replicated_nccl, aggregation, multinode, checkpoint_compatibility)",
    )
    qualify_cmd.add_argument(
        "--timeout",
        type=_validate_qualify_timeout,
        default=180.0,
        help="hard wall-clock execution timeout in seconds (maximum 180s; default: 180.0)",
    )
    qualify_cmd.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show real-time test progress and intermediate diagnostics",
    )
    qualify_cmd.add_argument(
        "--json",
        action="store_true",
        help="print receipt JSON to stdout in addition to saving to disk",
    )

    return parser


def _parse_set_value(raw: str) -> object:
    val = raw.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null":
        return None
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        if not _all_finite(parsed):
            raise ValueError(f"--set value {raw!r} is not finite")
        return parsed
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def _all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return True


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


def _load_stored_run_model(run_dir: Path):
    """Load stored Model from run_dir and synchronize its name with manifest.json if renamed via `hypergan tag`."""
    from . import api as hg

    m = hg._load_stored_run_model(run_dir)
    manifest_p = Path(run_dir) / "manifest.json"
    if manifest_p.is_file():
        try:
            manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            if isinstance(manifest_data, dict):
                m_cfg = manifest_data.get("config") if isinstance(manifest_data.get("config"), dict) else {}
                manifest_name = m_cfg.get("name") or manifest_data.get("name")
                if manifest_name and isinstance(getattr(m, "spec", None), dict):
                    if m.spec.get("name") != manifest_name:
                        m = hg.override(m, {"name": manifest_name})
        except Exception:
            pass
    return m


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

    needs_reexec = bool(_reexec_unsatisfied(target_python, target_src_paths, target_env_vars))
    return needs_reexec, target_python, target_src_paths, target_env_vars


def _reexec_unsatisfied(target_python, target_src_paths, target_env_vars):
    """Describe each way this process differs from the requested interpreter, sources and environment."""
    unsatisfied = []
    if target_python:
        tp = Path(target_python)
        if (tp.parent.parent / "pyvenv.cfg").is_file():
            if Path(sys.prefix).resolve() != tp.parent.parent.resolve():
                unsatisfied.append(f"python venv (current: {sys.prefix}, target: {tp.parent.parent})")
        elif Path(sys.executable).resolve() != tp.resolve():
            unsatisfied.append(f"python (current: {sys.executable}, target: {target_python})")
    for k, v in target_env_vars.items():
        if os.environ.get(k) != str(v):
            unsatisfied.append(f"env var {k} (current: {os.environ.get(k)!r}, target: {v!r})")
    resolved_head = [str(Path(x).resolve()) for x in sys.path[:len(target_src_paths) + 3] if x]
    for p in target_src_paths:
        if str(Path(p).resolve()) not in resolved_head:
            unsatisfied.append(f"src path {p} not in sys.path")
    return unsatisfied


def _cli_argv(args):
    """The arguments this command was invoked with, without the program name."""
    return list(getattr(args, "_argv", None) or sys.argv[1:])


def _maybe_reexec(run_spec, ws_path, ws_data, run_dir, argv, *, prepare=None):
    """Re-execute as ``python -m hypergan ARGV`` when the run asks for another interpreter or environment.

    ``prepare(attempt, python, src_paths, env)`` runs first and may return
    replacement arguments. Returns only when no re-exec is needed.
    """
    needs_reexec, target_py, target_srcs, target_envs = _check_reexec(run_spec, ws_data)
    if not needs_reexec:
        return
    if os.environ.get("_HYPERGAN_REEXEC") == "1":
        unsatisfied = _reexec_unsatisfied(target_py, target_srcs, target_envs)
        raise RuntimeError(
            f"Re-exec loop detected: process was already re-executed with _HYPERGAN_REEXEC=1, "
            f"but target environment conditions remain unsatisfied: {', '.join(unsatisfied)}"
        )
    attempt = _determine_attempt(run_dir)
    if prepare is not None:
        argv = prepare(attempt, target_py, target_srcs, target_envs)
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
    os.execve(target_executable, [target_executable, "-m", "hypergan", *argv], new_env)


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



def _project_recipes(configs_dir):
    """The project's configs/*.toml recipe as a list of at most one.

    Several recipes are ambiguous, so a command given only the project refuses to pick one.
    """
    recipes = sorted(configs_dir.glob("*.toml")) if configs_dir.is_dir() else []
    if len(recipes) > 1:
        recipe_list = "\n".join(f"  - {p}" for p in recipes)
        raise ValueError(f"Multiple recipes found in {configs_dir}. Specify which recipe to use:\n{recipe_list}")
    return recipes


def _sampler_output_path(sample_path, sampler, view):
    """Where ``sample --sampler`` saves the sampler's result: next to the sample, never over it."""
    sample_path = Path(sample_path)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(sampler)).strip("-.") or "view"
    suffix = {"Image": ".png", "Audio": ".wav"}.get(type(view).__name__, sample_path.suffix)
    return sample_path.with_name(f"{sample_path.stem}-{label}{suffix}")


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



def _run_diff(target_a, target_b=None, sets=None, is_json=False):
    from . import api as hg
    res = hg.diff(target_a, target_b, sets=sets)
    if is_json:
        _print_json(res)
    elif res["identical"]:
        print(f"No differences found between {res['target_a']} and {res['target_b']}.")
    else:
        print(f"Differences between {res['target_a']} and {res['target_b']}:")
        print(f"{'Key':<35} {'Target A':<25} {'Target B':<25}")
        print("-" * 87)
        for k, diff in res["differences"].items():
            str_a = str(diff["a"]) if diff["a"] is not None else "(none)"
            str_b = str(diff["b"]) if diff["b"] is not None else "(none)"
            print(f"{k:<35} {str_a:<25} {str_b:<25}")
    return 0


def _run_doctor(is_json=False):
    import platform
    from . import api as hg
    from .inspection import format_bytes

    torch_installed = False
    torch_version = None
    cuda_avail = False
    cuda_ver = None
    devices = []
    try:
        import torch
        torch_installed = True
        torch_version = getattr(torch, "__version__", None)
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
            for idx in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(idx)
                devices.append({
                    "index": idx,
                    "name": prop.name,
                    "total_memory_bytes": getattr(prop, "total_memory", 0),
                    "major": getattr(prop, "major", 0),
                    "minor": getattr(prop, "minor", 0),
                })
    except Exception:
        pass

    ws_path = hg.find_workspace(Path.cwd())
    runs_root = None
    try:
        runs_root = str(hg.find_runs_root(Path.cwd()))
    except Exception:
        pass

    from .registry import get_registry_dir
    state_dir = str(get_registry_dir())

    check_dir = Path(runs_root) if runs_root else Path.cwd()
    total_bytes = 0
    free_bytes = 0
    try:
        usage = shutil.disk_usage(check_dir)
        total_bytes = usage.total
        free_bytes = usage.free
    except Exception:
        pass

    active_runs = []
    try:
        active_runs = hg.ps(all_runs=False)
    except Exception:
        pass

    queued_jobs = []
    try:
        from .queue import queue_ls
        queued_jobs = queue_ls(status_filter="queued")
    except Exception:
        pass

    doc_data = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "pytorch": {
            "installed": torch_installed,
            "version": torch_version,
            "cuda_available": cuda_avail,
            "cuda_version": cuda_ver,
            "device_count": len(devices),
            "devices": devices,
        },
        "workspace": {
            "root": str(ws_path.parent) if ws_path else None,
            "runs_root": runs_root,
            "state_dir": state_dir,
        },
        "storage": {
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "free_human": format_bytes(free_bytes) if free_bytes else "--",
        },
        "workload": {
            "active_runs": len(active_runs),
            "queued_jobs": len(queued_jobs),
        },
        "ok": True,
    }

    if is_json:
        _print_json(doc_data)
    else:
        print("HyperGAN Doctor System Diagnostics:")
        print(f"  Python:      {doc_data['python']['version']} ({doc_data['python']['executable']})")
        cuda_info = f"CUDA {doc_data['pytorch']['cuda_version']}, {len(devices)} GPU(s)" if cuda_avail else "CPU only / CUDA unavailable"
        pt_info = f"{doc_data['pytorch']['version']} ({cuda_info})" if torch_installed else "Not installed"
        print(f"  PyTorch:     {pt_info}")
        if devices:
            for d in devices:
                print(f"    GPU {d['index']}: {d['name']} ({format_bytes(d['total_memory_bytes'])})")
        print(f"  Workspace:   {doc_data['workspace']['root'] or '(none detected)'}")
        print(f"  Runs root:   {doc_data['workspace']['runs_root'] or '(none detected)'}")
        print(f"  Storage:     {doc_data['storage']['free_human']} free")
        print(f"  Workload:    {len(active_runs)} active run(s), {len(queued_jobs)} queued job(s)")
        print("  Status:      OK")
    return 0


def _run_report(run_targets=None, is_json=False):
    from . import api as hg
    from .registry import is_run_dir
    from .inspection import format_bytes
    from .bounded_cli_output import human_duration

    resolved_dirs = []
    if run_targets:
        for t in run_targets:
            resolved = hg.resolve_run_target(t)
            if not is_run_dir(resolved):
                raise FileNotFoundError(f"Run target '{t}' cannot be resolved to a valid run directory.")
            resolved_dirs.append(resolved)
    else:
        try:
            runs_list = hg.runs_ls(None)
            for r in runs_list:
                r_dir = r.get("run_dir")
                if r_dir and is_run_dir(Path(r_dir)):
                    resolved_dirs.append(Path(r_dir))
        except Exception:
            pass

    records = []
    for r_dir in resolved_dirs:
        m_data = {}
        manifest_p = r_dir / "manifest.json"
        if manifest_p.is_file():
            try:
                raw_m = json.loads(manifest_p.read_text(encoding="utf-8"))
                if isinstance(raw_m, dict):
                    m_data = raw_m
            except Exception:
                pass

        s_data = {}
        summary_p = r_dir / "summary.json"
        if summary_p.is_file():
            try:
                raw_s = json.loads(summary_p.read_text(encoding="utf-8"))
                if isinstance(raw_s, dict):
                    s_data = raw_s
            except Exception:
                pass

        run_id = m_data.get("run_id") or s_data.get("run_id") or r_dir.name
        name = m_data.get("name") or s_data.get("name") or r_dir.name
        slug = m_data.get("slug") or s_data.get("slug")
        status = s_data.get("status") or m_data.get("status", "unknown")
        step = s_data.get("step") if s_data.get("step") is not None else m_data.get("steps", 0)
        total_steps = (
            s_data.get("total_steps")
            or m_data.get("total_steps")
            or (m_data.get("config", {}).get("training", {}).get("steps") if isinstance(m_data.get("config"), dict) else None)
        )
        elapsed_sec = s_data.get("elapsed_seconds") or m_data.get("training_seconds") or m_data.get("seconds", 0)

        # summary.best_fid is set only when the primary metric is an FID
        # (summary.is_fid_metric); summary.best may be any metric.
        best_fid = s_data["best_fid"] if "best_fid" in s_data else m_data.get("best_fid")

        latest_metrics = s_data.get("latest_metrics") or m_data.get("latest_metrics") or {}
        if not isinstance(latest_metrics, dict):
            latest_metrics = {}
        loss_g = latest_metrics.get("loss/g_total", latest_metrics.get("loss_g"))

        latest_ckpt = s_data.get("checkpoint_path") or m_data.get("checkpoint_path")

        artifacts = []
        for cand in ["model.pt", "samples.png", "samples.json", "summary.json"]:
            if (r_dir / cand).is_file():
                artifacts.append(cand)
        if (r_dir / "checkpoints").is_dir():
            artifacts.append("checkpoints/")

        records.append({
            "run_id": run_id,
            "short_id": run_id[:8] if run_id else "",
            "name": name,
            "slug": slug,
            "run_dir": str(r_dir),
            "status": status,
            "step": step,
            "total_steps": total_steps,
            "elapsed_seconds": elapsed_sec,
            "best_fid": best_fid,
            "loss_g": loss_g,
            "latest_checkpoint": latest_ckpt,
            "artifacts": artifacts,
        })

    if is_json:
        _print_json(records)
    else:
        if not records:
            print("No runs registered in workspace.")
            return 0
        print("| Run | Status | Steps | Best FID | Loss G | Time | Latest Checkpoint | Artifacts |")
        print("|---|---|---|---|---|---|---|---|")
        for rec in records:
            run_lbl = f"{rec['name']} (`{rec['short_id']}`)" if rec['short_id'] else rec['name']
            steps_lbl = f"{rec['step']:,} / {rec['total_steps']:,}" if rec['total_steps'] else f"{rec['step']:,}"
            fid_lbl = f"{rec['best_fid']:.2f}" if rec['best_fid'] is not None else "--"
            loss_lbl = f"{rec['loss_g']:.4f}" if rec['loss_g'] is not None else "--"
            time_lbl = human_duration(rec['elapsed_seconds']) or "--"
            ckpt_lbl = f"`{rec['latest_checkpoint']}`" if rec['latest_checkpoint'] else "--"
            art_lbl = ", ".join(rec['artifacts']) if rec['artifacts'] else "--"
            print(f"| {run_lbl} | {rec['status']} | {steps_lbl} | {fid_lbl} | {loss_lbl} | {time_lbl} | {ckpt_lbl} | {art_lbl} |")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _parser().print_help()
        return 0
    args = _parser().parse_args(argv)
    args._argv = list(argv)
    if getattr(args, "global_json", False):
        setattr(args, "json", True)
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
            target = hg.resolve_run_target(args.run_dir) if args.run_dir is not None else hg.resolve_run_target(None)
            if not target.exists():
                raise FileNotFoundError(f"Run directory does not exist: {target}")
            # One follow loop for both formats; it never splits a record that is still being written.
            hg.logs(target, tail=args.tail, follow=args.follow, raw=bool(getattr(args, "json", False)), stream=sys.stdout)
        elif args.command == "stop":
            res = hg.stop(args.run_dir, stop_all=args.all, timeout=args.timeout)
            if args.json:
                _print_json(res)
            else:
                print(res.get("message", f"Stopped {res.get('short_id', '')}"))
        elif args.command == "tag":
            res = hg.tag(args.run_dir, args.name)
            if getattr(args, "json", False) or not sys.stdout.isatty():
                _print_json(res)
            else:
                print(res["message"])
        elif args.command == "rm":
            res = hg.rm(args.run_dir, force=args.force)
            if getattr(args, "json", False) or not sys.stdout.isatty():
                _print_json(res)
            else:
                print(res["message"])
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
                from .registry import is_run_dir
                for r in args.runs:
                    resolved = hg.resolve_run_target(r)
                    if not is_run_dir(resolved):
                        raise FileNotFoundError(f"Run target '{r}' cannot be resolved to a valid run directory (checked: {resolved}).")
                comp = hg.runs_compare(args.runs, metric=args.metric, at_steps=args.at)
                if getattr(args, "json", False):
                    _print_json(comp)
                elif args.csv:
                    from .inspection import format_comparison_csv
                    print(format_comparison_csv(comp), end="")
                else:
                    from .inspection import format_comparison_table
                    print(format_comparison_table(comp))
        elif args.command == "compare":
            from .registry import is_run_dir
            for r in args.runs:
                resolved = hg.resolve_run_target(r)
                if not is_run_dir(resolved):
                    raise FileNotFoundError(f"Run target '{r}' cannot be resolved to a valid run directory (checked: {resolved}).")
            comp = hg.runs_compare(args.runs, metric=args.metric, at_steps=args.at)
            if getattr(args, "json", False):
                _print_json(comp)
            elif args.csv:
                from .inspection import format_comparison_csv
                print(format_comparison_csv(comp), end="")
            else:
                from .inspection import format_comparison_table
                print(format_comparison_table(comp))
        elif args.command == "diff":
            d_sets = _parse_set_overrides(args.sets) if getattr(args, "sets", None) else {}
            return _run_diff(args.target_a, args.target_b, sets=d_sets, is_json=getattr(args, "json", False))
        elif args.command == "explain":
            explain_path = args.path
            if explain_path is None:
                explain_path = hg.resolve_run_target(None)
            elif explain_path.is_dir() and not (explain_path / "model.toml").is_file():
                configs_dir = explain_path / "configs"
                cand_tomls = _project_recipes(configs_dir)
                if cand_tomls:
                    explain_path = cand_tomls[0]
            m = hg.load(explain_path)
            e_sets = _parse_set_overrides(args.sets) if getattr(args, "sets", None) else {}
            if e_sets:
                e_sets = hg._expand_override_shorthand(m, e_sets)
                m = hg.override(m, e_sets)
            text = hg.explain(m)
            if getattr(args, "json", False):
                _print_json({"path": str(explain_path), "explanation": text})
            else:
                print(text)
            return 0
        elif args.command == "doctor":
            return _run_doctor(is_json=getattr(args, "json", False))
        elif args.command == "report":
            return _run_report(run_targets=args.runs, is_json=getattr(args, "json", False))
        elif args.command == "qualify":
            from .qualify import run_qualification_cli
            return run_qualification_cli(args)
        elif args.command == "gc":
            if args.all:
                res = hg.gc_all(root=args.run_dir, dry_run=args.dry_run, stride=args.stride)
                if getattr(args, "json", False):
                    _print_json(res)
                else:
                    from .inspection import format_bytes
                    cleaned = [r for r in res if not r.get("skipped")]
                    skipped = [r for r in res if r.get("skipped")]
                    cleaned_count = len(cleaned)
                    skipped_count = len(skipped)
                    reclaimed = sum(r.get("total_reclaimed", 0) for r in cleaned)
                    header = (
                        f"GC dry run across {cleaned_count + skipped_count} runs (would clean):"
                        if args.dry_run
                        else f"GC complete across {cleaned_count + skipped_count} runs:"
                    )
                    action_label = "Would clean:" if args.dry_run else "Cleaned: "
                    print(header)
                    print(f"  {action_label} {cleaned_count} runs (reclaimed {format_bytes(reclaimed)})")
                    if skipped_count:
                        print(f"  Skipped:  {skipped_count} active runs")
            else:
                # gc resolves strictly itself; the fuzzy read-only resolver must not pick its target.
                res = hg.gc(args.run_dir, dry_run=args.dry_run, stride=args.stride)
                if getattr(args, "json", False):
                    _print_json(res)
                else:
                    from .inspection import format_bytes
                    views_str = format_bytes(res.get("views_reclaimed_bytes", 0))
                    events_str = format_bytes(res.get("events_reclaimed_bytes", 0))
                    total_str = format_bytes(res.get("reclaimed_bytes", 0))
                    prefix = "Would clean" if args.dry_run else "Cleaned"
                    print(f"{prefix} run: {res.get('run_dir', args.run_dir)}")
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
            if getattr(args, "json", False):
                _print_json({"exported": str(out)})
            else:
                print(out)
        elif args.command == "data-check":
            _print_json(hg.data_check(args.config, output=args.output))
        elif args.command == "download-data":
            if getattr(args, "list_datasets", False):
                datasets = hg.list_datasets()
                if getattr(args, "json", False):
                    _print_json({"datasets": datasets})
                else:
                    for ds in datasets:
                        mib = ds["expected_bytes"] / (1024 * 1024)
                        print(f"{ds['name']} ({ds['version']}) - {ds['expected_bytes']} bytes ({mib:.2f} MiB) - {ds['url']}")
                        print(f"  {ds['usage_notice']}")
            elif getattr(args, "info", False):
                info = hg.dataset_info(args.dataset)
                if getattr(args, "json", False):
                    _print_json(info)
                else:
                    mib = info["expected_bytes"] / (1024 * 1024)
                    print(f"Dataset:         {info['name']} ({info['title']})")
                    print(f"Version:         {info['version']}")
                    print(f"Source URL:      {info['url']}")
                    print(f"Archive:         {info['archive_filename']} ({info['expected_bytes']} bytes, {mib:.2f} MiB)")
                    print(f"Archive SHA256:  {info['archive_sha256']}")
                    print(f"Extracted dir:   {info['extracted_dir']}")
                    print(f"Data factory:    {info['factory']}")
                    print("Pinned files:")
                    for fname, fsha in info["files_sha256"].items():
                        print(f"  {fname}: {fsha}")
                    print(f"Citation:        {info['citation']}")
                    print(f"Usage notice:    {info['usage_notice']}")
            else:
                hg.dataset_info(args.dataset)
                if args.output is None:
                    raise ValueError("hypergan download-data requires --output DIR (or pass --info / --list)")
                res = hg.download_dataset(
                    args.dataset,
                    args.output,
                    cache_dir=args.cache_dir,
                    url=args.url,
                    source_archive=args.source_archive,
                    verify_only=args.verify_only,
                    timeout=args.timeout,
                )
                if getattr(args, "json", False):
                    _print_json(res)
                else:
                    print(f"Prepared dataset: {res['dataset']} ({res['version']})")
                    print(f"Root directory:   {res['root']}")
                    print(f"Manifest:         {res['manifest_path']}")
                    print(f"Usage notice:     {res['usage_notice']}")
                    print("\nConfiguration snippet:\n" + res["config_snippet"].rstrip())
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
                        template=tmpl,
                        resolution=res_opt,
                        data_dir=dataset_arg,
                        output_config=out_opt,
                        device=args.device,
                    )
                    if getattr(args, "json", False):
                        _print_json(info)
                    else:
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
                # One naming rule for every template: a .toml path is the file,
                # anything else is a directory holding model.toml.
                dest = target_dest if target_dest.suffix == ".toml" and not target_dest.is_dir() else target_dest / "model.toml"
                if dataset_arg is not None or (tmpl is not None and tmpl != "gaussian-grid") or res_opt is not None:
                    from .recipe_generator import generate_recipe
                    saved = generate_recipe(
                        dataset_dir=dataset_arg,
                        output_path=dest,
                        device=args.device,
                        template=tmpl or "dcgan",
                        resolution=res_opt,
                    )
                    if getattr(args, "json", False):
                        # generate_recipe names the model after its file.
                        _print_json({"recipe": str(saved), "name": saved.stem, "device": args.device, "template": tmpl or "dcgan"})
                    else:
                        print(saved)
                else:
                    from .config import validate_device
                    validate_device(args.device)
                    m = hg.packaged("gaussian-grid")
                    if args.device:
                        m = hg.override(m, {"train.device": args.device})
                    if dest.exists():
                        raise FileExistsError(f"{dest} already exists; choose a new path")
                    saved = Path(hg.save(m, dest)).resolve()
                    if getattr(args, "json", False):
                        _print_json({"recipe": str(saved), "device": args.device, "template": "gaussian-grid"})
                    else:
                        print(saved)
            elif args.command == "recipes":
                _print_json(hg.packaged_list())
            elif args.command == "convert":
                raw = hg.load_recipe(args.recipe)
                m = hg.lift(raw, base=args.recipe.parent)
                saved = hg.save(m, args.output)
                print(saved)
            else:
                validate_path = args.path
                if validate_path.is_dir() and not (validate_path / "model.toml").is_file():
                    configs_dir = validate_path / "configs"
                    cand_tomls = _project_recipes(configs_dir)
                    if cand_tomls:
                        validate_path = cand_tomls[0]
                    elif (validate_path / "hypergan.toml").is_file():
                        ws_info = hg.load_workspace(validate_path / "hypergan.toml")
                        if getattr(args, "json", False):
                            _print_json(ws_info)
                        else:
                            print(
                                f"note: {validate_path / 'hypergan.toml'} is a workspace config; place model recipes in {validate_path / 'configs'}/*.toml",
                                file=sys.stderr,
                            )
                        return 0
                elif validate_path.is_file() and validate_path.name == "hypergan.toml":
                    ws_info = hg.load_workspace(validate_path)
                    configs_dir = validate_path.parent / "configs"
                    cand_tomls = _project_recipes(configs_dir)
                    if cand_tomls:
                        print(
                            f"note: {validate_path} is a workspace config; validating project recipe {cand_tomls[0]}",
                            file=sys.stderr,
                        )
                        validate_path = cand_tomls[0]
                    else:
                        if getattr(args, "json", False):
                            _print_json(ws_info)
                        else:
                            print(
                                f"note: {validate_path} is a workspace config (place model recipes in configs/*.toml)",
                                file=sys.stderr,
                            )
                        return 0

                m = hg.load(validate_path)
                v_sets = _parse_set_overrides(args.sets) if getattr(args, "sets", None) else {}
                if v_sets:
                    v_sets = hg._expand_override_shorthand(m, v_sets)
                    m = hg.override(m, v_sets)
                warnings = hg.validate(m)
                for warning in warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                do_runtime = getattr(args, "runtime", False)
                do_data = getattr(args, "data", False)

                def _run_check_data(model_obj):
                    try:
                        return hg.check_data(model_obj, output=getattr(args, "output", None))
                    except ValueError as exc:
                        if "requires dataset.type = 'image_folder'" in str(exc):
                            spec_d = getattr(model_obj, "spec", {}) if hasattr(model_obj, "spec") else {}
                            ds_sec = (spec_d.get("data") or spec_d.get("dataset") or {}) if isinstance(spec_d, dict) else {}
                            ds_type = ds_sec.get("type", "synthetic") if isinstance(ds_sec, dict) else "synthetic"
                            raise ValueError(
                                f"--data validation requires an image_folder dataset (recipe uses dataset.type = {ds_type!r}, which has no image directory to verify)"
                            ) from exc
                        raise

                if do_runtime and do_data:
                    resolved_cfg = dict(hg.resolve(m))
                    resolved_cfg["preflight"] = hg.preflight_check(m, profile=getattr(args, "profile", None), runtime=True)
                    resolved_cfg["data_check"] = _run_check_data(m)
                    _print_json(resolved_cfg)
                elif do_runtime:
                    _print_json(hg.preflight_check(m, profile=getattr(args, "profile", None), runtime=True))
                elif do_data:
                    _print_json(_run_check_data(m))
                elif getattr(args, "json", False):
                    _print_json(hg.resolve(m))
                else:
                    spec_d = getattr(m, "spec", m) if hasattr(m, "spec") else m
                    r_name = spec_d.get("name", validate_path.stem)
                    t_sec = spec_d.get("training", spec_d.get("train", {}))
                    dev = t_sec.get("device", "cuda")
                    steps = t_sec.get("steps", "--")
                    bs = t_sec.get("batch_size", "--")
                    print(f"Valid: {validate_path}")
                    print(f"  Recipe:     {r_name}")
                    print(f"  Device:     {dev}")
                    print(f"  Batch size: {bs}")
                    print(f"  Steps:      {steps}")
        elif args.command in {"train", "resume"}:
            from .execution import SERVICE_TIMEOUTS
            from .registry import is_run_dir as _is_valid_run_dir
            explicit_run_dir = getattr(args, "run_dir", None) is not None
            is_cpu_override = str(getattr(args, "devices", "") or "").strip().lower() == "cpu"
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
            if not is_resume_cmd:
                if getattr(args, "config", None) is None and getattr(args, "config_opt", None) is not None:
                    rd = getattr(args, "run_dir", None)
                    if rd is not None and _is_valid_run_dir(rd.resolve()):
                        args.config = rd
                    else:
                        args.config = args.config_opt
                        args.config_opt = None
                if getattr(args, "config", None) is None and getattr(args, "run_dir", None) is None:
                    _parser().error("the following arguments are required: config")
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
                config_override = getattr(args, "config", None) if is_resume_cmd else getattr(args, "config_opt", None)
                manifest_p = args.run_dir / "manifest.json"
                manifest_data = {}
                if manifest_p.is_file():
                    try:
                        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                if (
                    is_resume_cmd
                    and getattr(args, "steps", None) is None
                    and not sets
                    and getattr(args, "checkpoint", None) is None
                    and config_override is None
                    and getattr(args, "stop_after_steps", None) is None
                    and getattr(args, "max_seconds", None) is None
                    and not getattr(args, "progress_json", False)
                ):
                    m_status = str(manifest_data.get("status") or "").lower()
                    m_steps = manifest_data.get("steps")
                    m_cfg = manifest_data.get("config") if isinstance(manifest_data.get("config"), dict) else {}
                    m_total = (
                        manifest_data.get("total_steps")
                        or manifest_data.get("max_steps")
                        or (m_cfg.get("training", {}) if isinstance(m_cfg.get("training"), dict) else {}).get("steps")
                        or (m_cfg.get("train", {}) if isinstance(m_cfg.get("train"), dict) else {}).get("steps")
                    )
                    if m_status in ("complete", "completed") and isinstance(m_steps, int) and isinstance(m_total, int) and m_total > 0 and m_steps >= m_total:
                        print(f"Run '{args.run_dir.name}' is already complete at step {m_steps}/{m_total}. Pass --steps <N> to extend training.")
                        return 0

                if config_override:
                    m = hg.load(config_override)
                else:
                    try:
                        m = _load_stored_run_model(args.run_dir)
                    except Exception:
                        m = None
                if m and sets:
                    sets = hg._expand_override_shorthand(m, sets)
                    m = hg.override(m, sets)
                if is_cpu_override:
                    if m:
                        m = hg.override(m, {"train.device": "cpu"})
                    sets["train.device"] = "cpu"

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

                _maybe_reexec(run_spec, ws_path, ws_data, args.run_dir, _cli_argv(args))

                resume_model = m if (config_override or sets) else None
                prepared = hg.prepare_resume(args.run_dir, getattr(args, "checkpoint", None), resume_model, steps=getattr(args, "steps", None), sets=sets, **options)
            else:
                config_path = target_path
                if config_path is not None:
                    if config_path.is_dir() and not (config_path / "model.toml").is_file():
                        configs_dir = config_path / "configs"
                        cand_tomls = _project_recipes(configs_dir)
                        if cand_tomls:
                            print(
                                f"note: '{config_path}' is a project directory; selected recipe '{cand_tomls[0]}'",
                                file=sys.stderr,
                            )
                            config_path = cand_tomls[0]
                        elif (config_path / "hypergan.toml").is_file():
                            raise ValueError(
                                f"No recipes found in {configs_dir}. Place a model recipe in {configs_dir}/*.toml or specify a recipe file directly."
                            )
                    elif config_path.is_file() and config_path.name == "hypergan.toml":
                        configs_dir = config_path.parent / "configs"
                        cand_tomls = _project_recipes(configs_dir)
                        if cand_tomls:
                            print(
                                f"note: '{config_path}' is a workspace config; selected recipe '{cand_tomls[0]}'",
                                file=sys.stderr,
                            )
                            config_path = cand_tomls[0]
                        else:
                            raise ValueError(
                                f"No recipes found in {configs_dir}. Place a model recipe in {configs_dir}/*.toml or specify a recipe file directly."
                            )
                custom_name = getattr(args, "name", None)
                if args.run_dir is None:
                    args.run_dir = hg._default_run_dir(config_path, name=custom_name)
                else:
                    args.run_dir = args.run_dir.resolve()

                m = hg.load(config_path)
                if custom_name:
                    m = hg.override(m, {"name": custom_name})
                if sets:
                    sets = hg._expand_override_shorthand(m, sets)
                    m = hg.override(m, sets)
                if is_cpu_override:
                    m = hg.override(m, {"train.device": "cpu"})

                if not explicit_run_dir and (args.run_dir / "manifest.json").is_file() and os.environ.get("_HYPERGAN_REEXEC") != "1":
                    try:
                        hg.prepare(m, args.run_dir, args.steps, **options)
                    except ValueError as exc:
                        if hg._config_differs(exc):
                            old_run_dir = args.run_dir
                            branch_stem = custom_name or config_path.stem
                            full_name, _ = hg.generate_run_slug(branch_stem, parent_dir=old_run_dir.parent)
                            args.run_dir = (old_run_dir.parent / full_name).resolve()
                            print(
                                f"note: '{old_run_dir.name}' has a different configuration; starting new run in '{args.run_dir}'",
                                file=sys.stderr,
                            )

                run_spec = m.spec.get("run", {}) if isinstance(m.spec, dict) else {}
                ws_path = hg.find_workspace(config_path)
                ws_data = hg.load_workspace(ws_path) if ws_path else {}

                def _record_launch(attempt, target_py, target_srcs, target_envs):
                    # The re-executed process trains from the lowered recipe written here.
                    args.run_dir.mkdir(parents=True, exist_ok=True)
                    from .provenance import write_launch_provenance, snapshot_config_sources
                    from .model_file import collect_source_files

                    sources = collect_source_files(config_path, ws_path.parent if ws_path else None)
                    snapshot_config_sources(args.run_dir, sources)
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

                    from .toml_writer import dumps as _dumps
                    recipe_text = _dumps(hg.lower(m), table_depth=3)
                    (args.run_dir / "recipe.toml").write_text(recipe_text, encoding="utf-8")
                    return _reconstruct_argv_for_reexec(["hypergan", *_cli_argv(args)], str(config_path),
                                                        str(args.run_dir / "recipe.toml"), str(args.run_dir))

                _maybe_reexec(run_spec, ws_path, ws_data, args.run_dir, _cli_argv(args), prepare=_record_launch)

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
                    train_sec = (config_dict.get("training") or config_dict.get("train") or {}) if isinstance(config_dict, dict) else {}
                    dev = "cpu" if is_cpu_override else train_sec.get("device", "cuda")
                    dashboard_url = getattr(args, "_dashboard_url", None)
                    _print_training_header(args.run_dir, exp_name or str(args.run_dir.name), dev, dashboard_url)
                is_tty = bool(getattr(output.stdout.original, "isatty", lambda: False)()) if output and hasattr(output, "stdout") else False
                is_profiled = getattr(prepared, "profile", None) is not None
                human_footer = (not is_resume_cmd and not is_profiled or is_tty) and not getattr(args, "progress_json", False)
                run_manifest = prepared.run(on_event=output.progress)
                dash_footer = getattr(args, "_dashboard_url", None) or ("headless (viewer disabled)" if args.no_server else None)
                output.result(run_manifest, run_dir=args.run_dir, human=human_footer, dashboard_url=dash_footer)
        elif args.command == "sample":
            raw_inputs = getattr(args, "inputs", None) or []
            parsed_inputs = None
            if raw_inputs:
                if len(raw_inputs) == 1 and "=" not in raw_inputs[0]:
                    parsed_inputs = Path(raw_inputs[0])
                else:
                    parsed_inputs = {}
                    for item in raw_inputs:
                        if "=" not in item:
                            raise ValueError(
                                f"Invalid --input '{item}'; expected KEY=PATH when multiple --input flags are given"
                            )
                        k, v = item.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if not k or not v:
                            raise ValueError(f"Invalid --input '{item}'; expected non-empty KEY=PATH")
                        parsed_inputs[k] = Path(v)

            target_raw = args.run_dir
            target_p = Path(target_raw) if target_raw is not None else None
            is_pt_bundle = bool(target_p and (str(target_p).lower().endswith(".pt") or (target_p.is_file() and target_p.suffix.lower() == ".pt")))

            if is_pt_bundle:
                if not target_p.is_file():
                    raise FileNotFoundError(f"Bundle file not found: {target_p}")
                eff_seed = args.seed
                if eff_seed is None:
                    try:
                        import torch
                        state = torch.load(target_p, map_location="cpu", weights_only=True)
                        cfg = state.get("config", {}) if isinstance(state, dict) else {}
                        s_cfg = (cfg.get("sampling") or cfg.get("sample") or {}) if isinstance(cfg, dict) else {}
                        eff_seed = s_cfg.get("seed") if isinstance(s_cfg, dict) else None
                    except Exception:
                        pass
                if eff_seed is None or not isinstance(eff_seed, int):
                    eff_seed = 42

                from . import artifacts
                out_path = artifacts.sample(
                    target_p,
                    count=args.count or 16,
                    seed=eff_seed,
                    output=args.output,
                    inputs=parsed_inputs,
                    checkpoint=getattr(args, "checkpoint", None),
                    default_image_png=True,
                )
                if getattr(args, "json", False):
                    _print_json({"sample": str(out_path), "path": str(out_path)})
                else:
                    print(out_path)
            else:
                resolved_run = hg.resolve_run_target(target_raw)
                eff_seed = args.seed
                if eff_seed is None:
                    manifest_p = resolved_run / "manifest.json"
                    if manifest_p.is_file():
                        try:
                            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                            cfg = m_data.get("config", {}) if isinstance(m_data.get("config"), dict) else {}
                            s_cfg = (cfg.get("sampling") or cfg.get("sample") or {}) if isinstance(cfg, dict) else {}
                            eff_seed = s_cfg.get("seed") if isinstance(s_cfg, dict) else None
                        except Exception:
                            pass
                    if eff_seed is None:
                        recipe_p = resolved_run / "recipe.toml"
                        if recipe_p.is_file():
                            try:
                                import tomllib
                                r_data = tomllib.loads(recipe_p.read_text(encoding="utf-8"))
                                s_cfg = (r_data.get("sampling") or r_data.get("sample") or {}) if isinstance(r_data, dict) else {}
                                eff_seed = s_cfg.get("seed") if isinstance(s_cfg, dict) else None
                            except Exception:
                                pass
                if eff_seed is None or not isinstance(eff_seed, int):
                    eff_seed = 42

                s = hg.sample(
                    resolved_run,
                    count=args.count or 16,
                    seed=eff_seed,
                    inputs=parsed_inputs,
                    output=args.output,
                    checkpoint=getattr(args, "checkpoint", None),
                    default_image_png=True,
                )
                view_path = None
                if getattr(args, "sampler", None):
                    view_obj = hg.view(s, args.sampler, run=resolved_run)
                    if not hasattr(view_obj, "save"):
                        raise TypeError(f"sampler {args.sampler!r} returned {type(view_obj).__name__}, "
                                        "which has no save(path) to write it with")
                    view_out = _sampler_output_path(s.path, args.sampler, view_obj)
                    # A custom save() may return None; the file is still at view_out.
                    view_path = view_obj.save(view_out) or view_out
                if getattr(args, "json", False):
                    result = {"sample": str(s.path), "path": str(s.path)}
                    if view_path is not None:
                        result["view"] = str(view_path)
                    _print_json(result)
                else:
                    print(s.path)
                    if view_path is not None:
                        print(view_path)
        elif args.command == "queue":
            from .queue import (
                queue_add,
                queue_ls,
                queue_status,
                queue_stop,
                queue_clear,
                queue_run,
                install_service as queue_install_service,
                format_queue_table,
                format_queue_status,
            )
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
                res = queue_add(
                    args.model_file,
                    devices=args.devices,
                    run_dir=args.run_dir,
                    name=args.name,
                    steps=getattr(args, "steps", None),
                    sets=getattr(args, "sets", None),
                    no_web=getattr(args, "no_web", False),
                    wait=args.wait,
                    queue_file=getattr(args, "queue_file", None),
                )
                if getattr(args, "json", False):
                    _print_json(res)
                else:
                    print(f"Enqueued {res['id']}: {res['name']} (devices: {res.get('devices') or 'default'}, status: {res['status']})")
            elif args.queue_command == "ls":
                jobs = queue_ls(status_filter=getattr(args, "status", None), queue_file=getattr(args, "queue_file", None))
                if getattr(args, "json", False):
                    _print_json(jobs)
                else:
                    print(format_queue_table(jobs))
            elif args.queue_command == "status":
                st = queue_status(job_id=getattr(args, "job_id", None), queue_file=getattr(args, "queue_file", None), lines=getattr(args, "lines", 20))
                if getattr(args, "json", False):
                    _print_json(st)
                else:
                    print(format_queue_status(st))
            elif args.queue_command in {"stop", "clear"}:
                stop_all = getattr(args, "all", False) or (args.queue_command == "clear")
                job_id = getattr(args, "job_id", None)
                if not job_id and not stop_all:
                    raise ValueError("Specify a job_id to stop (e.g. `hypergan queue stop <job_id>`) or pass `--all`.")
                res = queue_stop(job_id=job_id, stop_all=stop_all, queue_file=getattr(args, "queue_file", None))
                if getattr(args, "json", False):
                    _print_json(res)
                else:
                    print(res["message"])
            elif args.queue_command == "run":
                is_json = bool(getattr(args, "json", False))
                return queue_run(
                    devices=args.devices,
                    poll_interval=args.poll_interval,
                    once=args.once,
                    no_web=getattr(args, "no_web", False),
                    queue_file=getattr(args, "queue_file", None),
                    json_output=is_json,
                )
            elif args.queue_command == "install-service":
                path, instructions = queue_install_service(
                    devices=args.devices,
                    name=args.name,
                    dry_run=args.dry_run,
                )
                print(instructions)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        if _debug(args):
            import traceback
            traceback.print_exc()
        if exc.name and exc.name.split(".")[0] in {"torch", "particlegan", "numpy"}:
            print("error: training dependencies are missing; install 'hypergan[train]' in this environment", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError, argparse.ArgumentTypeError) as exc:
        if _debug(args):
            import traceback
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _debug(args):
    return bool(getattr(args, "debug", False)) or os.environ.get("HYPERGAN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
