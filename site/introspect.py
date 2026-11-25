"""Live & Torch-free introspection engine for the HyperGAN documentation site."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

# Ensure sibling modules in site/ are importable regardless of invocation mode
_SITE_DIR = Path(__file__).resolve().parent
if str(_SITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SITE_DIR))

import markdown_render  # noqa: E402

SITE_URL = "https://hypergan.github.io/HyperGAN/"
REPO_URL = "https://github.com/HyperGAN/HyperGAN"


# ---------------------------------------------------------------------------
# API grouping metadata (covers all 129 symbols in hypergan.__all__)
# ---------------------------------------------------------------------------
_API_GROUPS: list[tuple[str, str, str, list[str]]] = [
    (
        "config-models",
        "Configuration & Models",
        "Build, load, override, validate, and serialize hypergan-model/1 configurations and HNDL/PyTorch components.",
        [
            "model",
            "load",
            "load_bundle",
            "from_dict",
            "override",
            "save",
            "dumps",
            "lower",
            "resolve",
            "plain",
            "hndl",
            "module",
            "sampler",
            "adversarial",
            "reconstruction",
            "objective",
            "prior_loss",
            "lift",
            "load_recipe",
            "packaged",
            "packaged_list",
            "resolve_inheritance",
            "interpolate_variables",
            "collect_source_files",
            "import_path",
        ],
    ),
    (
        "scaffolding-validation",
        "Project Scaffolding, Datasets & Validation",
        "Scaffold projects, manage starter templates, validate configs, run preflight checks, and cache datasets.",
        [
            "init_project",
            "new",
            "list_templates",
            "validate",
            "preflight",
            "data_check",
            "download_dataset",
            "dataset_info",
            "list_datasets",
            "get_cache_dir",
            "ensure_cached_url",
            "resolve_or_cache",
            "find_workspace",
            "load_workspace",
            "resolve_device_aliases",
            "find_runs_root",
        ],
    ),
    (
        "training-control",
        "Training & Run Control",
        "Prepare, launch, resume, and control training runs (blocking or non-blocking via RunHandle) and on-demand requests.",
        [
            "prepare",
            "prepare_resume",
            "train",
            "resume",
            "launch",
            "stop",
            "request_checkpoint",
            "request_status",
            "request_sample",
            "request_stop",
            "RunHandle",
        ],
    ),
    (
        "discovery-inspection",
        "Run Discovery, Inspection & Telemetry",
        "Discover runs, read status/logs/metrics/events, compare experiments, prune checkpoints, and compact storage.",
        [
            "open_run",
            "inspect",
            "status",
            "ps",
            "logs",
            "tag",
            "rm",
            "list_runs",
            "compare",
            "diff",
            "gc",
            "gc_all",
            "compact_events",
            "prune_checkpoints",
            "resolve_checkpoint",
            "resolve_run_target",
            "register_run",
            "sync_registry",
            "generate_run_slug",
            "read_run_info",
            "metrics",
            "evaluations",
            "observations",
            "catalog",
            "events",
            "samples",
            "project",
            "contributions",
        ],
    ),
    (
        "sampling-eval",
        "Sampling, Evaluation & Media",
        "Generate samples, interpolate latents, run snapshot evaluations (e.g. FID), and handle image/audio outputs.",
        [
            "sample",
            "evaluate",
            "import_evaluation",
            "interpolate",
            "image",
            "audio",
        ],
    ),
    (
        "export-serving",
        "Standalone Export, Inference & Observatory",
        "Export standalone generators (bundle, safetensors, pt2, onnx), load EMA inference bundles, and manage the Web Observatory.",
        [
            "export",
            "export_generator",
            "extract_ema_bundle",
            "load_inference",
            "LoadedGenerator",
            "view",
            "serve",
            "viewer_status",
            "stop_viewer",
            "ensure_dashboard_running",
            "dashboard_status",
            "stop_dashboard",
            "stop_projector",
            "stop_all_servers",
        ],
    ),
    (
        "job-queue",
        "Multi-GPU Job Queue",
        "Schedule, inspect, stop, and execute sequential or multi-GPU training queues and systemd user services.",
        [
            "queue_add",
            "queue_ls",
            "queue_status",
            "queue_stop",
            "queue_run",
            "queue_install_service",
            "queue_clear",
        ],
    ),
    (
        "diagnostics",
        "Diagnostics & Introspection",
        "Describe model architectures, explain gradient routing, fingerprint configs, and introspect schema & HNDL operators.",
        [
            "describe",
            "explain",
            "fingerprint",
            "schema",
            "operators",
            "ops",
        ],
    ),
    (
        "types",
        "Data & Metrics Types",
        "Typed dataclasses and structured return records exposed across the HyperGAN Python API.",
        [
            "Model",
            "Run",
            "Sample",
            "Image",
            "Audio",
            "TrainResult",
            "SampleResult",
            "ModelDescription",
            "ValidationWarning",
            "RunStatus",
            "PsSnapshot",
            "RunInfo",
        ],
    ),
    (
        "exceptions",
        "Exceptions",
        "Structured exception types raised by configuration validation, offline cache enforcement, and run control.",
        [
            "ModelFileError",
            "ConfigError",
            "StopTraining",
            "OfflineError",
        ],
    ),
]


# ---------------------------------------------------------------------------
# CLI grouping & fallback descriptions
# ---------------------------------------------------------------------------
_CLI_FALLBACK_HELP: dict[str, str] = {
    "serve": "Start the foreground Web Observatory HTTP/SSE server for a run or runs root.",
    "server-status": "Check the status, URL, and PID of the local Web Observatory server.",
    "stop-server": "Stop a running background Web Observatory server process.",
}

_CLI_GROUPS: list[tuple[str, str, str, list[str]]] = [
    (
        "project-config",
        "Project & Config",
        "Scaffold new projects, inspect packaged recipes, validate model files, check datasets, and query schema or HNDL ops.",
        [
            "new",
            "validate",
            "preflight",
            "data-check",
            "download-data",
            "convert",
            "recipes",
            "schema",
            "ops",
            "version",
        ],
    ),
    (
        "training-lifecycle",
        "Training & Lifecycle",
        "Launch, resume, checkpoint, tag, gracefully stop, and remove training runs.",
        [
            "train",
            "resume",
            "stop",
            "checkpoint",
            "checkpoints",
            "tag",
            "rm",
        ],
    ),
    (
        "inspection-telemetry",
        "Inspection & Telemetry",
        "Monitor active runs, stream logs, compare experiments, explain networks, and query Foldlite DB event/metric views.",
        [
            "ps",
            "status",
            "logs",
            "inspect",
            "runs",
            "compare",
            "diff",
            "report",
            "model",
            "explain",
            "events",
            "metrics",
            "project",
            "contributions",
        ],
    ),
    (
        "sampling-export",
        "Sampling, Evaluation & Export",
        "Sample images or tensors from EMA checkpoints, evaluate FID/metrics, and export standalone inference artifacts.",
        [
            "sample",
            "evaluate",
            "export",
        ],
    ),
    (
        "queue-maintenance",
        "Queue & Maintenance",
        "Orchestrate multi-GPU experiment queues, install user services, and compact finished run directories.",
        [
            "queue",
            "gc",
        ],
    ),
    (
        "observatory-diagnostics",
        "Web Observatory & Diagnostics",
        "Launch or manage the local Web Observatory dashboard and run hardware/CUDA/NCCL readiness diagnostics.",
        [
            "view",
            "serve",
            "server-status",
            "stop-server",
            "doctor",
            "qualify",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Sidebar doc groups (content-auditor IA — all 49 docs/*.md + GLOSSARY + CHANGELOG)
# ---------------------------------------------------------------------------
_DOC_GROUP_DEFS: list[dict[str, Any]] = [
    {
        "id": "getting-started",
        "title": "Getting Started",
        "description": "Install HyperGAN, scaffold your first model, and understand hypergan-model/1.",
        "files": [
            "docs/README.md",
            "docs/custom-images.md",
            "docs/configuration.md",
            "docs/migration.md",
        ],
    },
    {
        "id": "core-guides",
        "title": "Core Workflows & Guides",
        "description": "Python API, HNDL language, custom PyTorch modules, previews, FID & benchmarks, Observatory, recovery, and deployment.",
        "files": [
            "docs/python-api.md",
            "docs/hndl-reference.md",
            "docs/custom-pytorch.md",
            "docs/particle-sliders.md",
            "docs/image-data.md",
            "docs/image-previews.md",
            "docs/image-fid.md",
            "docs/benchmarks.md",
            "docs/local-web.md",
            "docs/observation.md",
            "docs/recovery.md",
            "docs/deployment.md",
        ],
    },
    {
        "id": "hardware-scaling",
        "title": "Hardware & Distributed Scaling",
        "description": "Single-GPU CUDA, execution profiles, multi-GPU NCCL readiness, scaling benchmarks, Docker, and ROCm.",
        "files": [
            "docs/cuda.md",
            "docs/execution.md",
            "docs/execution-profiles.md",
            "docs/multigpu-readiness.md",
            "docs/scaling-benchmark.md",
            "docs/colab.md",
            "docs/docker.md",
            "docs/rocm.md",
        ],
    },
    {
        "id": "recipes-examples",
        "title": "Recipes & Examples",
        "description": "Measured CIFAR-10, colorization, Tiny Transformer, TransGAN, StyleTransformer, and AE-GAN recipes.",
        "files": [
            "docs/cifar-recipe.md",
            "docs/cifar-tiny-transformer.md",
            "docs/colorization.md",
            "docs/tiny-transformer.md",
            "docs/logos-tiny-transformer-features.md",
            "docs/transgan-stable.md",
            "docs/transgan-upstream-comparison.md",
            "docs/simple-styletransformer.md",
            "docs/logos-ae-gan.md",
            "docs/color-words.md",
            "docs/image-core.md",
        ],
    },
    {
        "id": "reference",
        "title": "Auto-Generated Reference",
        "description": "Live introspected Python API, CLI, TOML schema, HNDL operators, and examples gallery.",
        "files": [],
        "reference_links": [
            {"title": "Python API Reference", "output_path": "reference/api/index.html"},
            {"title": "CLI Reference", "output_path": "reference/cli/index.html"},
            {"title": "TOML Model Schema", "output_path": "reference/schema/index.html"},
            {"title": "HNDL Operators", "output_path": "reference/ops/index.html"},
            {"title": "Examples Gallery", "output_path": "examples/index.html"},
        ],
    },
    {
        "id": "internals",
        "title": "Internals",
        "description": "ParticleGAN & HNDL boundaries, run lifecycle, Foldlite DB event views, and distributed worker internals.",
        "files": [
            "docs/particlegan.md",
            "docs/hndl-upstream-requirements.md",
            "docs/run-lifecycle.md",
            "docs/foldlitedb.md",
            "docs/event-views.md",
            "docs/distributed.md",
            "docs/distributed-recovery.md",
            "docs/accumulation.md",
            "docs/cpu-workers.md",
            "docs/cpu-worker-service.md",
            "docs/replicated-run-service.md",
            "docs/replicated-cuda.md",
            "docs/replicated-observation.md",
            "docs/gpu-readiness.md",
        ],
    },
    {
        "id": "project",
        "title": "Project",
        "description": "Glossary, Changelog, CI releases, and S3 asset publishing.",
        "files": [
            "GLOSSARY.md",
            "CHANGELOG.md",
            "docs/releasing.md",
            "docs/s3-releases.md",
        ],
    },
]


# Map canonical HNDL category titles to stable slugs
_HNDL_CAT_TITLE_TO_ID: dict[str, str] = {
    "activation": "activation",
    "arithmetic": "arithmetic",
    "convolution": "convolution",
    "core": "core",
    "memory": "memory",
    "normalization": "normalization",
    "pretrained": "pretrained",
    "regularization": "regularization",
    "sequence": "sequence",
    "shape": "shape",
    "spatial": "spatial",
    "vision": "vision",
}

_HNDL_CAT_DISPLAY: dict[str, str] = {
    "activation": "Activation",
    "arithmetic": "Arithmetic",
    "convolution": "Convolution",
    "core": "Core & Linear",
    "memory": "Memory & Recurrence",
    "normalization": "Normalization",
    "pretrained": "Pretrained Providers",
    "regularization": "Regularization & Augmentation",
    "sequence": "Sequence & Attention",
    "shape": "Shape & Layout",
    "spatial": "Spatial & Pooling",
    "vision": "Vision Blocks",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "master"


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned = inspect.cleandoc(text).strip()
    if not cleaned:
        return ""
    first_para = cleaned.split("\n\n", 1)[0]
    first_line = " ".join(line.strip() for line in first_para.splitlines() if line.strip())
    return first_line


def _format_annotation(ann: Any) -> str:
    if ann is inspect.Parameter.empty or ann is dataclasses.MISSING:
        return ""
    if isinstance(ann, str):
        return ann
    if isinstance(ann, type):
        return ann.__name__
    s = str(ann)
    s = s.replace("typing.", "").replace("pathlib.", "").replace("hypergan.api.", "").replace("hypergan.run_info.", "")
    return s


def _format_default(val: Any) -> str:
    if val is inspect.Parameter.empty or val is dataclasses.MISSING:
        return ""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, (list, tuple, dict, set)):
        s = repr(val)
        return s if len(s) <= 80 else s[:77] + "..."
    return repr(val)


def _source_location(obj: Any, repo_root: Path, repo_url: str, commit: str) -> tuple[str, int, str]:
    try:
        unwrapped = inspect.unwrap(obj)
        src_file = inspect.getsourcefile(unwrapped)
        if not src_file:
            return "", 0, ""
        _, line_no = inspect.getsourcelines(unwrapped)
        rel = Path(src_file).resolve().relative_to(repo_root.resolve()).as_posix()
        return rel, int(line_no), f"{repo_url}/blob/{commit}/{rel}#L{line_no}"
    except Exception:
        return "", 0, ""


def _ast_format_args(args: ast.arguments) -> str:
    parts: list[str] = []
    pos_all = list(args.posonlyargs) + list(args.args)
    num_pos_defaults = len(args.defaults)
    default_offset = len(pos_all) - num_pos_defaults

    for idx, arg in enumerate(pos_all):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if idx >= default_offset:
            piece += f" = {ast.unparse(args.defaults[idx - default_offset])}"
        parts.append(piece)
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg is not None:
        piece = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            piece += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(piece)
    elif args.kwonlyargs:
        parts.append("*")

    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        piece = kwarg.arg
        if kwarg.annotation is not None:
            piece += f": {ast.unparse(kwarg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        parts.append(piece)

    if args.kwarg is not None:
        piece = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            piece += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(piece)

    return "(" + ", ".join(parts) + ")"


def _introspect_loaded_generator_ast(
    repo_root: Path, repo_url: str, commit: str, aliases: list[str]
) -> dict[str, Any]:
    """AST fallback for LoadedGenerator when torch is unavailable."""
    artifacts_path = repo_root / "src" / "hypergan" / "artifacts.py"
    source_file = "src/hypergan/artifacts.py"
    source_line = 1
    doc = "Callable standalone generator module with baked particle table and inference helpers."
    signature = "(bundle: dict | Path | str, device: str = 'cpu')"
    methods: list[dict[str, str]] = []

    if artifacts_path.is_file():
        tree = ast.parse(artifacts_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "LoadedGenerator":
                source_line = int(node.lineno)
                raw_doc = ast.get_docstring(node)
                if raw_doc:
                    doc = raw_doc
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            signature = _ast_format_args(item.args)
                        elif not item.name.startswith("_"):
                            ret = f" -> {ast.unparse(item.returns)}" if item.returns is not None else ""
                            is_prop = any(
                                isinstance(dec, ast.Name) and dec.id == "property"
                                for dec in item.decorator_list
                            )
                            m_sig = "(property)" if is_prop else f"{_ast_format_args(item.args)}{ret}"
                            m_doc = ast.get_docstring(item) or ""
                            methods.append(
                                {
                                    "name": item.name,
                                    "signature": m_sig,
                                    "summary": _first_sentence(m_doc) or f"{item.name} helper method.",
                                }
                            )
                break

    return {
        "name": "LoadedGenerator",
        "kind": "class",
        "signature": signature,
        "summary": _first_sentence(doc),
        "doc": inspect.cleandoc(doc),
        "doc_html": markdown_render.render_docstring_html(doc),
        "aliases": aliases,
        "fields": [],
        "methods": methods,
        "source_file": source_file,
        "source_line": source_line,
        "source_url": f"{repo_url}/blob/{commit}/{source_file}#L{source_line}",
    }


def _extract_class_methods(cls: type) -> list[dict[str, str]]:
    """Extract public methods and properties defined directly on ``cls``."""
    methods: list[dict[str, str]] = []
    for name, raw in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(raw, property):
            doc = raw.fget.__doc__ if raw.fget else ""
            methods.append(
                {
                    "name": name,
                    "signature": "(property)",
                    "summary": _first_sentence(doc or ""),
                }
            )
            continue
        if isinstance(raw, (classmethod, staticmethod)):
            func = raw.__func__
        elif inspect.isfunction(raw):
            func = raw
        else:
            continue
        try:
            sig = inspect.signature(func)
            params = [p for idx, p in enumerate(sig.parameters.values()) if not (idx == 0 and p.name in ("self", "cls"))]
            clean_sig = str(sig.replace(parameters=params))
        except Exception:
            clean_sig = "()"
        methods.append(
            {
                "name": name,
                "signature": clean_sig,
                "summary": _first_sentence(func.__doc__ or ""),
            }
        )
    return methods


def _introspect_api(repo_root: Path, repo_url: str, commit: str) -> dict[str, Any]:
    import hypergan
    import hypergan.api as hg_api

    all_symbols = list(hypergan.__all__)
    reverse_aliases: dict[str, list[str]] = {}
    for alias_name, canonical_name in getattr(hg_api, "ALIASES", {}).items():
        reverse_aliases.setdefault(canonical_name, []).append(alias_name)
    for k in reverse_aliases:
        reverse_aliases[k].sort()

    records: dict[str, dict[str, Any]] = {}
    for name in all_symbols:
        aliases = reverse_aliases.get(name, [])

        # Special handling for callable module `metrics`
        if name == "metrics":
            api_fn = getattr(hg_api, "metrics", None)
            mod_obj = getattr(hypergan, "metrics", None)
            sig = str(inspect.signature(api_fn)) if api_fn is not None else "(run, names=None)"
            fn_doc = inspect.cleandoc(api_fn.__doc__ or "") if api_fn is not None else ""
            mod_doc = inspect.cleandoc(getattr(mod_obj, "__doc__", "") or "")
            combined_doc = fn_doc
            if mod_doc and mod_doc not in fn_doc:
                combined_doc = (fn_doc + "\n\n" + mod_doc).strip() if fn_doc else mod_doc
            src_file, src_line, src_url = _source_location(
                api_fn or mod_obj, repo_root, repo_url, commit
            )
            records[name] = {
                "name": name,
                "kind": "module",
                "signature": sig,
                "summary": _first_sentence(fn_doc or mod_doc),
                "doc": combined_doc,
                "doc_html": markdown_render.render_docstring_html(combined_doc),
                "aliases": aliases,
                "fields": [],
                "methods": [],
                "source_file": src_file,
                "source_line": src_line,
                "source_url": src_url,
            }
            continue

        # Special handling for LoadedGenerator (may fail Torch-free)
        if name == "LoadedGenerator":
            try:
                obj = getattr(hypergan, name)
                # Verify live class loaded properly
                if not isinstance(obj, type):
                    raise ImportError("LoadedGenerator did not resolve to a class")
            except Exception:
                records[name] = _introspect_loaded_generator_ast(
                    repo_root, repo_url, commit, aliases
                )
                continue
        else:
            obj = getattr(hypergan, name)

        if isinstance(obj, type):
            if issubclass(obj, BaseException):
                kind = "exception"
            elif dataclasses.is_dataclass(obj):
                kind = "dataclass"
            else:
                kind = "class"
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "()"
            doc = inspect.cleandoc(obj.__doc__ or "")
            fields: list[dict[str, str]] = []
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    if f.default is not dataclasses.MISSING:
                        d_str = _format_default(f.default)
                    elif f.default_factory is not dataclasses.MISSING:
                        factory_name = getattr(f.default_factory, "__name__", str(f.default_factory))
                        d_str = f"{factory_name}()"
                    else:
                        d_str = ""
                    fields.append(
                        {
                            "name": f.name,
                            "type": _format_annotation(f.type),
                            "default": d_str,
                        }
                    )
            methods = _extract_class_methods(obj) if kind in ("dataclass", "class") else []
            src_file, src_line, src_url = _source_location(obj, repo_root, repo_url, commit)
            records[name] = {
                "name": name,
                "kind": kind,
                "signature": sig,
                "summary": _first_sentence(doc),
                "doc": doc,
                "doc_html": markdown_render.render_docstring_html(doc),
                "aliases": aliases,
                "fields": fields,
                "methods": methods,
                "source_file": src_file,
                "source_line": src_line,
                "source_url": src_url,
            }
        else:
            kind = "function"
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "()"
            doc = inspect.cleandoc(getattr(obj, "__doc__", "") or "")
            src_file, src_line, src_url = _source_location(obj, repo_root, repo_url, commit)
            records[name] = {
                "name": name,
                "kind": kind,
                "signature": sig,
                "summary": _first_sentence(doc),
                "doc": doc,
                "doc_html": markdown_render.render_docstring_html(doc),
                "aliases": aliases,
                "fields": [],
                "methods": [],
                "source_file": src_file,
                "source_line": src_line,
                "source_url": src_url,
            }

    # Assemble categorized groups with zero omissions
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []
    for gid, gtitle, gdesc, names in _API_GROUPS:
        items = []
        for sym in names:
            if sym in records:
                items.append(records[sym])
                assigned.add(sym)
        if items:
            groups.append(
                {
                    "id": gid,
                    "title": gtitle,
                    "description": gdesc,
                    "items": items,
                }
            )

    leftover = [records[sym] for sym in all_symbols if sym not in assigned]
    if leftover:
        groups.append(
            {
                "id": "other",
                "title": "Additional Public Symbols",
                "description": "Additional exported public symbols from hypergan.__all__.",
                "items": leftover,
            }
        )

    return {
        "total": len(records),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# CLI introspection
# ---------------------------------------------------------------------------
_GLOBAL_FLAG_DESTS = {"json_output", "quiet", "json"}
_GLOBAL_OPTION_STRINGS = {"--json", "-q", "--quiet"}


def _action_default_str(action: argparse.Action) -> str:
    if action.default is argparse.SUPPRESS or action.default is None:
        return ""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "false" if isinstance(action, argparse._StoreTrueAction) else "true"
    return str(action.default)


def _format_cli_help(help_text: str | None) -> str:
    if not help_text or help_text is argparse.SUPPRESS:
        return ""
    return re.sub(r"\s+", " ", str(help_text).replace("%%", "%")).strip()


def _extract_command(
    name: str,
    parser: argparse.ArgumentParser,
    *,
    parent_cmd: str,
    aliases: list[str],
    summary_help: str | None,
) -> dict[str, Any]:
    full_command = f"{parent_cmd} {name}".strip()
    summary = _format_cli_help(summary_help) or _CLI_FALLBACK_HELP.get(name, "")
    description = _format_cli_help(parser.description) or summary

    # Clean usage string
    raw_usage = parser.format_usage().strip()
    if raw_usage.lower().startswith("usage:"):
        raw_usage = raw_usage[len("usage:") :].strip()
    # Replace argv[0] prefix with canonical `hypergan ...`
    tokens = raw_usage.split()
    if tokens:
        if tokens[0] != "hypergan":
            tokens[0] = "hypergan"
        usage = " ".join(tokens)
    else:
        usage = full_command

    positionals: list[dict[str, Any]] = []
    options: list[dict[str, Any]] = []
    subcommands: list[dict[str, Any]] = []

    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            subcommands.extend(_extract_subparsers(action, parent_cmd=full_command))
            continue
        if action.help is argparse.SUPPRESS:
            continue

        if not action.option_strings:
            metavar = action.metavar if isinstance(action.metavar, str) else action.dest.upper()
            required = action.nargs not in ("?", "*")
            positionals.append(
                {
                    "name": action.dest,
                    "metavar": metavar,
                    "help": _format_cli_help(action.help),
                    "required": required,
                }
            )
        else:
            # Skip global flags documented once in global_flags banner
            if any(flag in _GLOBAL_OPTION_STRINGS for flag in action.option_strings):
                continue
            flags_str = ", ".join(action.option_strings)
            if isinstance(action.metavar, tuple):
                metavar = " ".join(str(m) for m in action.metavar)
            elif action.metavar:
                metavar = str(action.metavar)
            elif isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse._CountAction,
                    argparse._AppendConstAction,
                    argparse._StoreConstAction,
                ),
            ):
                metavar = ""
            else:
                metavar = action.dest.upper()

            choices = [str(c) for c in action.choices] if action.choices else []
            options.append(
                {
                    "flags": flags_str,
                    "metavar": metavar,
                    "default": _action_default_str(action),
                    "choices": choices,
                    "help": _format_cli_help(action.help),
                }
            )

    return {
        "name": name,
        "full_command": full_command,
        "aliases": aliases,
        "usage": usage,
        "summary": summary,
        "description": description,
        "positionals": positionals,
        "options": options,
        "subcommands": subcommands,
    }


def _extract_subparsers(
    sub_action: argparse._SubParsersAction, *, parent_cmd: str
) -> list[dict[str, Any]]:
    help_map: dict[str, str] = {}
    for ca in sub_action._choices_actions:
        if ca.help is not argparse.SUPPRESS and ca.help:
            help_map[ca.dest] = str(ca.help)

    # Deduplicate aliases pointing to identical parser instances
    seen_ids: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for cmd_name, sub_parser in sub_action.choices.items():
        pid = id(sub_parser)
        if pid in seen_ids:
            seen_ids[pid]["aliases"].append(cmd_name)
            continue
        entry = {
            "name": cmd_name,
            "parser": sub_parser,
            "aliases": [],
            "help": help_map.get(cmd_name),
        }
        seen_ids[pid] = entry
        ordered.append(entry)

    commands: list[dict[str, Any]] = []
    for entry in ordered:
        commands.append(
            _extract_command(
                entry["name"],
                entry["parser"],
                parent_cmd=parent_cmd,
                aliases=entry["aliases"],
                summary_help=entry["help"],
            )
        )
    return commands


def _introspect_cli() -> dict[str, Any]:
    import hypergan.cli as hg_cli

    root_parser = hg_cli._parser()
    global_flags_seen: dict[str, dict[str, str]] = {}

    def _collect_global_flags(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if action.option_strings and any(
                f in _GLOBAL_OPTION_STRINGS for f in action.option_strings
            ):
                key = ", ".join(action.option_strings)
                if key not in global_flags_seen:
                    global_flags_seen[key] = {
                        "flags": key,
                        "metavar": str(action.metavar) if action.metavar else "",
                        "help": _format_cli_help(action.help),
                    }
            elif isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    _collect_global_flags(sub)

    _collect_global_flags(root_parser)

    top_commands: list[dict[str, Any]] = []
    for action in root_parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            top_commands = _extract_subparsers(action, parent_cmd="hypergan")
            break

    by_name = {cmd["name"]: cmd for cmd in top_commands}
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []

    for gid, gtitle, gdesc, cmd_names in _CLI_GROUPS:
        cmds = []
        for cname in cmd_names:
            if cname in by_name:
                cmds.append(by_name[cname])
                assigned.add(cname)
        if cmds:
            groups.append(
                {
                    "id": gid,
                    "title": gtitle,
                    "description": gdesc,
                    "commands": cmds,
                }
            )

    leftover = [cmd for cmd in top_commands if cmd["name"] not in assigned]
    if leftover:
        groups.append(
            {
                "id": "other",
                "title": "Additional Commands",
                "description": "Additional CLI subcommands.",
                "commands": leftover,
            }
        )

    def _count_commands(cmds: list[dict[str, Any]]) -> int:
        return sum(1 + _count_commands(c["subcommands"]) for c in cmds)

    total = _count_commands(top_commands)

    return {
        "total": total,
        "global_flags": list(global_flags_seen.values()),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# TOML Schema introspection
# ---------------------------------------------------------------------------
def _format_schema_default(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    return repr(val)


def _introspect_schema() -> dict[str, Any]:
    from hypergan.schema_info import MODEL_SCHEMA

    required_top = set(MODEL_SCHEMA.get("required", []))
    raw_sections = MODEL_SCHEMA.get("sections", {})

    top_level: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    total_fields = 0

    header_overrides = {
        "losses": "[[losses]]",
        "networks": "[networks.<name>]",
        "samplers": "[samplers.<name>]",
        "evaluations": "[evaluations.<id>]",
    }

    for sec_name, spec in raw_sections.items():
        sec_type = spec.get("type", "object")
        raw_fields = (
            spec.get("fields")
            or spec.get("properties")
            or (spec.get("items") or {}).get("fields")
        )
        if sec_type not in ("object", "array") and raw_fields is None:
            top_level.append(
                {
                    "name": sec_name,
                    "type": sec_type,
                    "default": _format_schema_default(spec.get("default")),
                    "choices": [str(c) for c in spec.get("choices", [])],
                    "required": sec_name in required_top,
                    "description": spec.get("description", ""),
                }
            )
            continue

        sec_required = set(spec.get("required", []))
        if "items" in spec and isinstance(spec["items"], dict):
            sec_required |= set(spec["items"].get("required", []))

        field_entries: list[dict[str, Any]] = []
        for fname, fspec in (raw_fields or {}).items():
            field_entries.append(
                {
                    "name": fname,
                    "type": fspec.get("type", "any"),
                    "default": _format_schema_default(fspec.get("default")),
                    "choices": [str(c) for c in fspec.get("choices", [])],
                    "required": fname in sec_required or bool(fspec.get("required", False)),
                    "description": fspec.get("description", ""),
                }
            )
        total_fields += len(field_entries)
        sections.append(
            {
                "id": sec_name,
                "name": sec_name,
                "toml_header": header_overrides.get(sec_name, f"[{sec_name}]"),
                "description": spec.get("description", ""),
                "fields": field_entries,
            }
        )

    return {
        "total_sections": len(top_level) + len(sections),
        "total_fields": total_fields,
        "top_level": top_level,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# HNDL Operators introspection (live + Torch-free fallback)
# ---------------------------------------------------------------------------
_PORT_RE = re.compile(r"Port\(name='([^']+)',\s*pattern=\((.*?)\),\s*dtype='([^']+)'")
_SYM_RE = re.compile(r"Sym\(name='([^']+)'(?:,\s*scale=([^)]+))?\)")


def _format_port_signature(port_str: str) -> str:
    """Format hndl Port repr string cleanly, e.g. x[B, C, H, W]."""
    m = _PORT_RE.search(port_str)
    if not m:
        return port_str
    pname, pattern_raw, _dtype = m.group(1), m.group(2), m.group(3)

    def _repl_sym(sm: re.Match[str]) -> str:
        sname = sm.group(1)
        scale = (sm.group(2) or "1").strip()
        return sname if scale in ("1", "1.0") else f"{scale}*{sname}"

    cleaned = _SYM_RE.sub(_repl_sym, pattern_raw)
    cleaned = cleaned.replace("'", "").replace('"', "").strip().rstrip(",")
    dims = ", ".join(part.strip() for part in cleaned.split(",") if part.strip())
    return f"{pname}[{dims}]" if dims else pname


def _extract_diff_augment_ast(repo_root: Path) -> dict[str, Any] | None:
    """Parse @operator(...) metadata for diff_augment from hndl_augmentation.py."""
    aug_path = repo_root / "src" / "hypergan" / "hndl_augmentation.py"
    if not aug_path.is_file():
        return None
    tree = ast.parse(aug_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "operator"
            ):
                continue
            op_name = "diff_augment"
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                op_name = dec.args[0].value
            kw = {k.arg: k.value for k in dec.keywords if k.arg}
            summary = ""
            if "summary" in kw and isinstance(kw["summary"], ast.Constant):
                summary = str(kw["summary"].value)
            shape = ""
            if "shape" in kw and isinstance(kw["shape"], ast.Constant):
                shape = str(kw["shape"].value)
            category = "regularization"
            if "category" in kw and isinstance(kw["category"], ast.Constant):
                category = str(kw["category"].value)
            args_list: list[dict[str, str]] = []
            if "args" in kw and isinstance(kw["args"], ast.Dict):
                for k_node, v_node in zip(kw["args"].keys, kw["args"].values):
                    if not (isinstance(k_node, ast.Constant) and isinstance(v_node, ast.Call)):
                        continue
                    aname = str(k_node.value)
                    atype = ast.unparse(v_node.args[0]) if len(v_node.args) >= 1 else "any"
                    adefault = ast.unparse(v_node.args[1]) if len(v_node.args) >= 2 else ""
                    ahelp = ""
                    for akw in v_node.keywords:
                        if akw.arg == "help" and isinstance(akw.value, ast.Constant):
                            ahelp = str(akw.value.value)
                    args_list.append(
                        {"name": aname, "type": atype, "default": adefault, "help": ahelp}
                    )
            inputs, outputs = [], []
            if "->" in shape:
                lhs, rhs = shape.split("->", 1)
                inputs = [lhs.strip()] if lhs.strip() else []
                outputs = [rhs.strip()] if rhs.strip() else []
            doc = ast.get_docstring(node) or ""
            return {
                "name": op_name,
                "category": category,
                "summary": summary,
                "description": inspect.cleandoc(doc) if doc else summary,
                "inputs": inputs,
                "outputs": outputs,
                "args": args_list,
            }
    return None


def _parse_arg_signature_list(sig_inner: str) -> list[dict[str, str]]:
    """Parse ``a, b=1, mode='mean'`` into structured arg dicts."""
    if not sig_inner.strip():
        return []
    args: list[dict[str, str]] = []
    # Split on top-level commas (respecting nested parens/brackets)
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in sig_inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())

    for part in parts:
        if not part:
            continue
        if "=" in part:
            aname, adefault = part.split("=", 1)
            aname = aname.strip()
            adefault = adefault.strip()
            if adefault in ("True", "False"):
                atype = "bool"
            elif adefault in ("None", "()"):
                atype = "any"
            elif adefault.startswith(("'", '"')):
                atype = "str"
            elif re.match(r"^-?\d+$", adefault):
                atype = "int"
            elif re.match(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", adefault, re.I):
                atype = "float"
            else:
                atype = "any"
            args.append({"name": aname, "type": atype, "default": adefault, "help": ""})
        else:
            args.append({"name": part, "type": "any", "default": "", "help": ""})
    return args


def _introspect_ops_fallback(repo_root: Path) -> dict[str, Any]:
    """Torch-free fallback parsing docs/hndl-reference.md + AST of hndl_augmentation.py."""
    ref_path = repo_root / "docs" / "hndl-reference.md"
    diff_aug = _extract_diff_augment_ast(repo_root)

    categories_map: dict[str, list[dict[str, Any]]] = {}
    seen_ops: set[str] = set()

    if ref_path.is_file():
        lines = ref_path.read_text(encoding="utf-8").splitlines()
        current_cat: str | None = None
        last_op: dict[str, Any] | None = None
        in_catalog = False

        cat_re = re.compile(r"^###\s+2\.\d+\s+([A-Za-z]+)")
        op_re = re.compile(r"^\d+\.\s+\*\*`([a-zA-Z0-9_]+)\((.*)\)`\*\*:\s*(.*)$")

        for line in lines:
            if line.startswith("## 2."):
                in_catalog = True
                continue
            if in_catalog and line.startswith("## ") and not line.startswith("## 2."):
                break
            if not in_catalog:
                continue

            cm = cat_re.match(line)
            if cm:
                first_word = cm.group(1).lower()
                current_cat = _HNDL_CAT_TITLE_TO_ID.get(first_word, first_word)
                categories_map.setdefault(current_cat, [])
                last_op = None
                continue

            om = op_re.match(line)
            if om and current_cat is not None:
                op_name = om.group(1).strip()
                if op_name in seen_ops:
                    last_op = None
                    continue
                seen_ops.add(op_name)
                if op_name == "diff_augment" and diff_aug is not None:
                    entry = dict(diff_aug)
                else:
                    sig_inner = om.group(2).strip()
                    summary = om.group(3).strip()
                    entry = {
                        "name": op_name,
                        "category": current_cat,
                        "summary": summary,
                        "description": summary,
                        "inputs": ["x"],
                        "outputs": ["out"],
                        "args": _parse_arg_signature_list(sig_inner),
                    }
                categories_map[current_cat].append(entry)
                last_op = entry
                continue

            # Handle continuation lines (e.g. pretrained)
            if last_op is not None and line.startswith("    ") and line.strip():
                extra = line.strip()
                if not last_op["summary"]:
                    last_op["summary"] = extra
                    last_op["description"] = extra
                else:
                    last_op["description"] = (last_op["description"] + " " + extra).strip()

    # Ensure diff_augment is always present even if docs/hndl-reference.md changed
    if diff_aug is not None and "diff_augment" not in seen_ops:
        categories_map.setdefault(diff_aug["category"], []).append(diff_aug)
        seen_ops.add("diff_augment")

    categories: list[dict[str, Any]] = []
    total = 0
    for cat_id in sorted(categories_map.keys()):
        ops_list = sorted(categories_map[cat_id], key=lambda o: o["name"])
        total += len(ops_list)
        categories.append(
            {
                "id": cat_id,
                "name": _HNDL_CAT_DISPLAY.get(cat_id, cat_id.replace("_", " ").title()),
                "count": len(ops_list),
                "ops": ops_list,
            }
        )

    return {
        "total": total,
        "mode": "fallback",
        "categories": categories,
    }


def _introspect_ops(repo_root: Path, *, force_fallback: bool = False) -> dict[str, Any]:
    if not force_fallback:
        try:
            from hypergan.ops_info import get_operators

            raw_ops = get_operators()
            categories_map: dict[str, list[dict[str, Any]]] = {}
            for raw in raw_ops:
                cat_id = (raw.get("category") or "general").strip().lower()
                args_list: list[dict[str, str]] = []
                for aname, ainfo in (raw.get("args") or {}).items():
                    default_raw = ainfo.get("default")
                    default_str = "" if default_raw is None else repr(default_raw) if isinstance(default_raw, str) else str(default_raw)
                    args_list.append(
                        {
                            "name": aname,
                            "type": str(ainfo.get("type") or "any"),
                            "default": default_str,
                            "help": str(ainfo.get("help") or ""),
                        }
                    )
                inputs = [_format_port_signature(p) for p in (raw.get("inputs") or [])]
                outputs = [_format_port_signature(p) for p in (raw.get("outputs") or [])]
                summary = (raw.get("summary") or "").strip()
                doc = (raw.get("doc") or "").strip()
                entry = {
                    "name": raw["name"],
                    "category": cat_id,
                    "summary": summary or _first_sentence(doc),
                    "description": doc or summary,
                    "inputs": inputs,
                    "outputs": outputs,
                    "args": args_list,
                }
                categories_map.setdefault(cat_id, []).append(entry)

            categories: list[dict[str, Any]] = []
            total = 0
            for cat_id in sorted(categories_map.keys()):
                ops_list = sorted(categories_map[cat_id], key=lambda o: o["name"])
                total += len(ops_list)
                categories.append(
                    {
                        "id": cat_id,
                        "name": _HNDL_CAT_DISPLAY.get(cat_id, cat_id.replace("_", " ").title()),
                        "count": len(ops_list),
                        "ops": ops_list,
                    }
                )
            return {
                "total": total,
                "mode": "live",
                "categories": categories,
            }
        except Exception:
            pass

    return _introspect_ops_fallback(repo_root)


# ---------------------------------------------------------------------------
# Examples, Packaged Recipes & Templates introspection
# ---------------------------------------------------------------------------
_EXAMPLE_DOC_SLUGS: dict[str, str] = {
    "cifar10-tiny-transformer": "cifar-tiny-transformer",
    "cifar10-transgan": "transgan-stable",
    "cifar-transgan-resnet18": "cifar-recipe",
    "cifar-transgan-scratch": "transgan-stable",
    "colorization-128": "colorization",
    "image-folder-colorization": "colorization",
    "image-folder-128": "custom-images",
    "image-folder-dcgan": "custom-images",
    "image-folder-tiny-transformer": "tiny-transformer",
    "color-words": "color-words",
    "gaussian-grid": "configuration",
    "paired-linear": "configuration",
    "cpu-single": "execution",
    "cpu-replicated": "execution",
    "cuda-replicated-nccl": "cuda",
}


def _extract_toml_meta(path: Path) -> tuple[str, str, list[str]]:
    """Extract (title, summary, tags) from a hypergan-model/1 or execution TOML file."""
    text = path.read_text(encoding="utf-8")
    tags: list[str] = []
    title = path.stem.replace("-", " ").replace("_", " ").title()
    summary = ""
    try:
        parsed = tomllib.loads(text)
        if parsed.get("name"):
            title = str(parsed["name"])
        if parsed.get("description"):
            summary = str(parsed["description"]).strip()
        defaults = parsed.get("defaults")
        if defaults:
            tags.append(str(defaults))
        data = parsed.get("data") or parsed.get("dataset") or {}
        if isinstance(data, dict):
            if data.get("builtin"):
                tags.append(str(data["builtin"]))
            if data.get("width") and data.get("height"):
                tags.append(f"{data['width']}x{data['height']}")
        if "networks" in parsed and isinstance(parsed["networks"], dict):
            tags.append(f"{len(parsed['networks'])} networks")
        if parsed.get("backend"):
            tags.append(str(parsed["backend"]))
        if parsed.get("devices") or parsed.get("ranks"):
            tags.append("distributed")
    except Exception:
        pass
    if not summary:
        # Check leading comment lines
        comments = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                comments.append(s.lstrip("#").strip())
            elif s:
                break
        summary = " ".join(comments[:2]) or f"HyperGAN TOML configuration ({path.name})."
    return title, summary, tags


def _extract_python_meta(path: Path) -> tuple[str, str, list[str]]:
    text = path.read_text(encoding="utf-8")
    summary = ""
    try:
        tree = ast.parse(text)
        doc = ast.get_docstring(tree)
        if doc:
            summary = _first_sentence(doc)
    except Exception:
        pass
    if not summary:
        summary = f"Config-first Python API walkthrough ({path.name})."
    title = path.stem.replace("_", " ").title()
    tags = ["python-api", "config-first"]
    if "replicated" in path.stem:
        tags.append("distributed")
    return title, summary, tags


def _extract_hndl_meta(path: Path) -> tuple[str, str, list[str]]:
    text = path.read_text(encoding="utf-8")
    comments = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
        elif s:
            break
    summary = " ".join(comments[:2]) or f"Reusable HNDL neural network architecture ({path.name})."
    stem = path.stem
    title = stem.replace("-", " ").title()
    tags = ["hndl"]
    if "generator" in stem:
        tags.append("generator")
    elif "discriminator" in stem or "critic" in stem:
        tags.append("critic")
    for res in ("32", "64", "128", "256"):
        if stem.endswith(f"-{res}") or f"-{res}-" in stem:
            tags.append(f"{res}px")
            break
    if "dinov3" in stem:
        tags.append("dinov3")
    elif "resnet" in stem:
        tags.append("resnet18")
    elif "transgan" in stem:
        tags.append("transgan")
    elif "tiny-transformer" in stem:
        tags.append("tiny-transformer")
    return title, summary, tags


def _introspect_examples(repo_root: Path, repo_url: str, commit: str) -> dict[str, list[dict[str, Any]]]:
    import hypergan

    # 1. Packaged recipes (src/hypergan/models/*.toml)
    packaged_recipes: list[dict[str, Any]] = []
    for rec in sorted(hypergan.packaged_list(), key=lambda r: r["name"]):
        rpath = Path(rec["path"])
        rel = rpath.resolve().relative_to(repo_root.resolve()).as_posix()
        code = rpath.read_text(encoding="utf-8")
        _title, summary, tags = _extract_toml_meta(rpath)
        if rec.get("scope"):
            tags = [rec["scope"], *tags]
        packaged_recipes.append(
            {
                "id": f"packaged-{rec['name']}",
                "title": rec["name"],
                "category": "packaged_recipes",
                "category_label": "Packaged Recipe",
                "path": rel,
                "github_url": f"{repo_url}/blob/{commit}/{rel}",
                "summary": summary,
                "tags": list(dict.fromkeys(tags)),
                "run_command": f"hypergan train {rec['name']} --steps 100",
                "language": "toml",
                "code": code,
                "code_html": markdown_render.highlight_code(code, "toml"),
                "doc_slug": _EXAMPLE_DOC_SLUGS.get(rec["name"]),
            }
        )

    # 2. Starter templates (hypergan new --list)
    starter_templates: list[dict[str, Any]] = []
    for tpl in hypergan.list_templates():
        tname = tpl["name"]
        tdesc = tpl.get("description", "")
        resolutions = tpl.get("resolutions") or []
        tags = ["template"] + ([f"{resolutions[0]}..{resolutions[-1]}px"] if resolutions else ["2d"])
        if resolutions:
            run_cmd = f"hypergan new my-{tname} --template {tname} --size {resolutions[0]}"
        else:
            run_cmd = f"hypergan new my-{tname} --template {tname}"
        snippet = f"# Scaffold a new {tname} project\n{run_cmd}\ncd my-{tname}\nhypergan validate model.toml\nhypergan train model.toml\n"
        starter_templates.append(
            {
                "id": f"template-{tname}",
                "name": tname,
                "title": tname,
                "description": tdesc,
                "category": "starter_templates",
                "category_label": "Starter Template",
                "path": "src/hypergan/recipe_generator.py",
                "github_url": f"{repo_url}/blob/{commit}/src/hypergan/recipe_generator.py",
                "summary": tdesc,
                "tags": tags,
                "run_command": run_cmd,
                "language": "bash",
                "code": snippet,
                "code_html": markdown_render.highlight_code(snippet, "bash"),
                "doc_slug": _EXAMPLE_DOC_SLUGS.get(tname, "configuration"),
            }
        )

    # 3. Active TOML examples (examples/*.toml)
    toml_examples: list[dict[str, Any]] = []
    examples_dir = repo_root / "examples"
    if examples_dir.is_dir():
        for p in sorted(examples_dir.glob("*.toml")):
            rel = p.relative_to(repo_root).as_posix()
            code = p.read_text(encoding="utf-8")
            title, summary, tags = _extract_toml_meta(p)
            toml_examples.append(
                {
                    "id": f"toml-{p.stem}",
                    "title": title,
                    "category": "toml_examples",
                    "category_label": "TOML Config",
                    "path": rel,
                    "github_url": f"{repo_url}/blob/{commit}/{rel}",
                    "summary": summary,
                    "tags": list(dict.fromkeys(tags)),
                    "run_command": f"hypergan train {rel} --steps 100",
                    "language": "toml",
                    "code": code,
                    "code_html": markdown_render.highlight_code(code, "toml"),
                    "doc_slug": _EXAMPLE_DOC_SLUGS.get(p.stem),
                }
            )

    # 4. Python API examples (examples/api/config_first/demo_*.py)
    python_examples: list[dict[str, Any]] = []
    api_examples_dir = examples_dir / "api" / "config_first"
    if api_examples_dir.is_dir():
        for p in sorted(api_examples_dir.glob("demo_*.py")):
            rel = p.relative_to(repo_root).as_posix()
            code = p.read_text(encoding="utf-8")
            title, summary, tags = _extract_python_meta(p)
            python_examples.append(
                {
                    "id": f"py-{p.stem}",
                    "title": title,
                    "category": "python_examples",
                    "category_label": "Python API",
                    "path": rel,
                    "github_url": f"{repo_url}/blob/{commit}/{rel}",
                    "summary": summary,
                    "tags": tags,
                    "run_command": f"python {rel}",
                    "language": "python",
                    "code": code,
                    "code_html": markdown_render.highlight_code(code, "python"),
                    "doc_slug": "python-api",
                }
            )

    # 5. Reusable HNDL networks (examples/networks/*.hndl)
    hndl_networks: list[dict[str, Any]] = []
    networks_dir = examples_dir / "networks"
    if networks_dir.is_dir():
        for p in sorted(networks_dir.glob("*.hndl")):
            rel = p.relative_to(repo_root).as_posix()
            code = p.read_text(encoding="utf-8")
            title, summary, tags = _extract_hndl_meta(p)
            hndl_networks.append(
                {
                    "id": f"hndl-{p.stem}",
                    "title": title,
                    "category": "hndl_networks",
                    "category_label": "HNDL Network",
                    "path": rel,
                    "github_url": f"{repo_url}/blob/{commit}/{rel}",
                    "summary": summary,
                    "tags": list(dict.fromkeys(tags)),
                    "run_command": f"hypergan explain {rel}",
                    "language": "hndl",
                    "code": code,
                    "code_html": markdown_render.highlight_code(code, "hndl"),
                    "doc_slug": "hndl-reference",
                }
            )

    # 6. Execution profiles (examples/execution/*.toml)
    execution_profiles: list[dict[str, Any]] = []
    exec_dir = examples_dir / "execution"
    if exec_dir.is_dir():
        for p in sorted(exec_dir.glob("*.toml")):
            rel = p.relative_to(repo_root).as_posix()
            code = p.read_text(encoding="utf-8")
            title, summary, tags = _extract_toml_meta(p)
            execution_profiles.append(
                {
                    "id": f"exec-{p.stem}",
                    "title": title,
                    "category": "execution_profiles",
                    "category_label": "Execution Profile",
                    "path": rel,
                    "github_url": f"{repo_url}/blob/{commit}/{rel}",
                    "summary": summary,
                    "tags": list(dict.fromkeys(["execution-profile", *tags])),
                    "run_command": f"hypergan preflight gaussian-grid --profile {rel}",
                    "language": "toml",
                    "code": code,
                    "code_html": markdown_render.highlight_code(code, "toml"),
                    "doc_slug": _EXAMPLE_DOC_SLUGS.get(p.stem, "execution"),
                }
            )

    return {
        "packaged_recipes": packaged_recipes,
        "starter_templates": starter_templates,
        "toml_examples": toml_examples,
        "python_examples": python_examples,
        "hndl_networks": hndl_networks,
        "execution_profiles": execution_profiles,
    }


# ---------------------------------------------------------------------------
# Documentation index (all 49 docs/*.md + GLOSSARY.md + CHANGELOG.md)
# ---------------------------------------------------------------------------
def _extract_md_title_desc(raw_md: str, fallback_title: str) -> tuple[str, str]:
    title = ""
    desc = ""
    lines = raw_md.splitlines()
    in_code = False
    para_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not title and stripped.startswith("# "):
            title = stripped[2:].strip().replace("`", "")
            continue
        if title and not desc:
            if not stripped:
                if para_lines:
                    desc = " ".join(para_lines)
                    break
                continue
            if stripped.startswith(("#", ">", "-", "*", "|", "---")):
                if para_lines:
                    desc = " ".join(para_lines)
                    break
                continue
            para_lines.append(stripped)

    if not desc and para_lines:
        desc = " ".join(para_lines)
    # Strip markdown link syntax from plain description
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
    desc = re.sub(r"`([^`]+)`", r"\1", desc)
    if len(desc) > 220:
        desc = desc[:217].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
    return title or fallback_title, desc


def _doc_route(source_path: str) -> tuple[str, str]:
    if source_path == "GLOSSARY.md":
        return "glossary", "glossary/index.html"
    if source_path == "CHANGELOG.md":
        return "changelog", "changelog/index.html"
    if source_path == "docs/README.md":
        return "overview", "docs/overview/index.html"
    stem = Path(source_path).stem
    return stem, f"docs/{stem}/index.html"


def _introspect_docs(
    repo_root: Path, repo_url: str, commit: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Collect all docs/*.md + GLOSSARY.md + CHANGELOG.md
    discovered: set[str] = set()
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        for md_file in sorted(docs_dir.glob("*.md")):
            discovered.add(f"docs/{md_file.name}")
    for root_doc in ("GLOSSARY.md", "CHANGELOG.md"):
        if (repo_root / root_doc).is_file():
            discovered.add(root_doc)

    # Map every discovered doc to a sidebar group with zero omissions
    file_to_group: dict[str, tuple[str, str]] = {}
    ordered_group_files: dict[str, list[str]] = {}
    for gdef in _DOC_GROUP_DEFS:
        gid = gdef["id"]
        gtitle = gdef["title"]
        ordered_group_files[gid] = []
        for fpath in gdef["files"]:
            if fpath in discovered:
                file_to_group[fpath] = (gid, gtitle)
                ordered_group_files[gid].append(fpath)

    # Any newly added docs/*.md not listed explicitly goes into Internals
    for fpath in sorted(discovered):
        if fpath not in file_to_group:
            file_to_group[fpath] = ("internals", "Internals")
            ordered_group_files["internals"].append(fpath)

    docs_list: list[dict[str, Any]] = []
    docs_by_source: dict[str, dict[str, Any]] = {}

    for gdef in _DOC_GROUP_DEFS:
        gid = gdef["id"]
        gtitle = gdef["title"]
        for source_path in ordered_group_files[gid]:
            full_path = repo_root / source_path
            raw_md = full_path.read_text(encoding="utf-8")
            slug, output_path = _doc_route(source_path)
            fallback_title = slug.replace("-", " ").title()
            title, description = _extract_md_title_desc(raw_md, fallback_title)
            entry = {
                "slug": slug,
                "output_path": output_path,
                "source_path": source_path,
                "github_url": f"{repo_url}/blob/{commit}/{source_path}",
                "title": title,
                "description": description,
                "group_id": gid,
                "group_title": gtitle,
                "raw_markdown": raw_md,
            }
            docs_list.append(entry)
            docs_by_source[source_path] = entry

    doc_groups: list[dict[str, Any]] = []
    for gdef in _DOC_GROUP_DEFS:
        gid = gdef["id"]
        group_docs = [docs_by_source[f] for f in ordered_group_files[gid] if f in docs_by_source]
        doc_groups.append(
            {
                "id": gid,
                "title": gdef["title"],
                "description": gdef["description"],
                "items": [
                    {
                        "slug": d["slug"],
                        "title": d["title"],
                        "output_path": d["output_path"],
                        "source_path": d["source_path"],
                    }
                    for d in group_docs
                ],
                "reference_links": list(gdef.get("reference_links", [])),
            }
        )

    return docs_list, doc_groups


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def collect_site_data(repo_root: Path, *, _force_ops_fallback: bool = False) -> dict[str, Any]:
    """Introspect the HyperGAN repository and return the complete site data model."""
    root = Path(repo_root).resolve()
    src_dir = root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import hypergan

    version = getattr(hypergan, "__version__", "0.0.0")
    commit = _git_commit(root)

    api_data = _introspect_api(root, REPO_URL, commit)
    cli_data = _introspect_cli()
    schema_data = _introspect_schema()
    ops_data = _introspect_ops(root, force_fallback=_force_ops_fallback)
    examples_data = _introspect_examples(root, REPO_URL, commit)
    docs_list, doc_groups = _introspect_docs(root, REPO_URL, commit)

    total_examples = sum(len(items) for items in examples_data.values())

    meta = {
        "version": version,
        "commit": commit,
        "site_url": SITE_URL,
        "repo_url": REPO_URL,
        "ops_mode": ops_data["mode"],
        "counts": {
            "api": api_data["total"],
            "cli": cli_data["total"],
            "schema_sections": schema_data["total_sections"],
            "schema_fields": schema_data["total_fields"],
            "ops": ops_data["total"],
            "examples": total_examples,
            "docs": len(docs_list),
        },
    }

    return {
        "meta": meta,
        "api": api_data,
        "cli": cli_data,
        "schema": schema_data,
        "ops": ops_data,
        "examples": examples_data,
        "docs": docs_list,
        "doc_groups": doc_groups,
    }
