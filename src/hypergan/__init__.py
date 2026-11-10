"""HyperGAN's lightweight public package.

Training dependencies are loaded only when a training or sampling API is used.
"""
from typing import TYPE_CHECKING

__version__ = "2.0.0a3"

if TYPE_CHECKING:
    from . import metrics as metrics
    from .api import (
        Audio as Audio,
        ConfigError as ConfigError,
        Image as Image,
        LoadedGenerator as LoadedGenerator,
        Model as Model,
        ModelDescription as ModelDescription,
        ModelFileError as ModelFileError,
        OfflineError as OfflineError,
        PsSnapshot as PsSnapshot,
        Run as Run,
        RunHandle as RunHandle,
        RunInfo as RunInfo,
        RunStatus as RunStatus,
        Sample as Sample,
        SampleResult as SampleResult,
        StopTraining as StopTraining,
        TrainResult as TrainResult,
        ValidationWarning as ValidationWarning,
        adversarial as adversarial,
        audio as audio,
        catalog as catalog,
        collect_source_files as collect_source_files,
        compact_events as compact_events,
        compare as compare,
        contributions as contributions,
        dashboard_status as dashboard_status,
        data_check as data_check,
        dataset_info as dataset_info,
        describe as describe,
        diff as diff,
        download_dataset as download_dataset,
        dumps as dumps,
        ensure_cached_url as ensure_cached_url,
        ensure_dashboard_running as ensure_dashboard_running,
        evaluate as evaluate,
        evaluations as evaluations,
        events as events,
        explain as explain,
        export as export,
        export_generator as export_generator,
        extract_ema_bundle as extract_ema_bundle,
        find_runs_root as find_runs_root,
        find_workspace as find_workspace,
        fingerprint as fingerprint,
        from_dict as from_dict,
        gc as gc,
        gc_all as gc_all,
        generate_run_slug as generate_run_slug,
        get_cache_dir as get_cache_dir,
        hndl as hndl,
        image as image,
        import_path as import_path,
        init_project as init_project,
        inspect as inspect,
        interpolate as interpolate,
        interpolate_variables as interpolate_variables,
        launch as launch,
        lift as lift,
        list_datasets as list_datasets,
        list_runs as list_runs,
        list_templates as list_templates,
        load as load,
        load_bundle as load_bundle,
        load_inference as load_inference,
        load_recipe as load_recipe,
        load_workspace as load_workspace,
        logs as logs,
        lower as lower,
        model as model,
        module as module,
        new as new,
        objective as objective,
        observations as observations,
        open_run as open_run,
        operators as operators,
        ops as ops,
        override as override,
        packaged as packaged,
        packaged_list as packaged_list,
        plain as plain,
        preflight as preflight,
        prepare as prepare,
        prepare_resume as prepare_resume,
        prior_loss as prior_loss,
        project as project,
        prune_checkpoints as prune_checkpoints,
        ps as ps,
        queue_add as queue_add,
        queue_clear as queue_clear,
        queue_install_service as queue_install_service,
        queue_ls as queue_ls,
        queue_run as queue_run,
        queue_status as queue_status,
        queue_stop as queue_stop,
        read_run_info as read_run_info,
        reconstruction as reconstruction,
        register_run as register_run,
        request_checkpoint as request_checkpoint,
        request_sample as request_sample,
        request_status as request_status,
        request_stop as request_stop,
        resolve as resolve,
        resolve_checkpoint as resolve_checkpoint,
        resolve_device_aliases as resolve_device_aliases,
        resolve_inheritance as resolve_inheritance,
        resolve_or_cache as resolve_or_cache,
        resolve_run_target as resolve_run_target,
        resume as resume,
        rm as rm,
        sample as sample,
        sampler as sampler,
        samples as samples,
        save as save,
        schema as schema,
        serve as serve,
        status as status,
        stop as stop,
        stop_all_servers as stop_all_servers,
        stop_dashboard as stop_dashboard,
        stop_projector as stop_projector,
        stop_viewer as stop_viewer,
        sync_registry as sync_registry,
        tag as tag,
        train as train,
        validate as validate,
        view as view,
        viewer_status as viewer_status,
    )

from .api import __all__ as _api_all, ALIASES as _ALIASES

__all__ = tuple(_api_all)
del _api_all


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    # The metrics module is callable as hg.metrics(run) and provides module attributes
    if name == "metrics":
        return importlib.import_module(".metrics", __name__)

    # Guard: only allow public symbols declared in __all__ (or their aliases)
    if name in __all__ or name in _ALIASES:
        api = importlib.import_module(".api", __name__)
        if hasattr(api, name):
            return getattr(api, name)

    # Allow direct submodule access (e.g. hypergan.cli, hypergan.config)
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    # Return __all__ plus standard module dunder attributes present in globals
    dunders = {k for k in globals() if k.startswith("__") and k.endswith("__")}
    return sorted(dunders | set(__all__))
