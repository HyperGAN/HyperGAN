"""Shared run policy, independent of numerical trainers and tensor state.

The execution interface is internal. Public training commands remain separate
from adapters that configure a fenced attempt before constructing workers.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Protocol
import uuid
import warnings

from .config import config_values, fingerprint, load_config, resolve_config, observation_fingerprint, resume_compatible
from .metrics import digest, metric_catalog, publish_catalog, select_metrics, select_preview_metrics, Throughput
from .previews import DEFAULT_KEEP, DEFAULT_NAME, KEEP_ALL, sample_name
from .metric_plugins import prepare_custom, ScalarMetrics
from .run_state import atomic_json, run_lock, sync_directory, validate_event_boundary
from .observation_io import ObservationIO
from .background_poll import BackgroundPoll
from .run_signals import GracefulStop

# How many published preview records the run manifest repeats. Preview
# retention holds far more than this (DEFAULT_KEEP, thinned); the manifest
# carries only a recent tail plus `preview_count` and stays small.
MANIFEST_PREVIEWS = 16


@dataclass(frozen=True)
class ExecutionInfo:
    step: int
    data_identity: dict
    recovery_reasons: list
    checkpoint_metadata: dict
    environment: dict | None = None


@dataclass(frozen=True)
class AttemptContext:
    run_id: str
    attempt_id: str
    attempt_index: int
    run_dir: Path
    attempt_dir: Path


class FatalExecutionError(RuntimeError):
    """The execution group is unusable; optional observation must not swallow it."""


class ObserverError(RuntimeError):
    """Optional delivery failed while the execution adapter remains usable."""


@dataclass(frozen=True)
class Restored:
    checkpoint_path: Path
    step: int
    # Qualified-but-accepted resume conditions, already warned where detected.
    warnings: tuple = ()


@dataclass(frozen=True)
class CompletedUpdate:
    step: int
    metrics: dict


@dataclass(frozen=True)
class PreviewResult:
    record: dict
    index: dict
    errors: list


@dataclass(frozen=True)
class ArtifactResult:
    bundle_path: Path
    sample_path: Path


class Execution(Protocol):
    def environment(self) -> dict: ...
    def start(self) -> ExecutionInfo: ...
    def restore(self, run_dir, checkpoint, run_id, config_sha256) -> Restored: ...
    def update(self) -> CompletedUpdate: ...
    def checkpoint(self, run_dir, metadata) -> Path: ...
    def preview(self, run_dir, identity, *, keep) -> PreviewResult | None: ...
    @property
    def inference_available(self) -> bool: ...
    def inference(self, bundle_dir, identity) -> ArtifactResult: ...
    def observe(self, callback, event) -> None: ...
    def shutdown(self) -> None: ...


PREVIEW_KEEP_EXPLICIT = 'explicit'
PREVIEW_KEEP_DEFAULT = 'default'


def resolve_preview_keep(manifest, preview_keep=None, preview_keep_source=None):
    """Resolve an attempt's preview bound and record how it was chosen.

    A bound the caller passed is explicit. Otherwise a bound an earlier attempt
    stored is inherited only when that attempt marked it explicit: a manifest
    written before this marker existed recorded whatever the default was at the
    time, so it resumes into the current `DEFAULT_KEEP` instead of pinning the
    run to a stale default forever. `hypergan train` on an existing run
    directory and `hypergan resume` both go through this one helper.
    """
    if preview_keep is not None:
        source = (preview_keep_source if preview_keep_source == PREVIEW_KEEP_DEFAULT
                  else PREVIEW_KEEP_EXPLICIT)
        return preview_keep, source
    stored = manifest.get('preview_keep')
    if type(stored) is int and manifest.get('preview_keep_source') == PREVIEW_KEEP_EXPLICIT:
        return stored, PREVIEW_KEEP_EXPLICIT
    return DEFAULT_KEEP, PREVIEW_KEEP_DEFAULT


def _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every=0,
              preview_keep=DEFAULT_KEEP, preview_name=DEFAULT_NAME):
    if type(preview_every) is not int or preview_every < 0:
        raise ValueError("preview_every must be a nonnegative integer; zero disables previews")
    if type(preview_keep) is not int or preview_keep < KEEP_ALL:
        raise ValueError("preview_keep must be a positive integer, "
                         f"or {KEEP_ALL} to keep every preview")
    sample_name(preview_name)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    if max_seconds is not None and (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be a finite positive number')
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError('stop_after_steps must be a positive integer')


def _candidate_attempt(run_dir, run_id):
    """Choose identity without writes; existing runs require the caller's lock."""
    root = run_dir / 'attempts'
    indexes = ([int(p.name.split('-')[0]) for p in root.iterdir()
                if p.is_dir() and p.name.split('-')[0].isdigit()] if root.exists() else [])
    index = max(indexes, default=0) + 1
    identity = f'{index:04d}-{uuid.uuid4().hex}'
    return AttemptContext(run_id, identity, index, run_dir, root / identity)


def _persist_attempt(context):
    root = context.run_dir / 'attempts'
    root.mkdir(exist_ok=True)
    context.attempt_dir.mkdir()
    sync_directory(root)


def _json_value(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Execution descriptor must contain finite JSON values') from exc


def _preserve_initial_source(run_dir, manifest):
    if 'initial_source' in manifest:
        return
    # Older runs recorded source on each immutable attempt. Check the first
    # historical attempt; missing/invalid history is explicitly a fallback, never
    # evidence that the most recently recorded release started the run.
    root = run_dir / 'attempts'
    candidates = (path for path in root.iterdir()
                  if path.is_dir() and not path.is_symlink()
                  and path.name.split('-')[0].isdigit()) if root.is_dir() else ()
    first = min(candidates, key=lambda path: (int(path.name.split('-')[0]), path.name), default=None)
    if first is not None:
        path = first / 'manifest.json'
        try:
            if path.is_symlink():
                raise ValueError('Attempt manifest must be an ordinary file')
            with path.open('rb') as stream:
                payload = stream.read(16 * 1024 * 1024 + 1)
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError('Attempt manifest exceeds its byte bound')
            original = json.loads(payload)
            if (isinstance(original, dict) and original.get('run_id') == manifest['run_id']
                    and isinstance(original.get('source'), dict)):
                manifest['initial_source'] = _json_value(original['source'])
                manifest['initial_source_origin'] = path.relative_to(run_dir).as_posix()
                return
        except (OSError, ValueError, RecursionError):
            pass
    manifest['initial_source'] = _json_value(manifest.get('source', {}))
    manifest['initial_source_origin'] = 'previous-run-manifest; original attempt unavailable'


def _configure_attempt(execution, context, preview_every, on_event):
    """Optional lightweight hook: no workers, model construction or writes here."""
    hook = getattr(execution, 'configure_attempt', None)
    if hook is None:
        return None
    value = hook(context, preview_every=preview_every, on_event=on_event)
    if (not isinstance(value, dict) or set(value) != {'execution', 'service_policy'}
            or not isinstance(value['execution'], dict) or not isinstance(value['service_policy'], dict)):
        raise ValueError('Attempt configuration requires execution and service_policy dictionaries')
    return _json_value(value)


def _apply_execution(manifest, descriptor):
    if descriptor is not None:
        manifest.update(descriptor)
        if descriptor['execution'].get('name') != 'cpu-single':
            manifest['qualification']['status'] = 'unqualified'
            # The native stream summary cannot describe rank-specific streams.
            manifest.pop('rng_streams', None)


def _cleanup_validation_failure(execution):
    try:
        execution.shutdown()
    except BaseException:
        pass  # Preserve the validation/restore error; no attempt is published.
    finally:
        cleanup = getattr(execution, 'close_observers', None)
        if cleanup is not None:
            try:
                cleanup()
            except BaseException:
                pass


def run_train(config_path, run_dir, steps=None, *, checkpoint_every=100, max_seconds=None,
          stop_after_steps=None, on_event=None, preview_every=0, preview_keep=None,
          preview_keep_source=None, preview_name=DEFAULT_NAME, execution_factory=None, tune=False):
    """Create a run; budgets stop only at complete D/G/EMA update boundaries."""
    if execution_factory is None:
        raise TypeError('run_train requires an execution_factory')
    if type(tune) is not bool:
        raise ValueError('tune must be a boolean')
    preview_keep, preview_keep_source = resolve_preview_keep({}, preview_keep, preview_keep_source)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    config = load_config(config_path)
    if steps is not None:
        raw = config_values(config)
        raw['training']['steps'] = steps
        config = resolve_config(raw)
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f'Run directory already exists: {run_dir}')
    prepare_custom(config)
    qualification = dict(config['qualification'])
    qualification['status'] = 'reference-only' if qualification['recipe_match'] else 'unqualified'
    qualification['runtime_qualification'] = 'not-certified; run numerical parity CI for this exact runtime'
    run_id = uuid.uuid4().hex
    context = _candidate_attempt(run_dir, run_id)
    execution = execution_factory(config)
    managed = False
    try:
        descriptor = _configure_attempt(execution, context, preview_every, on_event)
        if tune and not callable(getattr(execution, 'tune', None)):
            raise ValueError('Startup tuning is unsupported by this execution adapter')
        environment = execution.environment()
        run_dir.mkdir(parents=True, exist_ok=False)
        sync_directory(run_dir.parent)
        manifest = {'schema_version': 1, 'status': 'initializing', 'run_id': run_id,
                'run_dir': str(run_dir), 'config': config_values(config), 'config_sha256': fingerprint(config),
                'runtime': environment['runtime'], 'source': environment['source'], 'qualification': qualification,
                'warnings': list(config['warnings']), 'steps': 0, 'total_steps': config['training']['steps'],
                'global_batch_size': config['training']['batch_size'], 'resume_supported': False,
                'last_durable_step': None, 'checkpoint_path': None, 'next_sample_sequence': 1,
                'preview_every': preview_every, 'preview_keep': preview_keep,
                'preview_keep_source': preview_keep_source,
                'preview_name': sample_name(preview_name), 'previews': [], 'observation_errors': [],
                'rng_streams': {name: config['training']['seed'] + offset for name, offset in [
                    ('data', config['training']['data_seed_offset']),
                    ('prior', config['training']['prior_seed_offset']), ('penalty', 3)]}}
        if tune:
            manifest['initialization_tuning'] = {'status': 'pending'}
        manifest['rng_streams']['sampling'] = config['sampling']['seed']
        _apply_execution(manifest, descriptor)
        with run_lock(run_dir):
            managed = True
            return execute_run(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps, on_event,
                               execution=execution, context=context)
    except BaseException:
        if not managed:
            _cleanup_validation_failure(execution)
        raise


def run_resume(run_dir, checkpoint=None, config_path=None, *, checkpoint_every=None,
               max_seconds=None, stop_after_steps=None, on_event=None, preview_every=None,
               preview_keep=None, preview_keep_source=None, preview_name=None,
               execution_factory=None, steps=None, require_same_config=False):
    """Bind an in-memory candidate, then restore before publishing that attempt."""
    if execution_factory is None:
        raise TypeError('run_resume requires an execution_factory')
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f'Run directory does not exist: {run_dir}')
    with run_lock(run_dir):
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        required = {'schema_version', 'run_id', 'config', 'config_sha256', 'next_sample_sequence'}
        if not isinstance(manifest, dict) or not required.issubset(manifest) or manifest['schema_version'] != 1:
            raise ValueError('Run manifest has no supported full recovery contract')
        if 'execution' in manifest and not isinstance(manifest['execution'], dict):
            raise ValueError('Run manifest execution identity must be a dictionary')
        checkpoint_every = manifest.get('checkpoint_every', 100) if checkpoint_every is None else checkpoint_every
        preview_every = manifest.get('preview_every', 0) if preview_every is None else preview_every
        preview_keep, preview_keep_source = resolve_preview_keep(manifest, preview_keep, preview_keep_source)
        preview_name = manifest.get('preview_name') if preview_name is None else preview_name
        preview_name = sample_name(preview_name)
        _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
        config = load_config(config_path) if config_path is not None else resolve_config(manifest['config'])
        if steps is not None:
            if config_path is None:
                raise ValueError('A training step override requires an explicit configuration')
            raw = config_values(config)
            raw['training']['steps'] = steps
            config = resolve_config(raw)
        if require_same_config:
            original = resolve_config(manifest['config'])
            if not resume_compatible(config, original, include_observation=True):
                raise ValueError('Training configuration differs from the original run; use a new run directory for a different configuration')
        if fingerprint(config) != manifest['config_sha256']:
            if (not resume_compatible(config, manifest['config'])
                    or fingerprint(manifest['config']) != manifest['config_sha256']):
                raise ValueError('Resume configuration differs from the original run; only an increased '
                                 'training.steps with unchanged constant learning rate (lr_floor=1) is allowed')
        prepare_custom(config)
        context = _candidate_attempt(run_dir, manifest['run_id'])
        execution = execution_factory(config)
        try:
            descriptor = _configure_attempt(execution, context, preview_every, on_event)
            requested = descriptor['execution'] if descriptor is not None else None
            if json.dumps(requested, sort_keys=True, allow_nan=False) != json.dumps(manifest.get('execution'), sort_keys=True, allow_nan=False):
                raise ValueError('Resume numerical execution identity differs from the run manifest')
            restored = execution.restore(run_dir, checkpoint, manifest['run_id'], manifest['config_sha256'])
            _apply_execution(manifest, descriptor)
            # A resume condition the adapter qualified rather than rejected. The
            # adapter already warned; keep it durable on the run as well.
            recorded = manifest.setdefault('warnings', [])
            manifest['resume_warnings'] = list(restored.warnings)
            for warning in restored.warnings:
                if warning not in recorded:
                    recorded.append(warning)
            checkpoint_info = json.loads((Path(restored.checkpoint_path) / 'manifest.json').read_text())
            if checkpoint_info.get('event_boundary') is None:
                raise ValueError('Controller checkpoint requires a durable event boundary')
            validate_event_boundary(run_dir, checkpoint_info)
            manifest['recovery_parent'] = {
                'attempt_id': checkpoint_info['attempt_id'], 'step': restored.step,
                'checkpoint_id': Path(restored.checkpoint_path).name,
                'checkpoint_sha256': digest(checkpoint_info),
            }
            manifest['config'] = config_values(config)
            manifest['config_sha256'] = fingerprint(config)
            manifest['total_steps'] = config['training']['steps']
            _preserve_initial_source(run_dir, manifest)
            manifest.update(preview_every=preview_every, preview_keep=preview_keep,
                            preview_keep_source=preview_keep_source, preview_name=preview_name,
                            durable_event_boundary=checkpoint_info.get('event_boundary'),
                            checkpoint_path=str(restored.checkpoint_path), last_durable_step=restored.step,
                            resumed_from=str(restored.checkpoint_path), steps=restored.step)
        except BaseException:
            _cleanup_validation_failure(execution)
            raise
        return execute_run(config, run_dir, manifest, checkpoint_every, max_seconds,
                           stop_after_steps, on_event, execution=execution, context=context)


def execute_run(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps,
                on_event, *, execution, context=None):
    """Hold one adapter for an attempt; cleanup also covers early persistence errors."""
    shutdown_attempted = False
    primary_error = None
    def shutdown():
        nonlocal shutdown_attempted
        if not shutdown_attempted:
            shutdown_attempted = True
            execution.shutdown()
    try:
        with GracefulStop() as stop:
            observe_stop = getattr(execution, 'set_observation_stop', None)
            if observe_stop is not None:
                observe_stop(lambda: bool(stop.reason))
            return _execute_run(config, run_dir, manifest, checkpoint_every, max_seconds,
                            stop_after_steps, on_event, execution=execution, shutdown=shutdown,
                            context=context, stop=stop)
    except BaseException as error:
        primary_error = error
        try:
            shutdown()
        except BaseException:
            pass
        raise
    finally:
        cleanup = getattr(execution, 'close_observers', None)
        if cleanup is not None:
            try:
                cleanup()
            except BaseException as error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note(f'Progress worker cleanup also failed: {error}')


def _execute_run(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps,
             on_event, *, execution, shutdown, context=None, stop=None):
    started = time.monotonic()
    context = context or _candidate_attempt(run_dir, manifest['run_id'])
    from .provenance import hypergan_source
    manifest['source'] = dict(hypergan_source(), runtime_checked=False)
    _persist_attempt(context)
    index, attempt_id, attempt_dir = context.attempt_index, context.attempt_id, context.attempt_dir
    custom_metrics = ScalarMetrics(config)
    catalog = metric_catalog(config)
    catalog_revision = publish_catalog(run_dir, config)
    throughput = Throughput()
    global_batch_size = config['training']['batch_size']
    # Cumulative training time continues across attempts; only the wall clock
    # inside an attempt is added, so idle time between attempts never counts.
    training_baseline = manifest.get('training_seconds', 0.0)
    if type(training_baseline) not in (int, float) or not math.isfinite(training_baseline) or training_baseline < 0:
        raise ValueError('Run manifest training_seconds must be a finite, nonnegative number of seconds')
    training_baseline = float(training_baseline)

    def training_seconds(now=None):
        return training_baseline + max(0.0, (time.monotonic() if now is None else now) - started)

    manifest.update(metrics_catalog=catalog_revision, observation_sha256=observation_fingerprint(config))
    manifest.update(attempt_id=attempt_id, attempt_index=index, attempt_dir=str(attempt_dir),
                    status='initializing', checkpoint_every=checkpoint_every, stop_reason=None,
                    possible_lost_steps=0, training_seconds=training_baseline,
                    global_batch_size=global_batch_size,
                    samples_seen=manifest['steps'] * global_batch_size)
    for key in ('error', 'shutdown_error', 'metric_shutdown_error', 'progress_observation',
                'sample_path', 'bundle_path', 'steps_per_second'):
        manifest.pop(key, None)
    atomic_json(run_dir / 'manifest.json', manifest)
    journal = ObservationIO(run_dir, attempt_dir)
    sequence = 0
    dropped = None
    last_published = 0.0
    last_event_step = None
    custom_publications = 0
    checkpoint_custom_publications = 0
    preview_publications = 0
    checkpoint_preview_publications = 0

    def emit(event, *, _observe=True, _step=None, **values):
        nonlocal sequence, dropped, last_event_step
        row = dict(values, schema_version=2, event=event, run_id=manifest['run_id'],
                   stream_id='training', stream_generation=manifest['run_id'], catalog=catalog_revision,
                   attempt_id=attempt_id, sequence=sequence + 1,
                   step=manifest['steps'] if _step is None else _step, seconds=time.monotonic() - started)
        if dropped is not None:
            row['observation_gap'] = dict(dropped)
        boundary_event = event in {'start', 'resume', 'checkpoint', 'checkpoint_request',
                                   'checkpoint_boundary', 'observation_gap', 'complete',
                                   'stopped', 'failed', 'interrupted', 'tuning'}
        if not journal.append(row, wait=boundary_event):
            if dropped is None:
                dropped = {'dropped_train_events': 0, 'by_event': {},
                           'first_step': row['step'], 'last_step': row['step']}
            dropped['by_event'][event] = dropped['by_event'].get(event, 0) + 1
            dropped['first_step'] = min(dropped['first_step'], row['step'])
            dropped['last_step'] = max(dropped['last_step'], row['step'])
            totals = manifest.setdefault('dropped_observation_events', {})
            totals[event] = totals.get(event, 0) + 1
            if event == 'train':
                dropped['dropped_train_events'] += 1
                manifest['dropped_train_events'] = manifest.get('dropped_train_events', 0) + 1
            return None
        dropped = None
        sequence += 1
        last_event_step = row['step']
        if on_event is not None and _observe:
            def notify(value):
                try:
                    on_event(value)
                except FatalExecutionError:
                    raise
                except Exception as exc:
                    try:
                        warnings.warn(f'Run event observer failed: {exc}', RuntimeWarning)
                    except Warning:
                        pass
            try:
                from .bounded_cli_output import CLIProgress
                execution.observe(on_event if type(on_event) is CLIProgress else notify, dict(row))
            except FatalExecutionError:
                raise
            except ObserverError as exc:
                # A supervised adapter delivers its configured callback outside
                # this process. Its failure must remain observable without
                # recursively invoking that same failed observer.
                source_step = getattr(exc, 'observation_step', manifest['steps'])
                record = {'source': 'progress', 'step': source_step,
                          'attempt_id': attempt_id,
                          'error': f'{type(exc).__name__}: {exc}'[:1000]}
                manifest['observation_errors'] = [*manifest.get('observation_errors', []), record][-16:]
                publish(wait=False)
                emit('observer_error', _observe=False, _step=source_step,
                     source='progress', error=record['error'])
            except Exception as exc:
                if type(on_event) is not CLIProgress:
                    raise
                try:
                    warnings.warn(f'Run event observer failed: {exc}', RuntimeWarning)
                except Warning:
                    pass
            finally:
                if event in ('complete', 'stopped', 'failed', 'interrupted'):
                    status = getattr(execution, 'observation_status', lambda: None)()
                    if status is not None:
                        manifest['progress_observation'] = status
                        if stop is not None and stop.reason:
                            manifest['stop_reason'] = stop.reason
                        # Terminal callback results must outlive the API return;
                        # the live-status throttle may suppress this final update.
                        publish()
                        emit('observer_status', _observe=False, source='progress', delivery=status)
        return row

    def publish(*, wait=True):
        nonlocal last_published
        now = time.monotonic()
        if not wait and now - last_published < 0.25:
            journal.check()
            return
        manifest['seconds'] = now - started
        manifest['training_seconds'] = training_seconds(now)
        journal.publish(manifest, wait=wait)
        last_published = now

    request_reader = None
    evaluations = None

    def publish_custom(outcomes):
        nonlocal custom_publications
        for outcome in outcomes:
            custom_publications += 1
            emit('metric', _step=outcome['context']['step'],
                 metrics=outcome['metrics'], measurement_status=outcome['measurement_status'],
                 metric_publication='sampled' if outcome['metrics'] else 'failed')

    def collect_custom(*, final=False):
        try:
            outcomes = custom_metrics.close(drain=True,
                stop_requested=lambda: stop is not None and bool(stop.reason)) if final else custom_metrics.poll()
        except Exception as error:
            publish_custom(getattr(error, 'metric_outcomes', []))
            raise
        publish_custom(outcomes)

    lifecycle_started = False

    def begin_lifecycle():
        nonlocal lifecycle_started
        if lifecycle_started:
            return
        # Lifecycle must be sequence one, including before startup tuning. The
        # viewer uses this first event to establish the attempt's lineage.
        publish()  # A start/resume observer can immediately submit an attempt-bound request.
        parent = manifest.get('recovery_parent')
        emit('resume' if parent else 'start', config_sha256=fingerprint(config),
             observation_sha256=manifest['observation_sha256'],
             warnings=list(manifest.get('resume_warnings', [])),
             parent_attempt_id=parent['attempt_id'] if parent else None,
             restored_step=parent['step'] if parent else 0,
             checkpoint_id=parent['checkpoint_id'] if parent else None,
             checkpoint_sha256=parent['checkpoint_sha256'] if parent else None)
        lifecycle_started = True

    try:
        info = execution.start()
        if info.environment is not None:
            environment = _json_value(info.environment)
            if (not isinstance(environment, dict) or set(environment) != {'runtime', 'source'}
                    or any(not isinstance(value, dict) for value in environment.values())):
                raise ValueError('Execution environment requires runtime and source dictionaries')
            manifest.update(environment)
        manifest.setdefault('initial_source', _json_value(manifest.get('source', {})))
        manifest.setdefault('initial_source_origin', 'run-start')
        from .interval_evaluation import IntervalEvaluations
        def evaluation_event(event, **values):
            nonlocal custom_publications
            custom_publications += 1
            return emit(event, **values)
        manifest['steps'] = info.step
        evaluations = IntervalEvaluations(config, run_dir, manifest, execution, evaluation_event)
        evaluations.start()
        reasons = info.recovery_reasons
        manifest.update(resume_supported=not reasons, resume_unsupported_reasons=reasons,
                        data_identity=info.data_identity, status='running')
        tuning = manifest.get('initialization_tuning')
        if tuning is not None and tuning.get('status') == 'pending':
            if manifest.get('recovery_parent') or info.step != 0:
                raise ValueError('Startup tuning may only run before the first training update')
            if reasons:
                raise ValueError('Startup tuning requires full checkpoint support: ' + '; '.join(reasons))
            manifest['status'] = 'tuning'
            tuning.update(status='running', message='Measuring generator initialization')
            begin_lifecycle()
            publish()
            emit('tuning', tuning=dict(tuning))
            def tuning_progress(value):
                progress = _json_value(value)
                if not isinstance(progress, dict):
                    raise ValueError('Tuning progress must be a JSON object')
                tuning.update(progress, status='running')
                publish()
                emit('tuning', tuning=dict(tuning))
            try:
                result = _json_value(execution.tune(run_dir, on_event=tuning_progress))
                if not isinstance(result, dict):
                    raise ValueError('Tuning result must be a JSON object')
            except BaseException as error:
                tuning.update(status='failed', message=f'{type(error).__name__}: {error}'[:1000])
                publish()
                emit('tuning', tuning=dict(tuning))
                raise
            tuning.update(result, status='complete')
            manifest['status'] = 'running'
            publish()
            emit('tuning', tuning=dict(tuning))
        manifest['qualification']['resume'] = False
        manifest['qualification']['recovery_scope'] = 'Full-state protocol on the recorded execution device; custom hidden state is author responsibility'
        for reason in reasons:
            if reason not in manifest['warnings']:
                manifest['warnings'].append(reason)
            warnings.warn(reason, RuntimeWarning)
        metadata = dict(info.checkpoint_metadata, run_id=manifest['run_id'], attempt_id=attempt_id,
                        next_sample_sequence=manifest['next_sample_sequence'],
                        source=_json_value(manifest.get('source', {})),
                        initial_source=_json_value(manifest['initial_source']))
        if 'initialization_tuning' in manifest:
            metadata['initialization_tuning'] = _json_value(manifest['initialization_tuning'])

        def checkpoint_now(request_ids=None, observer=False):
            nonlocal checkpoint_custom_publications, checkpoint_preview_publications
            if not manifest['resume_supported']:
                return
            metadata['next_sample_sequence'] = manifest['next_sample_sequence']
            metadata['request_ids'] = list(request_ids or [])
            try:
                if dropped is not None or last_event_step != manifest['steps']:
                    emit('checkpoint_boundary', _observe=False)
                metadata['event_boundary'] = journal.commit_boundary()
                path = execution.checkpoint(run_dir, metadata)
            except FatalExecutionError:
                raise
            except Exception as exc:
                if observer:
                    return None, exc
                raise
            checkpoint_custom_publications = custom_publications
            checkpoint_preview_publications = preview_publications
            manifest.update(checkpoint_path=str(path), last_durable_step=manifest['steps'],
                            durable_event_boundary=metadata['event_boundary'],
                            possible_lost_steps=0)
            publish()
            emit('checkpoint', checkpoint_path=str(path), request_ids=list(request_ids or []))
            return path, None

        def observer_error(source, error, *, step=None):
            source_step = manifest['steps'] if step is None else step
            record = {'source': source, 'step': source_step, 'attempt_id': attempt_id,
                      'error': f'{type(error).__name__}: {error}'[:1000]}
            manifest['observation_errors'] = [*manifest.get('observation_errors', []), record][-16:]
            publish(wait=False)
            emit('observer_error', _step=source_step,
                 **{key: value for key, value in record.items() if key not in ('step', 'attempt_id')})

        def publish_preview_result(preview):
            nonlocal preview_publications
            if preview is None:
                return
            record, preview_index, errors = preview.record, preview.index, preview.errors
            # The manifest is rewritten and fsynced on every publication, so it
            # carries a recent tail; previews/index.json is the whole history.
            manifest['previews'] = preview_index['previews'][-MANIFEST_PREVIEWS:]
            manifest['preview_count'] = len(preview_index['previews'])
            manifest['preview_path'] = record['path']
            preview_publications += 1
            publish(wait=False)
            emit('preview', _step=record['step'], preview=record)
            metrics, statuses = select_preview_metrics(catalog, record)
            if metrics or statuses:
                emit('metric', _step=record['step'], metrics=metrics, measurement_status=statuses,
                     source='preview', preview_identity=record['identity'])
            for error in errors:
                observer_error('preview_retention', RuntimeError(error), step=record['step'])

        def collect_preview(*, final=False):
            hook = getattr(execution, 'close_previews' if final else 'poll_preview', None)
            if hook is None:
                return
            try:
                if final and hasattr(execution, 'preview_busy'):
                    while execution.preview_busy:
                        publish_preview_result(execution.poll_preview())
                        if stop is not None and stop.reason:
                            if execution.abort_previews():
                                manifest['cancelled_previews'] = manifest.get('cancelled_previews', 0) + 1
                                emit('preview_cancelled', reason=stop.reason)
                            return
                        if execution.preview_busy:
                            # Only terminal draining waits. Poll so a signal
                            # received during that wait can cancel immediately.
                            time.sleep(.01)
                    return
                publish_preview_result(hook())
            except FatalExecutionError:
                raise
            except Exception as exc:
                observer_error('preview', exc, step=getattr(exc, 'preview_context', {}).get('step'))

        def preview_now():
            if getattr(execution, 'preview_busy', False):
                manifest['skipped_previews_busy'] = manifest.get('skipped_previews_busy', 0) + 1
                emit('preview_skipped', reason='worker_busy')
                return
            identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                        'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence'],
                        'name': sample_name(manifest.get('preview_name'))}
            manifest['next_sample_sequence'] += 1
            publish()  # Reserve before rendering: failed or killed attempts never reuse a sequence.
            try:
                publish_preview_result(execution.preview(run_dir, identity, keep=manifest['preview_keep']))
            except FatalExecutionError:
                raise
            except Exception as exc:
                observer_error('preview', exc)
                return

        from .run_requests import pending_requests, acknowledge_request
        request_reader = BackgroundPoll(lambda: pending_requests(run_dir))

        def poll_requests(*, force=False):
            try:
                if force:
                    pending = request_reader.read_now()
                else:
                    ready, pending = request_reader.poll()
                    if not ready:
                        return
            except RuntimeError:
                return  # A producer holds the short queue lock; retry next boundary.
            except Exception as exc:
                observer_error('checkpoint_requests', exc)
                return
            matching = []
            def acknowledge(request, status, path=None, error=None, saved_step=None):
                try:
                    receipt = acknowledge_request(run_dir, request['request_id'], status=status,
                                                  attempt_id=attempt_id, checkpoint_path=str(path) if path else None,
                                                  step=(manifest['steps'] if saved_step is None else saved_step) if path else None, error=error)
                except Exception as exc:
                    observer_error('checkpoint_acknowledgement', exc)
                    return
                emit('checkpoint_request', receipt=receipt)
            for request in pending:
                if request['run_id'] != manifest['run_id'] or request['attempt_id'] != attempt_id:
                    acknowledge(request, 'rejected', error='Request targets a different run or attempt; it cannot be applied after resume')
                elif not manifest['resume_supported']:
                    acknowledge(request, 'rejected', error='Full training checkpoints are unsupported for this configuration')
                else:
                    matching.append(request)
            if matching and manifest.get('checkpoint_path'):
                # A lost acknowledgement must not repeat a still-identifiable save.
                try:
                    saved_path = Path(manifest['checkpoint_path'])
                    saved = json.loads((saved_path / 'manifest.json').read_text(encoding='utf-8'))
                    completed_ids = set(saved.get('request_ids', [])) if saved.get('attempt_id') == attempt_id else set()
                except Exception as exc:
                    observer_error('checkpoint_request_reconciliation', exc)
                    return
                remaining = []
                for request in matching:
                    if request['request_id'] in completed_ids:
                        acknowledge(request, 'succeeded', path=saved_path, saved_step=saved['step'])
                    else:
                        remaining.append(request)
                matching = remaining
            if matching:
                path, error = checkpoint_now([request['request_id'] for request in matching], observer=True)
                for request in matching:
                    acknowledge(request, 'succeeded' if path else 'rejected', path=path,
                                error=f'{type(error).__name__}: {error}'[:1000] if error else None)
                if error:
                    observer_error('manual_checkpoint', error)

        begin_lifecycle()
        if manifest['last_durable_step'] is None or manifest.get('resumed_from'):
            # Accepting an older recovery point must also move the default pointer,
            # even if this attempt stops before another update.
            checkpoint_now()
        publish()
        poll_requests(force=True)
        if (manifest['steps'] == 0 and not manifest.get('recovery_parent')
                and manifest['preview_every'] and not (stop is not None and stop.reason)):
            # Capture the actual initialization (including startup calibration)
            # before any update. Rendering still uses the normal async worker.
            preview_now()
        custom_metrics.start()
        attempt_steps = 0
        while manifest['steps'] < config['training']['steps']:
            if stop is not None and stop.reason:
                manifest['stop_reason'] = stop.reason
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                manifest['stop_reason'] = 'max_seconds'
                break
            if stop_after_steps is not None and attempt_steps >= stop_after_steps:
                manifest['stop_reason'] = 'stop_after_steps'
                break
            collect_custom()
            collect_preview()
            evaluations.poll()
            update_started = time.monotonic()
            completed = execution.update()
            step_seconds = time.monotonic() - update_started
            row = completed.metrics
            attempt_steps += 1
            manifest['steps'] = completed.step
            durable = manifest['last_durable_step']
            manifest['possible_lost_steps'] = manifest['steps'] - durable if durable is not None else manifest['steps']
            # One completed update consumes exactly one global batch; accumulation
            # and world size split that batch without changing how many real
            # examples the step drew.
            batch = row['global_batch_size'] if type(row.get('global_batch_size')) is int and row['global_batch_size'] > 0 else global_batch_size
            progress = {'samples_seen': completed.step * batch, 'training_seconds': training_seconds()}
            rate = throughput.observe(step_seconds)
            if rate is not None:
                progress['steps_per_second'] = rate
                manifest['steps_per_second'] = rate
            manifest.update(samples_seen=progress['samples_seen'], training_seconds=progress['training_seconds'])
            metrics, statuses, publication = select_metrics(config, catalog, row, completed.step, step_seconds, progress)
            custom_values, custom_statuses = custom_metrics.evaluate(dict(row, step=completed.step, step_seconds=step_seconds),
                {'run_id': manifest['run_id'], 'attempt_id': attempt_id, 'step': completed.step})
            metrics.update(custom_values)
            statuses.update(custom_statuses)
            if custom_values:
                publication = 'sampled'
            emit('train', metrics=metrics, measurement_status=statuses, metric_publication=publication,
                 **progress,
                 **{key: row[key] for key in ('global_batch_size', 'local_batch_size', 'world_size',
                                             'accumulation_steps', 'microbatch_size') if key in row})
            periodic_checkpoint = manifest['steps'] % checkpoint_every == 0
            if periodic_checkpoint:
                checkpoint_now()
            poll_requests(force=periodic_checkpoint)
            if not (stop is not None and stop.reason):
                evaluations.schedule()
            if not (stop is not None and stop.reason) and manifest['preview_every'] and manifest['steps'] % manifest['preview_every'] == 0:
                preview_now()
            publish(wait=False)
        evaluations.close(stop_requested=lambda: stop is not None and bool(stop.reason))
        collect_custom(final=True)
        collect_preview(final=True)
        poll_requests(force=True)
        # Reconcile one transient lost acknowledgement before ending the attempt.
        # Saved request IDs identify the existing checkpoint and avoid a second save.
        poll_requests(force=True)
        if (manifest['last_durable_step'] != manifest['steps']
                or custom_publications != checkpoint_custom_publications
                or preview_publications != checkpoint_preview_publications):
            checkpoint_now()
        if stop is not None and stop.reason:
            manifest['stop_reason'] = stop.reason
        if execution.inference_available and not (stop is not None and stop.reason):
            bundle_dir = attempt_dir / 'inference'
            bundle_dir.mkdir()
            sync_directory(attempt_dir)
            identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                                         'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence'],
                                         'name': sample_name(manifest.get('preview_name'))}
            manifest['next_sample_sequence'] += 1
            publish()
            artifacts = execution.inference(bundle_dir, identity)
            manifest.update(bundle_path=str(artifacts.bundle_path), sample_path=str(artifacts.sample_path))
        shutdown()
        manifest['status'] = 'complete' if manifest['steps'] == config['training']['steps'] else 'stopped'
        publish()
        emit(manifest['status'], stop_reason=manifest['stop_reason'], checkpoint_path=manifest['checkpoint_path'])
        request_reader.close(wait=True)
        journal.close()
        return manifest
    except BaseException as exc:
        # Execution may contain a half update: NEVER checkpoint in this handler.
        if request_reader is not None:
            request_reader.close()
        if stop is not None and stop.reason:
            manifest['stop_reason'] = stop.reason
        try:
            if evaluations is not None:
                evaluations.close(cancel=True)
        except BaseException as cleanup_error:
            manifest['evaluation_shutdown_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'[:1000]
        try:
            custom_metrics.close(drain=False)
        except BaseException as cleanup_error:
            manifest['metric_shutdown_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'[:1000]
        try:
            shutdown()
        except BaseException as cleanup_error:
            manifest['shutdown_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'[:1000]
        manifest.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        manifest.setdefault('initial_source', _json_value(manifest.get('source', {})))
        manifest.setdefault('initial_source_origin', 'run-start')
        try:
            if not journal.failed:
                publish()
                emit(manifest['status'], error=manifest['error'], checkpoint_path=manifest['checkpoint_path'])
            else:
                # The writer has stopped on failure; preserve failure visibility
                # without asking that failed worker to publish again.
                atomic_json(run_dir / 'manifest.json', manifest)
                atomic_json(attempt_dir / 'manifest.json', manifest)
        finally:
            journal.close()
        raise
