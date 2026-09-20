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

from .config import config_values, fingerprint, load_config, resolve_config, observation_fingerprint
from .metrics import digest, metric_catalog, publish_catalog, select_metrics
from .metric_plugins import prepare_custom, ScalarMetrics
from .run_state import atomic_json, repair_event_tail, run_lock, sync_directory


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
    def preview(self, run_dir, identity, *, keep) -> PreviewResult: ...
    @property
    def inference_available(self) -> bool: ...
    def inference(self, bundle_dir, identity) -> ArtifactResult: ...
    def observe(self, callback, event) -> None: ...
    def shutdown(self) -> None: ...


def _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every=0, preview_keep=3):
    if type(preview_every) is not int or preview_every < 0:
        raise ValueError("preview_every must be a nonnegative integer; zero disables previews")
    if type(preview_keep) is not int or not 1 <= preview_keep <= 100:
        raise ValueError("preview_keep must be between 1 and 100")
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


def run_train(config_path, run_dir, steps=None, *, checkpoint_every=100, max_seconds=None,
          stop_after_steps=None, on_event=None, preview_every=0, preview_keep=3, execution_factory=None):
    """Create a run; budgets stop only at complete D/G/EMA update boundaries."""
    if execution_factory is None:
        raise TypeError('run_train requires an execution_factory')
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep)
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
        environment = execution.environment()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': 1, 'status': 'initializing', 'run_id': run_id,
                'run_dir': str(run_dir), 'config': config_values(config), 'config_sha256': fingerprint(config),
                'runtime': environment['runtime'], 'source': environment['source'], 'qualification': qualification,
                'warnings': list(config['warnings']), 'steps': 0, 'total_steps': config['training']['steps'],
                'global_batch_size': config['training']['batch_size'], 'resume_supported': False,
                'last_durable_step': None, 'checkpoint_path': None, 'next_sample_sequence': 1,
                'preview_every': preview_every, 'preview_keep': preview_keep, 'previews': [], 'observation_errors': [],
                'rng_streams': {name: config['training']['seed'] + offset for name, offset in [
                    ('data', config['training']['data_seed_offset']),
                    ('prior', config['training']['prior_seed_offset']), ('penalty', 3)]}}
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
               preview_keep=None, execution_factory=None):
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
        preview_keep = manifest.get('preview_keep', 3) if preview_keep is None else preview_keep
        _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep)
        config = load_config(config_path) if config_path is not None else resolve_config(manifest['config'])
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
            checkpoint_info = json.loads((Path(restored.checkpoint_path) / 'manifest.json').read_text())
            manifest['recovery_parent'] = {
                'attempt_id': checkpoint_info['attempt_id'], 'step': restored.step,
                'checkpoint_id': Path(restored.checkpoint_path).name,
                'checkpoint_sha256': digest(checkpoint_info),
            }
            manifest['config'] = config_values(config)
            manifest.update(preview_every=preview_every, preview_keep=preview_keep,
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
    def shutdown():
        nonlocal shutdown_attempted
        if not shutdown_attempted:
            shutdown_attempted = True
            execution.shutdown()
    try:
        return _execute_run(config, run_dir, manifest, checkpoint_every, max_seconds,
                            stop_after_steps, on_event, execution=execution, shutdown=shutdown,
                            context=context)
    except BaseException:
        try:
            shutdown()
        except BaseException:
            pass
        raise


def _execute_run(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps,
             on_event, *, execution, shutdown, context=None):
    started = time.monotonic()
    context = context or _candidate_attempt(run_dir, manifest['run_id'])
    _persist_attempt(context)
    index, attempt_id, attempt_dir = context.attempt_index, context.attempt_id, context.attempt_dir
    custom_metrics = ScalarMetrics(config)
    catalog = metric_catalog(config)
    catalog_revision = publish_catalog(run_dir, config)
    manifest.update(metrics_catalog=catalog_revision, observation_sha256=observation_fingerprint(config))
    manifest.update(attempt_id=attempt_id, attempt_index=index, attempt_dir=str(attempt_dir),
                    status='initializing', checkpoint_every=checkpoint_every, stop_reason=None,
                    possible_lost_steps=0)
    for key in ('error', 'shutdown_error', 'sample_path', 'bundle_path'):
        manifest.pop(key, None)
    atomic_json(run_dir / 'manifest.json', manifest)
    repair_event_tail(run_dir / 'events.jsonl')
    sequence = 0

    def emit(event, *, _observe=True, **values):
        nonlocal sequence
        sequence += 1
        row = dict(values, schema_version=2, event=event, run_id=manifest['run_id'],
                   stream_id='training', stream_generation=manifest['run_id'], catalog=catalog_revision,
                   attempt_id=attempt_id, sequence=sequence,
                   step=manifest['steps'], seconds=time.monotonic() - started)
        with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
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
                execution.observe(notify, dict(row))
            except FatalExecutionError:
                raise
            except ObserverError as exc:
                # A supervised adapter delivers its configured callback outside
                # this process. Its failure must remain observable without
                # recursively invoking that same failed observer.
                record = {'source': 'progress', 'step': manifest['steps'],
                          'attempt_id': attempt_id,
                          'error': f'{type(exc).__name__}: {exc}'[:1000]}
                manifest['observation_errors'] = [*manifest.get('observation_errors', []), record][-16:]
                publish()
                emit('observer_error', _observe=False, source='progress', error=record['error'])
        return row

    def publish():
        manifest['seconds'] = time.monotonic() - started
        atomic_json(run_dir / 'manifest.json', manifest)
        atomic_json(attempt_dir / 'manifest.json', manifest)

    try:
        info = execution.start()
        if info.environment is not None:
            environment = _json_value(info.environment)
            if (not isinstance(environment, dict) or set(environment) != {'runtime', 'source'}
                    or any(not isinstance(value, dict) for value in environment.values())):
                raise ValueError('Execution environment requires runtime and source dictionaries')
            manifest.update(environment)
        reasons = info.recovery_reasons
        manifest['steps'] = info.step
        manifest.update(resume_supported=not reasons, resume_unsupported_reasons=reasons,
                        data_identity=info.data_identity, status='running')
        manifest['qualification']['resume'] = False
        manifest['qualification']['recovery_scope'] = 'Full-state protocol on the recorded execution device; custom hidden state is author responsibility'
        for reason in reasons:
            if reason not in manifest['warnings']:
                manifest['warnings'].append(reason)
            warnings.warn(reason, RuntimeWarning)
        metadata = dict(info.checkpoint_metadata, run_id=manifest['run_id'], attempt_id=attempt_id,
                        next_sample_sequence=manifest['next_sample_sequence'])

        def checkpoint_now(request_ids=None, observer=False):
            if not manifest['resume_supported']:
                return
            metadata['next_sample_sequence'] = manifest['next_sample_sequence']
            metadata['request_ids'] = list(request_ids or [])
            try:
                path = execution.checkpoint(run_dir, metadata)
            except FatalExecutionError:
                raise
            except Exception as exc:
                if observer:
                    return None, exc
                raise
            manifest.update(checkpoint_path=str(path), last_durable_step=manifest['steps'],
                            possible_lost_steps=0)
            publish()
            emit('checkpoint', checkpoint_path=str(path), request_ids=list(request_ids or []))
            return path, None

        def observer_error(source, error):
            record = {'source': source, 'step': manifest['steps'], 'attempt_id': attempt_id,
                      'error': f'{type(error).__name__}: {error}'[:1000]}
            manifest['observation_errors'] = [*manifest.get('observation_errors', []), record][-16:]
            publish()
            emit('observer_error', **{key: value for key, value in record.items() if key not in ('step', 'attempt_id')})

        def preview_now():
            identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                        'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence']}
            manifest['next_sample_sequence'] += 1
            publish()  # Reserve before rendering: failed or killed attempts never reuse a sequence.
            try:
                preview = execution.preview(run_dir, identity, keep=manifest['preview_keep'])
                record, preview_index, errors = preview.record, preview.index, preview.errors
            except FatalExecutionError:
                raise
            except Exception as exc:
                observer_error('preview', exc)
                return
            manifest['previews'] = preview_index['previews']
            manifest['preview_path'] = record['path']
            publish()
            emit('preview', preview=record)
            for error in errors:
                observer_error('preview_retention', RuntimeError(error))

        def poll_requests():
            from .run_requests import pending_requests, acknowledge_request
            try:
                pending = pending_requests(run_dir)
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

        publish()  # A start/resume observer can immediately submit an attempt-bound request.
        parent = manifest.get('recovery_parent')
        emit('resume' if parent else 'start', config_sha256=fingerprint(config),
             observation_sha256=manifest['observation_sha256'],
             parent_attempt_id=parent['attempt_id'] if parent else None,
             restored_step=parent['step'] if parent else 0,
             checkpoint_id=parent['checkpoint_id'] if parent else None,
             checkpoint_sha256=parent['checkpoint_sha256'] if parent else None)
        if manifest['last_durable_step'] is None or manifest.get('resumed_from'):
            # Accepting an older recovery point must also move the default pointer,
            # even if this attempt stops before another update.
            checkpoint_now()
        publish()
        poll_requests()
        attempt_steps = 0
        while manifest['steps'] < config['training']['steps']:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                manifest['stop_reason'] = 'max_seconds'
                break
            if stop_after_steps is not None and attempt_steps >= stop_after_steps:
                manifest['stop_reason'] = 'stop_after_steps'
                break
            update_started = time.monotonic()
            completed = execution.update()
            step_seconds = time.monotonic() - update_started
            row = completed.metrics
            attempt_steps += 1
            manifest['steps'] = completed.step
            durable = manifest['last_durable_step']
            manifest['possible_lost_steps'] = manifest['steps'] - durable if durable is not None else manifest['steps']
            metrics, statuses, publication = select_metrics(config, catalog, row, completed.step, step_seconds)
            custom_values, custom_statuses = custom_metrics.evaluate(dict(row, step=completed.step, step_seconds=step_seconds),
                {'run_id': manifest['run_id'], 'attempt_id': attempt_id, 'step': completed.step})
            metrics.update(custom_values)
            statuses.update(custom_statuses)
            if custom_values:
                publication = 'sampled'
            emit('train', metrics=metrics, measurement_status=statuses, metric_publication=publication,
                 samples_seen=completed.step * config['training']['batch_size'],
                 **{key: row[key] for key in ('global_batch_size', 'local_batch_size', 'world_size',
                                             'accumulation_steps', 'microbatch_size') if key in row})
            if manifest['steps'] % checkpoint_every == 0:
                checkpoint_now()
            poll_requests()
            if manifest['preview_every'] and manifest['steps'] % manifest['preview_every'] == 0:
                preview_now()
            publish()
        if manifest['last_durable_step'] != manifest['steps']:
            checkpoint_now()
        if execution.inference_available:
            bundle_dir = attempt_dir / 'inference'
            bundle_dir.mkdir()
            sync_directory(attempt_dir)
            identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                                         'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence']}
            manifest['next_sample_sequence'] += 1
            publish()
            artifacts = execution.inference(bundle_dir, identity)
            manifest.update(bundle_path=str(artifacts.bundle_path), sample_path=str(artifacts.sample_path))
        shutdown()
        manifest['status'] = 'complete' if manifest['steps'] == config['training']['steps'] else 'stopped'
        publish()
        emit(manifest['status'], stop_reason=manifest['stop_reason'], checkpoint_path=manifest['checkpoint_path'])
        return manifest
    except BaseException as exc:
        # Execution may contain a half update: NEVER checkpoint in this handler.
        try:
            shutdown()
        except BaseException as cleanup_error:
            manifest['shutdown_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'[:1000]
        manifest.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        publish()
        emit(manifest['status'], error=manifest['error'], checkpoint_path=manifest['checkpoint_path'])
        raise
