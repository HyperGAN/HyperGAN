"""Internal torch-free parent adapter for fixed-topology CPU worker execution.

Use from a Python main guard. These wrappers are not public train/resume CLI
profiles. The shared controller owns the run lock and every visible lifecycle
record. Workers prepare snapshots; only this parent publishes checkpoints.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .bounded_observer import BoundedObserver
from .config import config_values, fingerprint
from .cpu_worker_service import CPUWorkerService
from .distributed_commit import CheckpointCommitAuthority
from .execution_preflight import _resolve_profile
from .execution_profiles import load_execution_profile
from .run_controller import ArtifactResult, CompletedUpdate, ExecutionInfo, FatalExecutionError, PreviewResult, Restored

MAX_SAMPLE_COUNT = 1024


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _create_worker(*args):
    from .replicated_worker import create_worker
    return create_worker(*args)


def _handle_command(*args):
    from .replicated_worker import handle_command
    return handle_command(*args)


def _profile(value, config):
    resolved = load_execution_profile(value, config) if isinstance(value, (str, Path)) else _resolve_profile(value, config)
    if resolved['execution']['name'] != 'cpu-replicated-gloo':
        raise ValueError('Replicated execution requires the cpu-replicated-gloo profile')
    return resolved


def _policy(profile, values):
    defaults = {'startup_timeout': profile['preflight']['timeout'],
                'command_timeout': profile['preflight']['timeout'],
                'collective_timeout': profile['preflight']['collective_timeout'], 'total_timeout': 3600.0,
                'observer_timeout': 5.0, 'preview_timeout': 60.0}
    if values is not None:
        if not isinstance(values, dict) or set(values) - defaults.keys():
            raise ValueError('Unknown replicated service_policy field')
        defaults.update(values)
    for key, value in defaults.items():
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'service_policy.{key} must be finite positive seconds')
        defaults[key] = float(value)
    if defaults['collective_timeout'] > min(defaults['startup_timeout'], defaults['command_timeout']):
        raise ValueError('collective_timeout must not exceed startup_timeout or command_timeout')
    return defaults


class ReplicatedExecutionFactory:
    def __init__(self, profile, *, service_policy=None):
        self.profile, self.service_policy = profile, service_policy

    @staticmethod
    def environment():
        # Actual installed runtime/source is supplied by supervised workers in
        # ExecutionInfo, never inferred from a torch-free parent's environment.
        return {'runtime': {'device': 'cpu', 'backend': 'gloo', 'runtime_checked': False},
                'source': {'runtime_checked': False}}

    def __call__(self, config):
        return ReplicatedExecution(config, self.profile, service_policy=self.service_policy)


class ReplicatedExecution:
    def __init__(self, config, profile, *, service_policy=None):
        self.config = config
        if config['sampling']['count'] > MAX_SAMPLE_COUNT:
            raise ValueError(f'Replicated final inference sample count exceeds {MAX_SAMPLE_COUNT}; no samples were truncated')
        self.profile = _profile(profile, config)
        self.policy = _policy(self.profile, service_policy)
        self.context = self.service = self.authority = self.information = None
        self.step, self._inference_available = 0, False
        self._closed = self._poisoned = False
        self.observer = None

    def configure_attempt(self, context, *, preview_every, on_event):
        if self.context is not None:
            raise ValueError('Replicated execution attempt identity is immutable')
        if on_event is not None:
            self.observer = BoundedObserver(on_event, timeout=self.policy['observer_timeout'],
                run_id=context.run_id, attempt_id=context.attempt_id)
        self.context = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(context).items()}
        return {'execution': self.profile['execution'], 'service_policy': self.policy}

    def _fail(self, error):
        self._poisoned = True
        if self.service is not None:
            self.service._abort_preserving(error)
        if isinstance(error, (KeyboardInterrupt, SystemExit, FatalExecutionError)):
            raise error
        raise FatalExecutionError(f'Replicated execution failed: {type(error).__name__}: {error}') from error

    def _command(self, operation, payload=None):
        if self._closed or self._poisoned or self.service is None:
            raise FatalExecutionError('Replicated execution is closed or poisoned')
        try:
            return self.service.command(operation, payload)
        except BaseException as error:
            self._fail(error)

    def _results(self, response, *, expected_step=None):
        try:
            results = response['results']
            if type(results) is not list or len(results) != self.profile['execution']['world_size']:
                raise ValueError('Complete-boundary results require exactly one result per rank')
            for rank, item in enumerate(results):
                if (not isinstance(item, dict) or type(item.get('step')) is not int or item['step'] < 0
                        or item.get('ready') is not True or type(item.get('inference_available')) is not bool):
                    raise ValueError(f'Invalid complete-boundary result from rank {rank}')
                if expected_step is not None and item['step'] != expected_step:
                    raise ValueError(f'Rank {rank} returned step {item["step"]}, expected {expected_step}')
                if item['step'] != results[0]['step'] or item['inference_available'] != results[0]['inference_available']:
                    raise ValueError('Ranks disagree on completed step or inference availability')
            return results
        except BaseException as error:
            self._fail(error)

    def _open(self):
        if self._closed or self._poisoned:
            raise FatalExecutionError('Replicated execution is closed or poisoned')
        if self.context is None:
            raise ValueError('Configure an immutable attempt before starting replicated execution')
        if self.service is not None:
            return
        try:
            self.service = CPUWorkerService(_create_worker, _handle_command,
                args=(config_values(self.config), self.profile['execution'], self.context),
                run_id=self.context['run_id'], attempt_id=self.context['attempt_id'],
                world_size=self.profile['execution']['world_size'],
                **{key: self.policy[key] for key in ('startup_timeout', 'command_timeout', 'collective_timeout', 'total_timeout')})
            self.service.start()
            results = self._results(self._command('describe'), expected_step=0)
            info = results[0]['information']
            digest = hashlib.sha256(_encoded(info)).hexdigest()
            if any(item.get('identity_sha256') != digest for item in results):
                raise ValueError('Ranks disagree on execution identity acknowledgement')
            self.information = info
            if not info['recovery_reasons']:
                self.authority = CheckpointCommitAuthority(self.context['run_dir'], run_id=self.context['run_id'],
                    attempt_id=self.context['attempt_id'], identity=info['identity'])
        except BaseException as error:
            self._fail(error)

    def start(self):
        self._open()
        info = self.information
        return ExecutionInfo(step=self.step, data_identity=info['data_identity'],
            recovery_reasons=info['recovery_reasons'], checkpoint_metadata={}, environment=info['environment'])

    def restore(self, run_dir, checkpoint, run_id, config_sha256):
        if str(Path(run_dir).resolve()) != self.context['run_dir'] or run_id != self.context['run_id']:
            raise ValueError('Restore run differs from the configured attempt')
        if fingerprint(self.config) != config_sha256:
            raise ValueError('Resume configuration differs from the original run')
        self._open()
        if self.information['recovery_reasons']:
            raise ValueError('Recovery unsupported: ' + '; '.join(self.information['recovery_reasons']))
        results = self._results(self._command('restore', {'checkpoint': str(checkpoint) if checkpoint is not None else None}))
        try:
            if any(_encoded(item) != _encoded(results[0]) for item in results):
                raise ValueError('Ranks disagree on restored checkpoint')
            result = results[0]
            if result['saved_step'] != result['step']:
                raise ValueError('Restored step differs from saved checkpoint step')
            self.step, self._inference_available = result['step'], result['inference_available']
            return Restored(checkpoint_path=Path(result['checkpoint_path']), step=self.step)
        except BaseException as error:
            self._fail(error)

    def update(self):
        results = self._results(self._command('update'), expected_step=self.step + 1)
        try:
            if any(_encoded(item) != _encoded(results[0]) for item in results):
                raise ValueError('Ranks disagree on global training metrics')
            metrics = results[0]['metrics']
            expected = {'event': 'train', 'step': self.step + 1,
                **{key: self.profile['execution'][key] for key in ('global_batch_size', 'local_batch_size', 'world_size')}}
            if self.profile['execution']['accumulation_steps'] > 1:
                expected.update({key: self.profile['execution'][key] for key in ('accumulation_steps', 'microbatch_size')})
            if any(type(metrics.get(key)) is not type(value) or metrics[key] != value for key, value in expected.items()):
                raise ValueError('Training metrics differ from completed step/global batch profile')
            for key in ('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'lr_scale'):
                if type(metrics.get(key)) not in (float, int) or not math.isfinite(metrics[key]):
                    raise ValueError(f'Invalid finite global metric: {key}')
            if type(metrics.get('objectives')) is not list or any(type(value) not in (float, int) or not math.isfinite(value) for value in metrics['objectives']):
                raise ValueError('Invalid global objective metrics')
            self.step += 1
            self._inference_available = results[0]['inference_available']
            return CompletedUpdate(step=self.step, metrics=metrics)
        except BaseException as error:
            self._fail(error)

    def checkpoint(self, run_dir, metadata):
        try:
            if self.authority is None or str(Path(run_dir).resolve()) != self.context['run_dir']:
                raise ValueError('No checkpoint authority for this run')
            sequence = self.service.next_sequence
            response = self._command('prepare', {'metadata': metadata,
                'command_sequence': sequence, 'controller_id': self.authority.controller_id})
            if type(response.get('sequence')) is not int or response['sequence'] != sequence:
                raise ValueError('Checkpoint command acknowledgement has the wrong sequence')
            results = self._results(response, expected_step=self.step)
            receipt = results[0]['receipt']
            digest = hashlib.sha256(_encoded(receipt)).hexdigest()
            if any(item.get('receipt_sha256') != digest for item in results):
                raise ValueError('Ranks disagree on complete checkpoint preparation')
            self.service.assert_healthy()
            return self.authority.commit(receipt, expected_command_sequence=sequence)
        except BaseException as error:
            self._fail(error)

    @property
    def inference_available(self):
        return self._inference_available

    def inference(self, bundle_dir, identity):
        results = self._results(self._command('inference', {'bundle_dir': str(bundle_dir), 'identity': identity}), expected_step=self.step)
        try:
            self.service.assert_healthy()
            result = results[0]
            directory = Path(bundle_dir).resolve()
            bundle, sample = Path(result['bundle_path']), Path(result['sample_path'])
            if bundle != directory / 'model.pt' or sample.parent != directory or not bundle.is_file() or not sample.is_file():
                raise ValueError('Invalid completed inference artifact paths')
            return ArtifactResult(bundle_path=bundle, sample_path=sample)
        except BaseException as error:
            self._fail(error)

    def preview(self, run_dir, identity, *, keep):
        from .previews import publish_preview_payload
        from .snapshot_renderer import render_snapshot
        temporary, result, error = None, None, None
        try:
            self.service.assert_healthy()
            if str(Path(run_dir).resolve()) != self.context['run_dir']:
                raise ValueError('Preview run differs from configured attempt')
            temporary = tempfile.TemporaryDirectory(prefix='.preview-', dir=self.context['attempt_dir'])
            snapshot, output = Path(temporary.name) / 'snapshot.pt', Path(temporary.name) / 'preview.json'
            results = self._results(self._command('preview-snapshot', {'path': str(snapshot), 'identity': identity}),
                                    expected_step=self.step)
            try:
                descriptor = results[0]['snapshot']
            except BaseException as failure:
                self._fail(failure)
            payload = render_snapshot(snapshot, descriptor, identity, self.step, output,
                                      timeout=self.policy['preview_timeout'])
            record, index, errors = publish_preview_payload(run_dir, payload, identity, self.step, keep)
            result = PreviewResult(record=record, index=index, errors=errors)
        except BaseException as failure:
            error = failure
        finally:
            if temporary is not None:
                try:
                    temporary.cleanup()
                except BaseException as cleanup:
                    if error is None:
                        error = cleanup
                    elif hasattr(error, 'add_note'):
                        error.add_note(f'Preview cleanup also failed: {cleanup}')
            # Cover capture, rendering, publication AND temporary cleanup. Optional
            # observer failures cannot hide a dead/poisoned training group, and
            # cleanup must never replace an original fatal error or interrupt.
            try:
                if self._poisoned:
                    raise FatalExecutionError('Training group was poisoned during preview capture')
                self.service.assert_healthy()
            except BaseException as health:
                self._fail(error if isinstance(error, (FatalExecutionError, KeyboardInterrupt, SystemExit)) else health)
        if error is not None:
            raise error
        return result

    def observe(self, callback, event):
        # The controller's local warning wrapper is intentionally not sent to a
        # child. Configure validated the original importable callback once.
        if self.observer is None or self.observer.disabled:
            return
        try:
            self.observer.deliver(event)
        finally:
            # Terminal callbacks run after numerical shutdown. During training,
            # even a failed optional callback must not hide a failed rank.
            if self.service is not None and not self._closed:
                try:
                    self.service.assert_healthy()
                except BaseException as error:
                    self._fail(error)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.service is not None:
                if self._poisoned:
                    self.service.abort()
                else:
                    self.service.close()
        finally:
            if self.authority is not None:
                self.authority.close()


def run_train(config_path, run_dir, steps=None, *, profile, service_policy=None,
              checkpoint_every=100, max_seconds=None, stop_after_steps=None,
              on_event=None, preview_every=0, preview_keep=3):
    """Internal shared-controller training; numerical imports remain in workers."""
    from .run_controller import run_train as controller_train
    return controller_train(config_path, run_dir, steps, checkpoint_every=checkpoint_every,
        max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
        preview_every=preview_every, preview_keep=preview_keep,
        execution_factory=ReplicatedExecutionFactory(profile, service_policy=service_policy))


def run_resume(run_dir, checkpoint=None, config_path=None, *, profile=None, service_policy=None,
               checkpoint_every=None, max_seconds=None, stop_after_steps=None,
               on_event=None, preview_every=None, preview_keep=None):
    """Internal strict fixed-topology resume, validating before attempt publication."""
    from .run_controller import run_resume as controller_resume
    if profile is None:
        manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
        execution = manifest.get('execution', {})
        if execution.get('name') != 'cpu-replicated-gloo':
            raise ValueError('Run has no cpu-replicated-gloo execution identity')
        profile = {'schema_version': 1, 'execution': {key: execution[key] for key in ('name', 'world_size', 'accumulation_steps')}}
    return controller_resume(run_dir, checkpoint, config_path, checkpoint_every=checkpoint_every,
        max_seconds=max_seconds, stop_after_steps=stop_after_steps, on_event=on_event,
        preview_every=preview_every, preview_keep=preview_keep,
        execution_factory=ReplicatedExecutionFactory(profile, service_policy=service_policy))
