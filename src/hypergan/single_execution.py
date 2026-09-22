"""Synchronous native CPU/CUDA adapter; numerical state stays on this side."""
from pathlib import Path
import copy
from types import SimpleNamespace

import torch

from .checkpoints import capture_rng, read_checkpoint, restore_rng, restore_trainer, write_checkpoint
from .checkpoint_compatibility import CURRENT_VERSION, validate_runtime, validate_implementation
from .config import config_values, fingerprint, resume_compatible
from .metrics import validate_update_scalars
from .run_controller import ArtifactResult, CompletedUpdate, ExecutionInfo, PreviewResult, Restored
from .training import ReferenceTrainer, _implementation, _recovery_contract, apply_backend_policy, runtime_info, source_info
from .recipes import execution_device


class SingleProcessExecution:
    def __init__(self, config):
        self._config = config
        apply_backend_policy(config)
        self._device = execution_device(config['training']['device'])
        self._trainer = None
        self._last_batch = None
        self._previous_threads = None
        self._previous_device = None
        self._ready = False
        self._closed = False
        self._previews = None

    def environment(self):
        return {'runtime': runtime_info(self._device), 'source': source_info()}

    def _open(self):
        if self._closed:
            raise RuntimeError('Execution has already shut down')
        if self._previous_threads is None:
            self._previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        if self._trainer is None:
            device = self._device
            if device.type == 'cuda':
                self._previous_device = torch.cuda.current_device()
                torch.cuda.set_device(device)
            self._trainer = ReferenceTrainer(self._config)
            self._ready = True

    def _boundary(self):
        if self._closed or not self._ready:
            raise RuntimeError('Execution is not at a complete update boundary')

    def start(self):
        self._open()
        rng = capture_rng()
        streams = {name: value.get_state() for name, value in self._trainer.streams.items()}
        try:
            contract, reasons = _recovery_contract(self._trainer)
            metadata = {'config': config_values(self._config), 'config_sha256': fingerprint(self._config),
                        'runtime': runtime_info(self._trainer.device), 'data_contract': contract,
                        'implementation': _implementation(self._trainer),
                        'hypergan_checkpoint_version': CURRENT_VERSION, 'source': source_info()}
        finally:
            restore_rng(rng)
            for name, state in streams.items():
                self._trainer.streams[name].set_state(state)
        return ExecutionInfo(step=self._trainer.step, data_identity=contract['identity'],
                             recovery_reasons=reasons, checkpoint_metadata=metadata,
                             environment={'runtime': runtime_info(self._trainer.device), 'source': source_info()})

    def restore(self, run_dir, checkpoint, run_id, config_sha256):
        target, info, state = read_checkpoint(run_dir, checkpoint)
        if info['run_id'] != run_id:
            raise ValueError('Checkpoint belongs to a different run')
        if fingerprint(self._config) != config_sha256:
            import json
            saved = json.loads((Path(run_dir) / 'manifest.json').read_text())['config']
            if not resume_compatible(self._config, saved) or fingerprint(saved) != config_sha256:
                raise ValueError('Resume configuration differs from the run manifest')
        if fingerprint(self._config) != info['config_sha256']:
            if (not resume_compatible(self._config, info['config'])
                    or fingerprint(info['config']) != info['config_sha256']):
                raise ValueError('Resume configuration differs from checkpoint; only an increased '
                                 'training.steps with unchanged constant learning rate (lr_floor=1) is allowed')
        apply_backend_policy(self._config)
        # Warns on stderr as it is detected; the controller also records them on
        # the run so the owner can read them after the attempt has scrolled past.
        runtime_warnings = validate_runtime(info['runtime'], runtime_info(self._config['training']['device']))
        self._open()
        contract, reasons = _recovery_contract(self._trainer)
        if reasons:
            raise ValueError('Recovery unsupported: ' + '; '.join(reasons))
        if contract != info['data_contract']:
            raise ValueError('Resume data identity or state protocol differs from checkpoint')
        validate_implementation(info['implementation'], _implementation(self._trainer))
        self._ready = False
        self._last_batch = restore_trainer(self._trainer, state)
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        self._ready = True
        return Restored(checkpoint_path=target, step=self._trainer.step,
                        warnings=tuple(runtime_warnings))

    def tune(self, run_dir, on_event=None):
        """Calibrate owned initialization at step zero, before any optimizer update."""
        from .startup_tuning import tune_initialized
        self._boundary()
        if self._trainer.step != 0:
            raise ValueError('Startup tuning cannot modify a trained checkpoint')
        self._ready = False
        result = tune_initialized(self._trainer, run_dir, on_event=on_event)
        self._ready = True
        return result

    def update(self):
        self._boundary()
        self._ready = False
        row, self._last_batch = self._trainer.update()
        validate_update_scalars(row, len(self._config["objectives"]))
        self._ready = True
        return CompletedUpdate(step=self._trainer.step, metrics=row)

    def checkpoint(self, run_dir, metadata):
        self._boundary()
        if self._trainer.device.type == 'cuda':
            torch.cuda.synchronize(self._trainer.device)
        return write_checkpoint(run_dir, self._trainer, self._last_batch, metadata)

    def preview(self, run_dir, identity, *, keep):
        from .preview_snapshot import capture_snapshot_state
        from .preview_worker import PreviewWorker
        self._boundary()
        if self.preview_busy:
            raise RuntimeError('Preview worker is busy; skip before reserving or capturing another preview')
        if self._previews is None:
            self._previews = PreviewWorker()
        snapshot = capture_snapshot_state(self._trainer, self._last_batch, identity)
        self._previews.submit(None, None, dict(identity), self._trainer.step, run_dir, keep,
                              snapshot_state=snapshot)

    def evaluation_snapshot(self, run_dir, identity):
        from .evaluation_snapshot import capture_evaluation_state
        self._boundary()
        return {'state': capture_evaluation_state(self._trainer, identity)}

    @property
    def preview_busy(self):
        return self._previews is not None and self._previews.busy

    def poll_preview(self, *, wait=False):
        result = self._previews.poll(wait=wait) if self._previews is not None else None
        return PreviewResult(**result) if result is not None else None

    def close_previews(self):
        return self.poll_preview(wait=True)

    def abort_previews(self):
        return self._previews.abort() if self._previews is not None else False

    @property
    def inference_available(self):
        return self._last_batch is not None

    def inference(self, bundle_dir, identity):
        from .artifacts import save_bundle, sample
        self._boundary()
        rng = capture_rng()
        streams = {name: value.get_state() for name, value in self._trainer.streams.items()}
        try:
            snapshot = SimpleNamespace(config=copy.deepcopy(self._config), step=self._trainer.step,
                ema_graph=copy.deepcopy(self._trainer.ema_graph), ema_prior=copy.deepcopy(self._trainer.ema_prior),
                artifact_identity=copy.deepcopy(identity))
            save_bundle(bundle_dir, snapshot, copy.deepcopy(self._last_batch))
            sample_path = sample(bundle_dir, count=self._config['sampling']['count'], seed=self._config['sampling']['seed'])
        finally:
            restore_rng(rng)
            for name, state in streams.items():
                self._trainer.streams[name].set_state(state)
        return ArtifactResult(bundle_path=Path(bundle_dir) / 'model.pt', sample_path=sample_path)

    def observe(self, callback, event):
        from .bounded_cli_output import CLIProgress
        # The internal sink only submits bounded text and refreshes console
        # policy. It cannot touch numerical state, so avoid copying/restoring
        # Python, NumPy and every visible CUDA RNG on each progress event.
        if type(callback) is CLIProgress:
            callback(event)
            return
        rng, threads = capture_rng(), torch.get_num_threads()
        device = torch.cuda.current_device() if torch.cuda.is_initialized() else None
        try:
            callback(event)
        finally:
            restore_rng(rng)
            if device is not None:
                torch.cuda.set_device(device)
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)

    def shutdown(self):
        if self._closed:
            return
        try:
            self.abort_previews()
        finally:
            self._shutdown_training()

    def _shutdown_training(self):
        self._closed = True
        self._ready = False
        if self._previous_device is not None:
            torch.cuda.set_device(self._previous_device)
            self._previous_device = None
        if self._previous_threads is not None:
            torch.set_num_threads(self._previous_threads)
            self._previous_threads = None
