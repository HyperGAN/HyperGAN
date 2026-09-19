"""Synchronous native CPU/CUDA adapter; numerical state stays on this side."""
from pathlib import Path
import copy
from types import SimpleNamespace

import torch

from .checkpoints import capture_rng, read_checkpoint, restore_rng, restore_trainer, write_checkpoint
from .config import config_values, fingerprint
from .run_controller import ArtifactResult, CompletedUpdate, ExecutionInfo, PreviewResult, Restored
from .training import ReferenceTrainer, _implementation, _recovery_contract, runtime_info, source_info
from .recipes import execution_device


class SingleProcessExecution:
    def __init__(self, config):
        self._config = config
        self._device = execution_device(config['training']['device'])
        self._trainer = None
        self._last_batch = None
        self._previous_threads = None
        self._previous_device = None
        self._ready = False
        self._closed = False

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
                        'implementation': _implementation(self._trainer)}
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
        if fingerprint(self._config) != info['config_sha256'] or fingerprint(self._config) != config_sha256:
            raise ValueError('Resume configuration differs from checkpoint; total training schedule cannot change')
        if runtime_info(self._config['training']['device']) != info['runtime']:
            raise ValueError('Resume runtime/topology differs from checkpoint')
        self._open()
        contract, reasons = _recovery_contract(self._trainer)
        if reasons:
            raise ValueError('Recovery unsupported: ' + '; '.join(reasons))
        if contract != info['data_contract']:
            raise ValueError('Resume data identity or state protocol differs from checkpoint')
        if _implementation(self._trainer) != info['implementation']:
            raise ValueError('Resume implementation differs from checkpoint')
        self._ready = False
        self._last_batch = restore_trainer(self._trainer, state)
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        self._ready = True
        return Restored(checkpoint_path=target, step=self._trainer.step)

    def update(self):
        self._boundary()
        self._ready = False
        row, self._last_batch = self._trainer.update()
        self._ready = True
        return CompletedUpdate(step=self._trainer.step, metrics=row)

    def checkpoint(self, run_dir, metadata):
        self._boundary()
        if self._trainer.device.type == 'cuda':
            torch.cuda.synchronize(self._trainer.device)
        return write_checkpoint(run_dir, self._trainer, self._last_batch, metadata)

    def preview(self, run_dir, identity, *, keep):
        from .previews import publish_preview
        self._boundary()
        record, index, errors = publish_preview(run_dir, self._trainer, self._last_batch, identity, keep=keep)
        return PreviewResult(record=record, index=index, errors=errors)

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
        self._closed = True
        self._ready = False
        if self._previous_device is not None:
            torch.cuda.set_device(self._previous_device)
            self._previous_device = None
        if self._previous_threads is not None:
            torch.set_num_threads(self._previous_threads)
            self._previous_threads = None
