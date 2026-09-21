"""Bounded, read-only training retention probe; writes only JSON to stdout.

Run from the checkout with its installed Python dependencies:
CUDA_VISIBLE_DEVICES=0 python reports/memory-leak-audit-2026-09-20/training-memory-probe.py CONFIG.toml
This executes 128 real updates in memory, without creating a training run.
"""
import gc
import json
import sys
import time

import torch

from hypergan.config import load_config
from hypergan.evaluation_snapshot import capture_evaluation_state
from hypergan.preview_snapshot import capture_snapshot_state
from hypergan.training import ReferenceTrainer


def nbytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(nbytes(value) for value in state.values())
    if isinstance(state, (tuple, list)):
        return sum(nbytes(value) for value in state)
    return 0


def main():
    config = load_config(sys.argv[1])
    torch.set_num_threads(1)
    started = time.monotonic()
    trainer = ReferenceTrainer(config)

    def report(label):
        torch.cuda.synchronize()
        with open('/proc/self/status') as source:
            memory = {line.split(':')[0]: line.split(':')[1].strip()
                      for line in source if line.startswith(('VmRSS:', 'VmHWM:'))}
        print(json.dumps({'label': label,
                          'allocated': torch.cuda.memory_allocated(),
                          'reserved': torch.cuda.memory_reserved(),
                          'peak_allocated': torch.cuda.max_memory_allocated(),
                          'memory': memory,
                          'seconds': round(time.monotonic() - started, 3)}), flush=True)

    for index in range(128):
        row, batch = trainer.update()
        if (index + 1) % 32 == 0:
            gc.collect()
            report(f'update-{index + 1}')
    for name, capture in [
        ('preview', lambda identity: capture_snapshot_state(trainer, batch, identity)),
        ('evaluation', lambda identity: capture_evaluation_state(trainer, identity)),
    ]:
        for index in range(3):
            torch.cuda.reset_peak_memory_stats()
            frozen = capture({'run_id': 'memory-audit', 'attempt_id': 'bounded',
                              'sample_sequence': index + 1})
            del frozen
            gc.collect()
            report(f'{name}-capture-{index + 1}')
    print(json.dumps({
        'generator_bytes': nbytes(trainer.ema_graph.models['generator'].state_dict()),
        'ema_graph_bytes': nbytes(trainer.ema_graph.state_dict()),
        'ema_prior_bytes': nbytes(trainer.ema_prior.state_dict()),
        'dataset_bytes': nbytes(trainer.data.images) + nbytes(trainer.data.labels),
    }))


if __name__ == '__main__':
    main()
