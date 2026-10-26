"""Real image components: NCCL updates, activation replay and exact recovery.

Synthetic tensors avoid downloads and image-quality claims. The installed CIFAR
generator and pixel critic still exercise convolution, normalization, attention
and the critic's double backward on both GPUs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.multi_gpu]

WORKER = '''
import copy
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.recipes import move_tensors

rank, root = int(sys.argv[1]), Path(sys.argv[2])
torch.set_num_threads(1)
torch.cuda.set_device(rank)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dist.init_process_group('nccl', init_method=(root / 'store').as_uri(), rank=rank,
                        world_size=2, timeout=timedelta(seconds=30))
try:
    config = resolve_config({
        'training': {'device': 'cuda', 'batch_size': 4, 'steps': 2, 'seed': 42},
        'components': {
            'generator': {'factory': 'hypergan.image_components:CIFARGenerator',
                          'args': {'z_dim': 4}, 'inputs': {'z': 'latent'}},
            'discriminator': {'factory': 'hypergan.image_components:_PixelDiscriminator',
                              'args': {'width': 8},
                              'inputs': {'x': 'candidate', 'xt': 'batch.context'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 1, 'kappa': .01},
    })
    real = torch.linspace(-1, 1, 4 * 3 * 32 * 32, device='cuda').reshape(4, 3, 32, 32).chunk(2)[rank]
    batch = {'real': real, 'context': torch.zeros_like(real)}
    ids = torch.arange(4, device='cuda').chunk(2)[rank]
    results = []
    for factor in (1, 2):
        trainer = ReplicatedTrainer(config, accumulation_steps=factor)
        before = trainer._digest(trainer.graph.models['generator'].state_dict())
        row, last = trainer.update(batch, (trainer.prior.z[ids], ids))
        assert all(math.isfinite(row[key]) for key in ('d_loss', 'g_loss', 'gradient_penalty'))
        assert row['gradient_penalty'] > 0
        assert trainer._digest(trainer.graph.models['generator'].state_dict()) != before
        trainer._assert_replicas('image update')
        run = root / f'images-a{factor}'
        run.mkdir(exist_ok=True)
        save_distributed_checkpoint(run, trainer, last,
            {'run_id': f'images-a{factor}', 'attempt_id': 'first', 'next_sample_sequence': 1})
        trainer.update(batch, (trainer.prior.z[ids], ids))
        expected = trainer._digest(move_tensors(copy.deepcopy(trainer_state(trainer, last)), 'cpu'))
        restored = ReplicatedTrainer(config, accumulation_steps=factor)
        restore_distributed_checkpoint(run, restored, {'run_id': f'images-a{factor}'})
        assert restored.step == 1
        _, restored_batch = restored.update(batch, (restored.prior.z[ids], ids))
        actual = restored._digest(move_tensors(trainer_state(restored, restored_batch), 'cpu'))
        assert actual == expected, f'image recovery changed full state at accumulation {factor}'
        assert restored.step == 2 and restored.checkpoint_ready
        restored._assert_replicas('restored image update')
        results.append({'accumulation_steps': factor, 'exact_recovery': True})
    (root / f'rank-{rank}.json').write_text(json.dumps(results))
finally:
    dist.destroy_process_group()
'''


def test_two_gpu_image_components_accumulate_and_resume_exactly(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Image NCCL qualification requires two GPUs'
    script = tmp_path / 'image_worker.py'
    script.write_text(WORKER)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f'rank-{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen(
                [sys.executable, *(['-I'] if sys.flags.isolated else []), str(script), str(rank), str(tmp_path)],
                stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + 120
        for child in children:
            child.wait(timeout=max(.1, deadline - time.monotonic()))
        for log in logs:
            log.flush()
        errors = '\n'.join(path.read_text() for path in sorted(tmp_path.glob('rank-*.log')))
        assert all(child.returncode == 0 for child in children), errors
        for rank in range(2):
            assert json.loads((tmp_path / f'rank-{rank}.json').read_text()) == [
                {'accumulation_steps': 1, 'exact_recovery': True},
                {'accumulation_steps': 2, 'exact_recovery': True}]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for log in logs:
            log.close()
