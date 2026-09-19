"""Full worker-group metric publication and observation-only restore changes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from hypergan.config import write_default


DRIVER = '''
import json
from pathlib import Path
import sys
from hypergan.replicated_execution import run_train, run_resume

if __name__ == '__main__':
    root, device = Path(sys.argv[1]), sys.argv[2]
    profile = {'schema_version': 1, 'execution': {
        'name': 'cuda-replicated-nccl' if device == 'cuda' else 'cpu-replicated-gloo',
        'world_size': 2, 'accumulation_steps': 2}}
    options = {'profile': profile}
    run_train(root / 'on.toml', root / 'on', **options)
    run_train(root / 'off.toml', root / 'off', **options)
    stopped = run_train(root / 'on.toml', root / 'split', stop_after_steps=1, checkpoint_every=1, **options)
    run_resume(root / 'split', config_path=root / 'off.toml', **options)
    zero = run_resume(root / 'split', checkpoint=stopped['checkpoint_path'],
                      config_path=root / 'on.toml', max_seconds=1e-12, **options)
    assert zero['steps'] == 1
    run_resume(root / 'split', **options)
    (root / 'lineage.json').write_text(json.dumps({'parent': stopped, 'zero': zero}))
'''


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def verify_replicated_metrics(tmp_path, device):
    config = write_default(tmp_path / 'on.toml', device=device)
    config.write_text(config.read_text().replace('steps = 5', 'steps = 3')
                      .replace('num_particles = 20000', 'num_particles = 32')
                      .replace('count = 256', 'count = 8').replace('lazy_k = 1', 'lazy_k = 2'))
    (tmp_path / 'off.toml').write_text(config.read_text() + '\n[metrics]\npreset = "none"\n')
    driver = tmp_path / 'metrics_driver.py'
    driver.write_text(DRIVER)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    result = subprocess.run([sys.executable, str(driver), str(tmp_path), device],
                            capture_output=True, text=True, timeout=240, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    def states(name):
        root = tmp_path / name / 'distributed-checkpoints'
        pointer = json.loads((root / 'latest.json').read_text())
        return [torch.load(root / pointer['checkpoint'] / f'rank-{rank:05d}.pt', weights_only=True) for rank in range(2)]
    baseline = states('on')
    same(baseline, states('off'))
    same(baseline, states('split'))
    for name in ('on', 'off', 'split'):
        events = [json.loads(line) for line in (tmp_path / name / 'events.jsonl').read_text().splitlines()]
        for row in events:
            if row['event'] != 'train':
                continue
            assert 'd_loss' not in row and 'g_loss' not in row
            if name == 'off':
                assert row['metrics'] == {}
            if row['metrics']:
                m = row['metrics']
                assert abs(m['loss/d_total'] - m['loss/d_adversarial'] - m['loss/gradient_penalty']) < 1e-6
                assert abs(m['loss/g_total'] - m['loss/g_adversarial'] - m['loss/prior_regularizer']) < 1e-6
                assert row['world_size'] == 2 and row['accumulation_steps'] == 2
    lineage = json.loads((tmp_path / 'lineage.json').read_text())
    event = next(row for row in events if row['event'] == 'resume' and row['attempt_id'] == lineage['zero']['attempt_id'])
    assert event['parent_attempt_id'] == lineage['parent']['attempt_id'] and event['restored_step'] == 1
    assert not [row for row in events if row['event'] == 'train' and row['attempt_id'] == lineage['zero']['attempt_id']]


def test_cpu_replicated_metrics_on_off_and_revision_resume(tmp_path):
    verify_replicated_metrics(tmp_path, 'cpu')
