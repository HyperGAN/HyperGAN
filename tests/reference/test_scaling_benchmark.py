"""Exercise the real native/replicated benchmark without a cloud allocation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.heavy
@pytest.mark.parametrize('mode,batch_size', [('strong', '8'), ('weak', '4')])
def test_local_scaling_native_and_two_rank_gloo(tmp_path, mode, batch_size):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / 'scaling'
    completed = subprocess.run(
        [sys.executable, *(['-I'] if sys.flags.isolated else []), str(root / 'scripts' / 'benchmark_scaling.py'),
         '--backend', 'gloo', '--workload', 'mlp', '--mode', mode,
         '--batch-size', batch_size, '--warmup', '1',
         '--steps', '2', '--topologies', '1,2', '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=100)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'passed'
    assert [row['topology'] for row in report['results']] == ['1', '2']
    assert len({row['initial_sha256'] for row in report['results']}) == 1
    assert all(row['samples_per_second'] > 0 for row in report['results'])
    assert 'Samples/sec' in (output / 'comparison.md').read_text()
