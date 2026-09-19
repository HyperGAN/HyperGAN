"""Required opt-in real two-GPU gate; CPU CI never selects tests/cuda."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_gpu_nccl_collectives_gradients_and_timeout_cleanup(tmp_path):
    receipt = tmp_path / 'nccl.json'
    environment = dict(os.environ)
    # Keep NCCL's post-timeout diagnostic wait explicit and bounded; the
    # independent parent deadline still owns cleanup if its watchdog stalls.
    environment['TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC'] = '1000'
    process = subprocess.run(
        [sys.executable, *(['-I'] if sys.flags.isolated else []),
         str(Path(__file__).with_name('nccl_smoke.py')), '--output', str(receipt),
         '--fault-timeout', '--collective-timeout', '5', '--timeout', '35'],
        env=environment, capture_output=True, text=True, timeout=110)
    assert process.returncode == 0, f'{process.stdout}\n{process.stderr}\nRank logs: {receipt}.artifacts'
    result = json.loads(receipt.read_text())
    assert result['passed'] and result['collectives']['passed']
    assert result['missing_peer']['all_ranks_entered_fault']
    assert result['missing_peer']['nccl_timeout_reported']
    assert result['missing_peer']['failure'] == 'rank-failed'
    for phase in ('collectives', 'missing_peer'):
        assert result[phase]['seconds'] < 40  # 35-second job limit + 4-second reap grace.
        assert all(rank['reaped'] for rank in result[phase]['ranks'])
        if sys.platform.startswith('linux'):
            assert all(not Path(f"/proc/{rank['pid']}").exists() for rank in result[phase]['ranks'])
    assert len({row['uuid'] for row in result['collectives']['devices']}) == 2
