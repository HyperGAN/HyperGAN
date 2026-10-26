"""Qualification must exercise the requested topology and inherited GPU mapping."""
import pytest

from hypergan import qualify
from hypergan.cli import _parser


def test_cuda_world_size_option_reaches_the_runner(tmp_path, monkeypatch):
    args = _parser().parse_args(['qualify', '--cuda-world-size', '4', '--dry-run',
                                '--suite', 'cuda_replicated_nccl', '-o', str(tmp_path / 'receipt.json')])
    created = []
    original = qualify.QualificationRunner

    def runner(**kwargs):
        created.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(qualify, 'QualificationRunner', runner)
    assert qualify.run_qualification_cli(args) == 0
    assert created[0]['cuda_world_size'] == 4


@pytest.mark.parametrize('count', [1, 65, True, 2.5])
def test_invalid_cuda_world_size_is_rejected(count):
    with pytest.raises(ValueError, match='cuda_world_size'):
        qualify.QualificationRunner(cuda_world_size=count)


def test_requesting_four_gpus_does_not_qualify_only_two():
    runner = qualify.QualificationRunner(cuda_world_size=4)
    result = runner.run_cuda_replicated_nccl({
        'device_count': 2, 'cuda': {'available': True, 'nccl': True}})
    assert result['status'] == 'skipped'
    assert '4' in result['details']['reason']


def test_cuda_consensus_checks_every_requested_rank(monkeypatch):
    runner = qualify.QualificationRunner(cuda_world_size=4)
    ranks = [{'passed': True, 'final_weights': {'weight': [1.]}} for _ in range(4)]
    ranks[-1] = {'passed': True, 'final_weights': {'weight': [2.]}}
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return ranks

    monkeypatch.setattr(runner, '_run_subprocess_ranks', run)
    with pytest.raises(qualify.QualificationCheckFailed, match='diverged'):
        runner.run_cuda_replicated_nccl({
            'device_count': 4, 'cuda': {'available': True, 'nccl': True}})
    assert calls[0]['world_size'] == 4
    assert calls[0]['cuda_devices'] == [0, 1, 2, 3]


def test_gpu_subset_uses_visible_indices_and_rejects_hidden_devices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-third,GPU-first,GPU-second')
    assert qualify._cuda_device_mask([2, 0]) == 'GPU-second,GPU-third'
    with pytest.raises(ValueError, match='outside CUDA_VISIBLE_DEVICES'):
        qualify._cuda_device_mask([3])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='outside CUDA_VISIBLE_DEVICES'):
        qualify._cuda_device_mask([0, 1])


@pytest.mark.parametrize('mask', ['3,1', 'GPU-first,GPU-second', None])
def test_rank_subprocesses_preserve_gpu_mapping_and_nccl_tuning(tmp_path, monkeypatch, mask):
    if mask is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setenv('NCCL_P2P_DISABLE', '0')
    monkeypatch.delenv('NCCL_IB_DISABLE', raising=False)
    monkeypatch.setattr(qualify, 'WORKER_DRIVER_CODE', '''
import json, os, sys
from pathlib import Path
Path(sys.argv[6]).write_text(json.dumps({
    key: os.environ.get(key) for key in
    ('CUDA_VISIBLE_DEVICES', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE')}))
''')
    results = qualify.QualificationRunner()._run_subprocess_ranks(
        backend='nccl', world_size=2, cuda_devices=[0, 1])
    for result in results:
        assert result['CUDA_VISIBLE_DEVICES'] == (mask or '0,1')
        assert result['NCCL_P2P_DISABLE'] == '0'
        assert result['NCCL_IB_DISABLE'] is None


@pytest.mark.heavy
@pytest.mark.parametrize('world_size', [4, 8])
def test_full_training_driver_handles_more_than_two_ranks(world_size):
    runner = qualify.QualificationRunner()
    results = runner._run_subprocess_ranks(backend='gloo', world_size=world_size)
    assert len(results) == world_size
    assert [result['rank'] for result in results] == list(range(world_size))
    assert all(result['passed'] for result in results)
    assert all(result['final_weights'] == results[0]['final_weights'] for result in results)
    assert all(len(result['losses']) == 2 for result in results)
