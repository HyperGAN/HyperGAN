"""Independent complete NCCL job recovery and failure acceptance on two GPUs."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch


_spec = importlib.util.spec_from_file_location(
    '_cpu_job_fixtures', Path(__file__).parents[1] / 'reference/test_replicated_job_acceptance.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


FAULTS = '''
_INSTALLED = False


def install_faults():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    import hypergan.replicated_worker as worker
    original = worker.handle_command
    def handle(state, operation, payload):
        trainer = state['trainer']
        mode = os.environ.get('HG_ACCEPTANCE_MODE')
        root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
        if operation == 'update' and trainer.step == 1:
            if mode == 'rank-exit' and dist.get_rank() == 1:
                (root / 'rank-fault').write_text('rank-exit')
                os._exit(31)
            if mode == 'partial':
                calls = [0]
                def backward(gradient):
                    calls[0] += 1
                    if calls[0] == 2 and dist.get_rank() == 1:
                        assert all(int(row['step']) == 2 for row in trainer.opt_d.state.values())
                        assert all(int(row['step']) == 1 for row in trainer.opt_g.state.values())
                        (root / 'rank-fault').write_text('partial')
                        raise RuntimeError('injected NCCL second G microbatch failure')
                    return gradient
                next(trainer.graph.models['generator'].parameters()).register_hook(backward)
        try:
            return original(state, operation, payload)
        except BaseException:
            if mode == 'partial' and operation == 'update':
                assert trainer._poisoned and not trainer.checkpoint_ready and trainer.step == 1
                (root / f'poisoned-{dist.get_rank()}').write_text('true')
            raise
    worker.handle_command = handle
'''

MODELS = (_base.MODELS
    .replace('import torch.distributed as dist\n',
             'import torch.distributed as dist\ntorch.use_deterministic_algorithms(True)\ntorch.backends.cudnn.benchmark = False\n')
    .replace('        super().__init__(**kwargs)', '        super().__init__(**kwargs)\n        install_faults()', 1)
    .replace('torch.randn(len(x), 2)', 'torch.randn(len(x), 2, device=x.device)')
    .replace('    def load_state_dict(self, state):\n',
             "    def load_state_dict(self, state):\n        if os.environ.get('HG_ACCEPTANCE_MODE', '').startswith('corrupt-'):\n            (Path(os.environ['HG_ACCEPTANCE_MARKERS']) / 'unexpected-data-load').write_text('called')\n")) + FAULTS

CALLBACK = '''
import json
import os
from pathlib import Path
import random


def observe(event):
    random.random()
    root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
    with (root / 'callback-pids').open('a') as output:
        output.write(str(os.getpid()) + '\\n')
    with (root / 'delivered.jsonl').open('a') as output:
        output.write(json.dumps(event) + '\\n')
    event.clear()
'''

DRIVER = '''
import json
import os
from pathlib import Path
import sys
from hypergan import run_controller
from hypergan.replicated_execution import run_train, run_resume
from hypergan.run_state import run_lock
from nccl_observer import observe


def reaped(markers):
    assert len(list(markers.glob('rank-*.pid'))) == 2
    pids = set(map(int, (markers / 'all-ranks.txt').read_text().splitlines()))
    if (markers / 'callback-pids').exists():
        pids.update(map(int, (markers / 'callback-pids').read_text().splitlines()))
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError('managed process not reaped: ' + str(pid))


if __name__ == '__main__':
    config, run, markers, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    markers.mkdir(exist_ok=True)
    os.environ['HG_ACCEPTANCE_MARKERS'] = str(markers)
    os.environ['HG_ACCEPTANCE_MODE'] = mode
    assert 'torch' not in sys.modules, 'coordinator imported numerical runtime'
    original_atomic = run_controller.atomic_json
    def audited(path, value):
        if Path(path).name == 'manifest.json' and value.get('status') in ('complete', 'stopped', 'failed'):
            reaped(markers)
        return original_atomic(path, value)
    run_controller.atomic_json = audited
    policy = {'startup_timeout': 60, 'command_timeout': 45, 'collective_timeout': 20,
              'total_timeout': 240, 'preview_timeout': 40, 'observer_timeout': 20}
    observed = mode in ('split', 'resume', 'replay')
    options = {'service_policy': policy, 'on_event': observe if observed else None}
    try:
        if mode.startswith('corrupt-'):
            before = (run / 'manifest.json').read_bytes()
            pointer = (run / 'distributed-checkpoints/latest.json').read_bytes()
            attempts = sorted(path.name for path in (run / 'attempts').iterdir())
            try:
                run_resume(run, **options)
            except (ValueError, RuntimeError) as error:
                assert 'cuda' in str(error).lower() and 'rng' in str(error).lower(), str(error)
            else:
                raise AssertionError('invalid rank CUDA state was accepted')
            assert (run / 'manifest.json').read_bytes() == before
            assert (run / 'distributed-checkpoints/latest.json').read_bytes() == pointer
            assert sorted(path.name for path in (run / 'attempts').iterdir()) == attempts
            assert not (markers / 'unexpected-data-load').exists()
            result = {'rejected': True}
        elif mode in ('resume', 'replay', 'orphan'):
            if mode == 'replay':
                options.update(checkpoint=json.loads((run / 'saved-stop.json').read_text())['checkpoint_path'], max_seconds=1e-9)
            result = run_resume(run, **options)
        else:
            result = run_train(config, run,
                profile={'schema_version': 1, 'execution': {'name': 'cuda-replicated-nccl', 'world_size': 2, 'accumulation_steps': 2}},
                checkpoint_every=1, stop_after_steps=2 if mode == 'split' else None,
                preview_every=1 if observed else 0, preview_keep=2, **options)
    except (ValueError, RuntimeError) as error:
        if mode not in ('partial', 'rank-exit'):
            raise
        result = json.loads((run / 'manifest.json').read_text())
        assert (markers / 'rank-fault').read_text() == mode
        assert result['status'] == 'failed' and result['steps'] == result['last_durable_step'] == 1
        assert result['possible_lost_steps'] == 0
        assert not result.get('sample_path') and not result.get('bundle_path')
        assert 'rank' in str(error) and result['attempt_id'] in str(error)
    else:
        assert mode not in ('partial', 'rank-exit'), 'injected rank failure unexpectedly succeeded'
    reaped(markers)
    with run_lock(run):
        pass
    if mode == 'split':
        (run / 'saved-stop.json').write_text(json.dumps(result))
    assert 'torch' not in sys.modules
    (markers / 'result.json').write_text(json.dumps(result))
'''


@pytest.fixture(autouse=True)
def two_cuda_devices():
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Full NCCL recovery acceptance requires two local GPUs'


def _setup(tmp_path):
    driver, config = _base._setup(tmp_path)
    (tmp_path / 'job_fixture.py').write_text(MODELS, encoding='utf-8')
    (tmp_path / 'nccl_observer.py').write_text(CALLBACK, encoding='utf-8')
    driver.write_text(DRIVER, encoding='utf-8')
    config.write_text(config.read_text().replace('device = "cpu"', 'device = "cuda"')
                      .replace('count = 256', 'count = 8'), encoding='utf-8')
    return driver, config


def _environment():
    return {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def _run(driver, config, run, markers, mode):
    process = subprocess.Popen(_base._command(driver, config, run, markers, mode), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True, env=_environment())
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('NCCL whole-job acceptance exceeded independent deadline')
    assert process.returncode == 0, stdout + stderr
    return json.loads((markers / 'result.json').read_text())


def _portable(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu', 'checkpoint contains a rank-owned CUDA tensor'
    elif isinstance(value, dict):
        for item in value.values():
            _portable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _portable(item)


def _states(run):
    # No map_location: the serialized checkpoint itself must be CPU-portable.
    states = _base._states(run)
    for state in states:
        _portable(state)
        assert 'cuda' in state['rng'], 'checkpoint omitted global CUDA RNG'
    return states


def _equal(left, right):
    for actual, expected in zip(_states(left), _states(right)):
        _base._same(actual, expected)


def test_fresh_nccl_groups_resume_observed_state_and_earlier_selection_exactly(tmp_path):
    driver, config = _setup(tmp_path)
    full = _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    stopped = _run(driver, config, tmp_path / 'split', tmp_path / 'split-pids', 'split')
    old_bytes = _base._artifacts(stopped)
    resumed = _run(driver, config, tmp_path / 'split', tmp_path / 'resume-pids', 'resume')
    assert full['status'] == resumed['status'] == 'complete'
    assert resumed['execution']['name'] == 'cuda-replicated-nccl'
    assert not resumed['observation_errors']
    _equal(tmp_path / 'full', tmp_path / 'split')
    before = json.loads((tmp_path / 'split/previews/index.json').read_text())['previews']
    assert [row['step'] for row in before] == [3, 4]
    replayed = _run(driver, config, tmp_path / 'split', tmp_path / 'replay-pids', 'replay')
    assert replayed['status'] == 'stopped' and replayed['steps'] == 2
    assert json.loads((tmp_path / 'split/distributed-checkpoints/latest.json').read_text())['step'] == 2
    final = _run(driver, config, tmp_path / 'split', tmp_path / 'final-pids', 'resume')
    _equal(tmp_path / 'full', tmp_path / 'split')
    after = json.loads((tmp_path / 'split/previews/index.json').read_text())['previews']
    assert len(after) == 2 and all(row['identity']['attempt_id'] == final['attempt_id'] for row in after)
    assert min(row['identity']['sample_sequence'] for row in after) > max(row['identity']['sample_sequence'] for row in before)
    assert stopped['next_sample_sequence'] < resumed['next_sample_sequence'] < replayed['next_sample_sequence'] < final['next_sample_sequence']
    assert Path(stopped['bundle_path']).read_bytes() == old_bytes[0]
    assert Path(stopped['sample_path']).read_bytes() == old_bytes[1]
    _base._artifacts(final)


@pytest.mark.parametrize('mode', ['partial', 'rank-exit'])
def test_nccl_rank_failure_never_checkpoints_partial_state_and_recovers(tmp_path, mode):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    failed = _run(driver, config, tmp_path / 'fault', tmp_path / 'failure-pids', mode)
    assert failed['status'] == 'failed'
    assert json.loads((tmp_path / 'fault/distributed-checkpoints/latest.json').read_text())['step'] == 1
    events = [json.loads(line) for line in (tmp_path / 'fault/events.jsonl').read_text().splitlines()]
    assert [row['step'] for row in events if row['event'] == 'train'] == [1]
    assert [row['step'] for row in events if row['event'] == 'checkpoint'] == [0, 1]
    if mode == 'partial':
        assert all((tmp_path / f'failure-pids/poisoned-{rank}').read_text() == 'true' for rank in range(2))
    result = _run(driver, config, tmp_path / 'fault', tmp_path / 'takeover-pids', 'resume')
    assert result['status'] == 'complete' and result['attempt_index'] == 2
    _base._artifacts(result)
    _equal(tmp_path / 'full', tmp_path / 'fault')


def test_nccl_coordinator_death_reaps_native_blocked_ranks_and_allows_takeover(tmp_path):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    _run(driver, config, tmp_path / 'split', tmp_path / 'split-pids', 'split')
    harness = tmp_path / 'nccl_death.py'
    harness.write_text(_base.DEATH_HARNESS, encoding='utf-8')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), str(harness),
                                str(driver), str(config), str(tmp_path / 'split'), str(tmp_path / 'death-pids')],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True, env=_environment())
    try:
        stdout, stderr = process.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('NCCL coordinator-death harness exceeded independent deadline')
    assert process.returncode == 0, stdout + stderr
    proof = json.loads((tmp_path / 'death-pids/death-proof.json').read_text())
    result = _run(driver, config, tmp_path / 'split', tmp_path / 'takeover-pids', 'resume')
    assert result['status'] == 'complete' and result['attempt_index'] == 3
    assert result['attempt_id'] != proof['attempt_id']
    _base._artifacts(result)
    _equal(tmp_path / 'full', tmp_path / 'split')


@pytest.mark.parametrize('kind', ['device', 'inventory'])
def test_cuda_rank_rng_corruption_rejects_before_attempt_or_load_hooks(tmp_path, kind):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'run', tmp_path / 'stop-pids', 'split')
    root = tmp_path / 'run/distributed-checkpoints'
    generation = root / json.loads((root / 'latest.json').read_text())['checkpoint']
    manifest_path = generation / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    path = generation / 'rank-00001.pt'
    state = torch.load(path, weights_only=True)
    assert state['rng']['cuda_device'] == 1
    if kind == 'device':
        state['rng']['cuda_device'] = 0
    else:
        del state['rng']['cuda']
    torch.save(state, path)
    manifest['ranks'][1].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    result = _run(driver, config, tmp_path / 'run', tmp_path / 'reject-pids', 'corrupt-' + kind)
    assert result['rejected'] is True
