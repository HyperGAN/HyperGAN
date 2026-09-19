"""Installed public CLI acceptance for supervised local execution and recovery.

Only a custom recipe's import directory is added by the subprocess bootstrap;
commands and parsing use the public CLI, without patching execution internals.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest

from hypergan.run_state import run_lock


_spec = importlib.util.spec_from_file_location(
    '_public_job_fixtures', Path(__file__).with_name('test_replicated_observer_acceptance.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

MODELS = _base.MODELS.replace(
    'torch.randn(len(x), 2)', 'torch.randn(len(x), 2, device=x.device)')


def setup(tmp_path, device='cpu'):
    _, config = _base._base._setup(tmp_path)
    (tmp_path / 'job_fixture.py').write_text(MODELS, encoding='utf-8')
    config.write_text(config.read_text().replace('device = "cpu"', f'device = "{device}"')
                      .replace('count = 256', 'count = 8'), encoding='utf-8')
    profile = tmp_path / 'execution.toml'
    name = 'cuda-replicated-nccl' if device == 'cuda' else 'cpu-replicated-gloo'
    profile.write_text(f'schema_version = 1\n[execution]\nname = "{name}"\n'
                       'world_size = 2\naccumulation_steps = 2\n', encoding='utf-8')
    return config, profile


def command(tmp_path, *args):
    bootstrap = ('import sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];'
                 'from hypergan.cli import main;result=main();'
                 'assert "torch" not in sys.modules,"replicated CLI imported torch";'
                 'raise SystemExit(result)')
    return [sys.executable, '-I', '-c', bootstrap, str(tmp_path), 'hypergan', *map(str, args)]


def environment(markers):
    markers.mkdir(exist_ok=True)
    return {**os.environ, 'HG_ACCEPTANCE_MARKERS': str(markers),
            'HG_ACCEPTANCE_MODE': 'public', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def reaped(markers):
    rows = [json.loads(line) for line in (markers / 'managed.jsonl').read_text().splitlines()]
    assert any(row['role'] == 'rank' for row in rows)
    for pid in {row['pid'] for row in rows} | {row['broker'] for row in rows}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def cli(tmp_path, markers, *args, success=True):
    process = subprocess.Popen(command(tmp_path, *args), cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True, env=environment(markers))
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('Public CLI exceeded independent whole-job deadline')
    assert (process.returncode == 0) == success, stdout + stderr
    if (markers / 'managed.jsonl').exists():
        reaped(markers)
    return stdout, stderr


def result(stdout, *, progress=False):
    if progress:
        rows = [json.loads(line) for line in stdout.splitlines()]
        assert rows[-1]['event'] == 'result'
        return rows[-1]['manifest']
    return json.loads(stdout)


def same_runs(left, right):
    actual_states, expected_states = _base._base._states(left), _base._base._states(right)
    assert len(actual_states) == len(expected_states) == 2
    for actual, expected in zip(actual_states, expected_states):
        _base._base._same(actual, expected)


def recover_public_job(tmp_path, device='cpu'):
    config, profile = setup(tmp_path, device)
    full, split = tmp_path / 'full', tmp_path / 'split'
    common = ('--no-server', '--checkpoint-every', 1, '--startup-timeout', 60,
              '--command-timeout', 45, '--collective-timeout', 20, '--total-timeout', 240)
    expected = result(cli(tmp_path, tmp_path / 'full-pids', 'train', config,
                          '--run-dir', full, '--profile', profile, '--no-previews', *common)[0])
    stopped = result(cli(tmp_path, tmp_path / 'stop-pids', 'train', config,
                         '--run-dir', split, '--profile', profile, '--stop-after-steps', 2,
                         '--preview-every', 1, '--preview-keep', 2, '--progress-json', *common)[0],
                     progress=True)
    assert stopped['status'] == 'stopped' and stopped['steps'] == stopped['last_durable_step'] == 2
    old_bundle, old_sample, first_sequence = _base._base._artifacts(stopped)
    stop_states = _base._base._states(split)
    # Profile is persisted, not re-read from the initial filename on resume.
    profile.unlink()
    resumed = result(cli(tmp_path, tmp_path / 'resume-pids', 'resume', split,
                         '--no-server', '--command-timeout', 55, '--total-timeout', 300)[0])
    assert expected['status'] == resumed['status'] == 'complete'
    assert resumed['steps'] == resumed['last_durable_step'] == 4
    assert resumed['execution'] == stopped['execution']
    assert resumed['execution']['accumulation_steps'] == 2
    assert resumed['service_policy']['command_timeout'] == 55
    assert resumed['service_policy']['total_timeout'] == 300
    assert resumed['attempt_id'] != stopped['attempt_id']
    assert resumed['preview_every'] == 1 and resumed['preview_keep'] == 2
    same_runs(full, split)
    assert Path(stopped['bundle_path']).read_bytes() == old_bundle
    assert Path(stopped['sample_path']).read_bytes() == old_sample
    assert _base._base._artifacts(resumed)[2] > first_sequence
    # Zero-update earlier selection must actually restore and reselect the old state.
    replay = result(cli(tmp_path, tmp_path / 'replay-pids', 'resume', split, '--no-server',
                        '--checkpoint', stopped['checkpoint_path'], '--max-seconds', '1e-9')[0])
    assert replay['status'] == 'stopped' and replay['steps'] == replay['last_durable_step'] == 2
    for actual, saved in zip(_base._base._states(split), stop_states):
        _base._base._same(actual, saved)
    again = result(cli(tmp_path, tmp_path / 'again-pids', 'resume', split, '--no-server')[0])
    assert again['status'] == 'complete' and again['attempt_index'] == 4
    same_runs(full, split)
    for run in (full, split):
        with run_lock(run):
            pass
    assert not list(split.glob('observations/viewer-*.json'))
    return config, split


def test_public_cpu_profile_observation_and_earlier_resume_are_exact(tmp_path):
    recover_public_job(tmp_path)


def test_public_profile_conflicts_reject_before_viewer_and_attempt_creation(tmp_path):
    config, profile = setup(tmp_path)
    run = tmp_path / 'run'
    cli(tmp_path, tmp_path / 'start-pids', 'train', config, '--run-dir', run,
        '--profile', profile, '--stop-after-steps', 1, '--no-server')
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob('*') if path.is_file()}
    conflicting = tmp_path / 'conflicting.toml'
    conflicting.write_text(profile.read_text().replace('accumulation_steps = 2', 'accumulation_steps = 1'))
    # A competing service port makes the ordering observable: identity rejection
    # must win, without viewer construction or failed-attempt artifacts.
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        _, stderr = cli(tmp_path, tmp_path / 'conflict-pids', 'resume', run,
                        '--profile', conflicting, '--server',
                        '--server-port', occupied.getsockname()[1], success=False)
    assert any(word in stderr.lower() for word in ('identity', 'profile', 'execution'))
    assert 'address already in use' not in stderr.lower()
    assert not (tmp_path / 'conflict-pids' / 'managed.jsonl').exists()
    assert before == {path.relative_to(run): path.read_bytes() for path in run.rglob('*') if path.is_file()}


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
@pytest.mark.parametrize('closed', [False, True], ids=['unread-pipe', 'closed-pipe'])
def test_public_replicated_output_cannot_hold_workers_or_run_lock(tmp_path, closed, stream):
    import fcntl
    config, profile = setup(tmp_path)
    run, markers = tmp_path / 'run', tmp_path / 'pids'
    steps = 40 if stream == 'stdout' else 120
    output = tmp_path / 'other-stream.log'
    other = output.open('w')
    process = subprocess.Popen(command(tmp_path, 'train', config, '--run-dir', run,
                                       '--profile', profile, '--no-server', '--steps', steps,
                                       *(['--progress-json'] if stream == 'stdout' else [])),
                               cwd=tmp_path, stdout=subprocess.PIPE if stream == 'stdout' else other,
                               stderr=subprocess.PIPE if stream == 'stderr' else other,
                               text=True, start_new_session=True, env=environment(markers))
    pipe = getattr(process, stream)
    try:
        fcntl.fcntl(pipe.fileno(), fcntl.F_SETPIPE_SZ, 4096)
        if closed:
            pipe.close()
        # Deliberately never drain stdout until the coordinator exits. A normal
        # blocking writer cannot fit the event stream plus final result in 4 KiB.
        process.wait(timeout=150)
        other.flush()
        other_output = output.read_text()
        assert process.returncode == 0, other_output
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['last_durable_step'] == steps
        reaped(markers)
        with run_lock(run):
            pass
        durable_events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        assert len([event for event in durable_events if event['event'] == 'train']) == steps
        assert 'Traceback' not in other_output
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        if not pipe.closed:
            pipe.close()
        other.close()
