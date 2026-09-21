"""Adversarial preflight from real fresh worker groups, without a training run."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest


# Heavy: every test here starts real subprocesses or multi-rank jobs and
# measured at a second or more; see reports/test-durations-2026-09-20.txt.
pytestmark = pytest.mark.heavy


FACTORIES = '''
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from hypergan.recipes import MLP

class Generator(MLP):
    def __init__(self, *, root, mode, **kwargs):
        if mode == 'single':
            assert not dist.is_initialized(), 'single CPU construction unexpectedly initialized Gloo'
            rank = 0
        else:
            assert dist.is_initialized(), 'replicated construction needs its Gloo group'
            rank = dist.get_rank()
        Path(root, f'rank-{rank}.pid').write_text(str(os.getpid()))
        print(f'PYTHON-CONSTRUCTOR rank={rank}', flush=True)
        os.write(1, f'NATIVE-CONSTRUCTOR rank={rank}\\n'.encode())
        if rank == 1 and mode == 'constructor':
            raise ValueError('rank-one fixture constructor refused its input')
        if rank == 1 and mode == 'hang':
            time.sleep(120)
        super().__init__(**kwargs)

    def forward(self, *args, **kwargs):
        raise AssertionError('preflight executed generator forward')

class Data:
    resume_stateless = True

    def __init__(self, *, mode, **unused):
        self.mode = mode

    def resume_identity(self):
        return {'dataset': 'independent-preflight-fixture',
                'revision': dist.get_rank() if self.mode == 'identity' else 0}

    def __call__(self, *args, **kwargs):
        raise AssertionError('preflight consumed a data batch')
'''

DRIVER = '''
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import time
from hypergan.config import DEFAULT, resolve_config, write_default
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.execution_preflight import preflight

if __name__ == '__main__':
    root, mode = Path(sys.argv[1]), sys.argv[2]
    raw = copy.deepcopy(DEFAULT)
    raw['components']['generator']['factory'] = 'acceptance_factories:Generator'
    raw['components']['generator']['args'].update(root=str(root), mode=mode)
    raw['data'] = {'factory': 'acceptance_factories:Data', 'args': {'mode': mode}}
    config = resolve_config(raw)
    profile = resolve_execution_profile({
        'schema_version': 1,
        'execution': {'name': 'cpu-single' if mode == 'single' else 'cpu-replicated-gloo', 'world_size': 1 if mode == 'single' else 2, 'accumulation_steps': 1 if mode == 'single' else 2},
        'preflight': {'timeout': 8 if mode == 'hang' else 25,
                      'collective_timeout': 5 if mode == 'hang' else 15},
    }, config)
    started = time.monotonic()
    try:
        if mode in ('success', 'single'):
            from hypergan.cli import main
            config_path = write_default(root / 'project', device="cpu")
            content = config_path.read_text()
            content = content.replace('factory = "mlp"', 'factory = "acceptance_factories:Generator"', 1)
            content = content.replace('[components.generator.args]', '[components.generator.args]\\nroot = ' + json.dumps(str(root)) + '\\nmode = ' + json.dumps(mode))
            content = content.replace('factory = "gaussian_grid"', 'factory = "acceptance_factories:Data"')
            content = content.replace('[data.args]', '[data.args]\\nmode = ' + json.dumps(mode))
            config_path.write_text(content)
            profile_path = root / 'profile.toml'
            profile_text = 'schema_version = 1\\n[execution]\\nname = "cpu-replicated-gloo"\\nworld_size = 2\\naccumulation_steps = 2\\n[preflight]\\ntimeout = 25\\ncollective_timeout = 15\\n'
            if mode == 'single':
                profile_text = profile_text.replace('cpu-replicated-gloo', 'cpu-single').replace('world_size = 2', 'world_size = 1').replace('accumulation_steps = 2', 'accumulation_steps = 1')
            profile_path.write_text(profile_text)
            assert main(['preflight', str(config_path), '--profile', str(profile_path), '--runtime']) == 0
            outcome = {'cli_returncode': 0}
        else:
            preflight(config, profile)
            raise AssertionError(f'unexpected successful preflight: {mode}')
    except (ValueError, RuntimeError, TimeoutError) as error:
        assert mode not in ('success', 'single'), str(error)
        outcome = {'error': str(error), 'error_type': type(error).__name__}
    outcome['seconds'] = time.monotonic() - started
    assert not multiprocessing.active_children(), 'preflight leaked a direct child'
    Path(root, 'outcome.json').write_text(json.dumps(outcome), encoding='utf-8')
'''


@pytest.mark.parametrize('mode', ['single', 'success', 'constructor', 'identity', 'hang'])
def test_runtime_preflight_bounds_real_rank_failures_without_running_models(tmp_path, mode):
    (tmp_path / 'acceptance_factories.py').write_text(FACTORIES, encoding='utf-8')
    script = tmp_path / 'preflight_driver.py'
    script.write_text(DRIVER, encoding='utf-8')
    # Isolated installed-wheel checks cannot import cwd implicitly; the driver
    # explicitly receives its own trusted fixture directory below.
    code = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', code,
               str(tmp_path), str(script), str(tmp_path), mode]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=os.name == 'posix')
    try:
        stdout, stderr = process.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        pytest.fail('preflight exceeded independent outer deadline')
    assert process.returncode == 0, stdout + stderr
    if mode in ('success', 'single'):
        report = json.loads(stdout)  # Native Gloo/custom output must not pollute CLI JSON.
        assert report['runtime_checked'] is True
        assert 'PYTHON-CONSTRUCTOR rank=0' in stderr
        assert ('NATIVE-CONSTRUCTOR rank=0' if mode == 'single' else 'NATIVE-CONSTRUCTOR rank=1') in stderr
        if mode == 'single':
            assert report['identity']['runtime']['backend'] == 'none'
    outcome = json.loads((tmp_path / 'outcome.json').read_text(encoding='utf-8'))
    pids = sorted(tmp_path.glob('rank-*.pid'))
    assert len(pids) == (1 if mode == 'single' else 2), 'every actual rank constructor must execute'
    if os.name == 'posix':
        for path in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
    if mode == 'constructor':
        assert 'rank 1' in outcome['error']
        assert 'rank-one fixture constructor refused its input' in outcome['error']
    elif mode == 'identity':
        assert 'rank' in outcome['error'].lower() and 'identity' in outcome['error'].lower()
        assert 'data' in outcome['error'].lower(), outcome['error']
    elif mode == 'hang':
        assert outcome['seconds'] < 15
        assert any(word in outcome['error'].lower() for word in ('timeout', 'timed out', 'exceeded'))
    assert not list(tmp_path.rglob('model.pt'))
    assert not list(tmp_path.rglob('state.pt'))
    assert not list(tmp_path.rglob('manifest.json'))
