"""Fresh-process construction checks, explicit limitations and dependency errors."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


FIXTURES = '''
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(4, 2)
        if kind == 'nonfinite':
            self.register_buffer('sentinel', torch.tensor(float('nan')), persistent=False)
        elif kind == 'complex':
            self.register_buffer('sentinel', torch.tensor(1 + 2j), persistent=False)
        elif kind == 'threads':
            torch.set_num_threads(2)
    def forward(self, x):
        raise AssertionError('preflight must not execute forward')
    def get_extra_state(self):
        if self.kind == 'identity-threads':
            torch.set_num_threads(2)
        return {}
    def set_extra_state(self, state):
        pass

class UnknownData:
    def __call__(self, count, generator):
        raise AssertionError('preflight must not sample data')
'''


DRIVER = '''
import importlib.abc
import json
from pathlib import Path
import sys
from hypergan.config import resolve_config
from hypergan.execution_preflight import preflight
from hypergan.execution_profiles import resolve_execution_profile

def main():
    mode, root = sys.argv[1:]
    if mode.startswith('missing-'):
        missing = mode.split('-', 1)[1]
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == missing:
                    raise ModuleNotFoundError('No module named ' + repr(missing), name=missing)
        sys.meta_path.insert(0, Block())
    config = {'training': {'steps': 3, 'batch_size': 8},
              'prior': {'args': {'num_particles': 20, 'z_dim': 4}},
              'components': {
                'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
                'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}}},
              'sampling': {'count': 4}}
    if mode in ('nonfinite', 'complex', 'threads', 'identity-threads'):
        config['components']['generator'] = {'factory': 'preflight_fixtures:Generator', 'args': {'kind': mode}, 'inputs': {'x': 'latent'}}
    if mode == 'unsupported':
        config['data'] = {'factory': 'preflight_fixtures:UnknownData', 'args': {}}
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo' if mode in ('replicated', 'unsupported') else 'cpu-single'},
               'preflight': {'timeout': 25, 'collective_timeout': 10}}
    if mode == 'replicated':
        profile['execution']['accumulation_steps'] = 2
        config = resolve_config(config)
        profile = resolve_execution_profile(profile, config)
    try:
        result = preflight(config, profile)
        assert mode in ('single', 'replicated', 'unsupported', 'expected'), mode
        assert result['runtime_checked'] and result['scope'] == 'construction-only'
        assert result['identity']['runtime']['threads'] == 1
        assert all(rank['step'] == 0 for rank in result['ranks'])
        assert result['checker'].keys() == {'hypergan.execution_preflight', 'hypergan.execution_profiles', 'hypergan.cpu_workers'}
        if mode == 'replicated':
            assert result['identity']['strategy']['gradient_reduction'] == 'post-backward-mean'
            assert result['identity']['execution']['microbatch_size'] == 2
        if mode == 'unsupported':
            assert not result['identity']['recovery']['supported']
            assert any('resume_stateless' in warning for warning in result['warnings'])
        if mode == 'expected':
            profile['preflight'] = {'timeout': 26, 'collective_timeout': 11}
            matched = preflight(config, profile, expected_identity=result['identity'])
            assert matched['identity'] == result['identity']
            forged = json.loads(json.dumps(result['identity']))
            forged['runtime']['threads'] = True
            try:
                preflight(config, profile, expected_identity=forged)
            except ValueError as error:
                assert 'identity.runtime.threads (type differs)' in str(error)
            else:
                raise AssertionError('boolean masquerading as thread count was accepted')
            config['data'] = {'factory': 'gaussian_grid', 'args': {'side': 12, 'noise': .015}}
            try:
                preflight(config, profile, expected_identity=result['identity'])
            except ValueError as error:
                assert 'identity.data_contract.identity.specification.args.side' in str(error)
            else:
                raise AssertionError('changed actual data identity was accepted')
        print(json.dumps(result))
    except RuntimeError as error:
        assert mode not in ('single', 'replicated', 'unsupported', 'expected'), str(error)
        message = str(error)
        if mode.startswith('missing-'):
            assert missing in message and 'hypergan[train]' in message and 'rank 0' in message
        elif mode == 'nonfinite':
            assert 'graph.buffer.models.generator.sentinel' in message and 'nonfinite' in message
        elif mode == 'complex':
            assert 'graph.buffer.models.generator.sentinel' in message and 'float32' in message
        elif mode == 'threads':
            assert 'threads' in message and 'found 2' in message
        elif mode == 'identity-threads':
            assert 'Identity/state inspection changed CPU runtime threads' in message
        print(json.dumps({'error': message}))
    assert not {'torch', 'numpy', 'particlegan'} & sys.modules.keys(), 'parent imported numerical runtime'

if __name__ == '__main__':
    main()
'''


@pytest.mark.parametrize('mode', ['single', 'replicated', 'unsupported', 'expected', 'nonfinite', 'complex', 'threads', 'identity-threads',
                                  'missing-torch', 'missing-numpy', 'missing-particlegan'])
def test_preflight_runtime_contract(tmp_path, mode):
    driver = tmp_path / 'preflight_driver.py'
    # Install the optional-import blocker at module load in every spawned child.
    program = DRIVER
    if mode.startswith('missing-'):
        program = program.replace("def main():", """class BlockChild(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        missing = 'MISSING'
        if fullname.split('.')[0] == missing:
            raise ModuleNotFoundError('No module named ' + repr(missing), name=missing)
sys.meta_path.insert(0, BlockChild())

def main():""".replace('MISSING', mode.split('-', 1)[1]))
    driver.write_text(program)
    (tmp_path / 'preflight_fixtures.py').write_text(FIXTURES)
    # Explicit path makes trusted fixtures importable even for installed -I tests.
    driver.write_text("import sys\nsys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))\n" + driver.read_text())
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(driver), mode, str(tmp_path)],
                            capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report.get('status') == 'passed' if mode in ('single', 'replicated', 'unsupported', 'expected') else 'error' in report
    assert not list(tmp_path.rglob('checkpoints')) and not list(tmp_path.rglob('model.pt'))


def test_forged_resolved_profile_is_rejected_before_launch(monkeypatch):
    from hypergan.config import resolve_config
    import hypergan.execution_preflight as runtime
    from hypergan.execution_profiles import resolve_execution_profile
    config = resolve_config({})
    profile = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-single'}}, config)
    profile['execution']['microbatch_size'] += 1
    def forbidden(*args, **kwargs):
        raise AssertionError('a forged profile reached worker launch')
    monkeypatch.setattr(runtime, 'launch_cpu_workers', forbidden)
    with pytest.raises(ValueError, match='profile.execution.microbatch_size'):
        runtime.preflight(config, profile)
