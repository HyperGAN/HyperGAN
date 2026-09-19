"""CLI structural preflight is inert and separates numerical identity from policy."""
import json
import subprocess
import sys

from hypergan.config import write_default


BLOCKED_CLI = '''
import importlib.abc
import multiprocessing
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'particlegan', 'PIL', 'preflight_custom'}:
            raise AssertionError('structural preflight imported runtime/custom code: ' + fullname)
def no_workers(*args, **kwargs):
    raise AssertionError('structural preflight tried to start workers')
sys.meta_path.insert(0, BlockRuntime())
multiprocessing.get_context = no_workers
from hypergan.cli import main
raise SystemExit(main(sys.argv[1:]))
'''


def _structural(config, profile):
    return subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), '-c',
                           BLOCKED_CLI, 'preflight', str(config), '--profile', str(profile)],
                          capture_output=True, text=True, timeout=15)


def _profile(path, *, timeout=60, collective_timeout=15, accumulation=1, world_size='2'):
    path.write_text(f'''schema_version = 1
[execution]
name = "cpu-replicated-gloo"
world_size = {world_size}
accumulation_steps = {accumulation}
[preflight]
timeout = {timeout}
collective_timeout = {collective_timeout}
''', encoding='utf-8')


def test_structural_cli_never_loads_custom_code_or_workers_and_excludes_timeouts_from_identity(tmp_path):
    config = write_default(tmp_path / 'project', device="cpu")
    config.write_text(config.read_text().replace('factory = "mlp"', 'factory = "preflight_custom:Generator"', 1))
    profile = tmp_path / 'profile.toml'
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
    results = []
    for timeout, collective, accumulation in [(60, 15, 1), (30, 5, 1), (30, 5, 2)]:
        _profile(profile, timeout=timeout, collective_timeout=collective, accumulation=accumulation)
        completed = _structural(config, profile)
        assert completed.returncode == 0, completed.stderr
        assert 'warning:' in completed.stderr, 'custom recipe remains visibly unqualified'
        report = json.loads(completed.stdout)
        assert report['stage'] == 'structural' and report['runtime_checked'] is False
        results.append(report['profile'])
    assert results[0]['execution'] == results[1]['execution']
    assert results[0]['preflight'] != results[1]['preflight']
    assert results[1]['execution'] != results[2]['execution']
    assert results[2]['execution']['global_batch_size'] == 16
    assert results[2]['execution']['local_batch_size'] == 8
    assert results[2]['execution']['microbatch_size'] == 4
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob('*')} == before | {profile.relative_to(tmp_path)}


def test_structural_cli_reports_invalid_profile_without_runtime_or_partial_json(tmp_path):
    config = write_default(tmp_path / 'project', device="cpu")
    profile = tmp_path / 'profile.toml'
    _profile(profile, world_size='true')
    completed = _structural(config, profile)
    assert completed.returncode == 1
    assert completed.stdout == ''
    assert 'world_size' in completed.stderr
    assert 'Traceback' not in completed.stderr
