"""The isolated observer service does not require the training dependency extra."""
import subprocess
import sys


SCRIPT = '''
import importlib.abc
import os
import sys
from hypergan.cpu_worker_service import CPUWorkerService

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'particlegan', 'PIL'):
            raise AssertionError('group-free service imported: ' + fullname)

sys.meta_path.insert(0, Block())

def factory(rank, world_size):
    assert rank == 0 and world_size == 1
    return {'pid': os.getpid()}

def handler(state, operation, payload):
    assert operation == 'echo'
    return dict(state, payload=payload)

if __name__ == '__main__':
    import hypergan.previews
    import hypergan.snapshot_renderer
    with CPUWorkerService(factory, handler, run_id='run', attempt_id='attempt',
            world_size=1, initialize_process_group=False, startup_timeout=10,
            command_timeout=10, total_timeout=20) as service:
        result = service.command('echo', {'one': 1})['results'][0]
        assert result['payload'] == {'one': 1}
        assert result['pid'] == service.worker_pids[0]
        service.assert_healthy()
    assert service._process is None  # close joined the broker on every platform.
    if sys.platform == 'linux':
        from pathlib import Path
        assert not Path('/proc', str(result['pid'])).exists(), 'worker survived service close'
'''


def test_group_free_service_and_preview_parent_import_without_torch(tmp_path):
    script = tmp_path / 'group_free.py'
    script.write_text(SCRIPT)
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(script)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
