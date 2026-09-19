"""Required local GPU check for the production independent broker path."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_nccl_broker_binds_ranks_and_reaps_failed_group(tmp_path):
    driver = tmp_path / 'nccl_service_driver.py'
    driver.write_text('''
import json, os, sys, time
from pathlib import Path
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world):
    import torch
    import torch.distributed as dist
    assert dist.get_backend() == 'nccl'
    assert torch.cuda.current_device() == rank
    return rank

def handler(rank, operation, payload):
    import torch
    import torch.distributed as dist
    if operation == 'collective':
        value = torch.tensor([float(rank + 1)], device='cuda:' + str(rank))
        dist.all_reduce(value)
        torch.cuda.synchronize(rank)
        return {'rank': rank, 'device': str(value.device), 'sum': value.item()}
    if operation == 'crash':
        if rank == 1:
            os._exit(17)
        time.sleep(60)
    raise ValueError(operation)

if __name__ == '__main__':
    assert 'torch' not in sys.modules
    service = CPUWorkerService(factory, handler, backend='nccl', run_id='test', attempt_id='gpu',
                               startup_timeout=20, command_timeout=10, collective_timeout=5, total_timeout=40)
    with service:
        result = service.command('collective')['results']
        assert result == [{'rank': 0, 'device': 'cuda:0', 'sum': 3.0},
                          {'rank': 1, 'device': 'cuda:1', 'sum': 3.0}]
        service.assert_healthy()
        started = time.monotonic()
        try:
            service.command('crash')
        except RuntimeError as error:
            assert 'rank 1' in str(error) and ('17' in str(error) or 'closed' in str(error))
            assert time.monotonic() - started < 10
        else:
            raise AssertionError('Lost rank must fail the whole command')
    assert 'torch' not in sys.modules
    pids = [service.broker_pid, *service.worker_pids]
    assert all(not Path('/proc/' + str(pid)).exists() for pid in pids)
    Path(__file__ + '.json').write_text(json.dumps({'pids': pids, 'result': result}))
''')
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(driver)],
                            capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(Path(str(driver) + '.json').read_text())
    assert len(receipt['pids']) == 3
    assert all(not Path(f'/proc/{pid}').exists() for pid in receipt['pids'])


def test_nccl_preflight_constructs_in_broker_and_reports_common_runtime(tmp_path):
    driver = tmp_path / 'nccl_preflight_driver.py'
    driver.write_text('''
import json, sys
from pathlib import Path
from hypergan.execution_preflight import preflight

if __name__ == '__main__':
    profile = {'schema_version': 1, 'execution': {'name': 'cuda-replicated-nccl', 'accumulation_steps': 2},
               'preflight': {'timeout': 30, 'collective_timeout': 10}}
    result = preflight({'training': {'device': 'cuda'}}, profile)
    assert result['runtime_checked'] and result['scope'] == 'construction-only'
    assert result['identity']['runtime']['device'] == 'cuda'
    assert result['identity']['runtime']['backend'] == 'nccl'
    assert len(result['identity']['runtime']['cuda']['rank_devices']) == 2
    assert len(result['ranks']) == 2 and all(row['step'] == 0 for row in result['ranks'])
    assert result['identity']['execution']['microbatch_size'] == 4
    assert 'hypergan.cpu_worker_service' in result['checker']
    assert not {'torch', 'numpy', 'particlegan'} & sys.modules.keys()
    Path(__file__ + '.json').write_text(json.dumps(result))
''')
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(driver)],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(Path(str(driver) + '.json').read_text())
    assert report['identity']['recovery']['supported']
    assert not list(tmp_path.rglob('checkpoints'))
