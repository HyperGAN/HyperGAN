"""Shared-controller evidence at real accumulated update and persistence faults."""
import importlib.util
import json
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    '_replicated_job_fixtures', Path(__file__).with_name('test_replicated_job_acceptance.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


FAULTS = '''
_FAULT_INSTALLED = False


def install_faults():
    global _FAULT_INSTALLED
    if _FAULT_INSTALLED:
        return
    _FAULT_INSTALLED = True
    import hypergan.replicated_worker as worker
    original = worker.handle_command
    def handle(state, operation, payload):
        trainer = state['trainer']
        mode = os.environ.get('HG_ACCEPTANCE_MODE')
        markers = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
        if mode == 'stage' and operation == 'prepare' and trainer.step == 2 and dist.get_rank() == 0:
            import hypergan.distributed_checkpoints as checkpoints
            original_write = checkpoints.atomic_json
            def fail_manifest(path, value):
                if Path(path).name == 'manifest.json' and '.prepared' in Path(path).parts:
                    (markers / 'fault-entered').write_text('stage')
                    raise OSError('injected staging disk failure')
                return original_write(path, value)
            checkpoints.atomic_json = fail_manifest
        if mode in ('g-backward', 'ema') and operation == 'update' and trainer.step == 1:
            if mode == 'g-backward':
                count = [0]
                def backward(gradient):
                    count[0] += 1
                    if count[0] == 2 and dist.get_rank() == 1:
                        assert all(int(item['step']) == 2 for item in trainer.opt_d.state.values())
                        assert all(int(item['step']) == 1 for item in trainer.opt_g.state.values())
                        (markers / 'fault-entered').write_text('g-backward')
                        raise RuntimeError('injected second G microbatch backward failure')
                    return gradient
                next(trainer.graph.models['generator'].parameters()).register_hook(backward)
            elif dist.get_rank() == 1:
                import hypergan.distributed_training as numerical
                def fail_ema(*args, **kwargs):
                    assert all(int(item['step']) == 2 for item in trainer.opt_d.state.values())
                    assert all(int(item['step']) == 2 for item in trainer.opt_g.state.values())
                    (markers / 'fault-entered').write_text('ema')
                    raise RuntimeError('injected EMA failure after G optimizer')
                numerical.update_ema = fail_ema
        try:
            return original(state, operation, payload)
        except BaseException:
            if mode in ('g-backward', 'ema') and operation == 'update':
                assert trainer.step == 1 and trainer._poisoned and not trainer.checkpoint_ready
                (markers / f'poisoned-rank-{dist.get_rank()}').write_text('true')
            raise
    worker.handle_command = handle
'''

MODELS = _base.MODELS.replace('        super().__init__(**kwargs)',
                             '        super().__init__(**kwargs)\n        install_faults()', 1) + FAULTS

DRIVER = '''
import json
import os
from pathlib import Path
import sys
from hypergan import run_controller
from hypergan.replicated_execution import run_train, run_resume
from hypergan.run_state import run_lock


def reaped(markers):
    assert len(list(markers.glob('rank-*.pid'))) == 2
    for pid in set(map(int, (markers / 'all-ranks.txt').read_text().splitlines())):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError('terminal publication before rank reaping: ' + str(pid))


if __name__ == '__main__':
    config, run, markers, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    markers.mkdir(exist_ok=True)
    os.environ['HG_ACCEPTANCE_MARKERS'] = str(markers)
    os.environ['HG_ACCEPTANCE_MODE'] = mode
    assert 'torch' not in sys.modules
    original_publish = run_controller.atomic_json
    def audited_publish(path, value):
        if Path(path).name == 'manifest.json' and value.get('status') in ('complete', 'stopped', 'failed'):
            reaped(markers)
        return original_publish(path, value)
    run_controller.atomic_json = audited_publish
    if mode in ('rename', 'pointer', 'after-pointer'):
        import hypergan.distributed_commit as commit
        original_rename, original_pointer = Path.rename, commit.atomic_json
        def rename(path, target):
            if mode == 'rename' and '-step-00000002-' in Path(target).name:
                (markers / 'fault-entered').write_text(mode)
                raise OSError('injected canonical rename failure')
            return original_rename(path, target)
        def pointer(path, value):
            if Path(path).name == 'latest.json' and value['step'] == 2:
                if mode == 'after-pointer':
                    original_pointer(path, value)
                (markers / 'fault-entered').write_text(mode)
                raise OSError('injected canonical pointer failure ' + mode)
            return original_pointer(path, value)
        if mode == 'rename':
            Path.rename = rename
        else:
            commit.atomic_json = pointer
    policy = {'startup_timeout': 40, 'command_timeout': 25, 'collective_timeout': 15, 'total_timeout': 120}
    try:
        if mode == 'resume':
            result = run_resume(run, service_policy=policy)
        else:
            result = run_train(config, run,
                profile={'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2}},
                checkpoint_every=1, service_policy=policy)
    except (ValueError, RuntimeError, OSError) as error:
        if mode in ('full', 'resume'):
            raise
        result = json.loads((run / 'manifest.json').read_text())
        assert result['status'] == 'failed' and 'injected' in str(error)
        assert result['last_durable_step'] == 1
        expected = 1 if mode in ('g-backward', 'ema') else 2
        assert result['steps'] == expected and result['possible_lost_steps'] == expected - 1
        assert not result.get('bundle_path') and not result.get('sample_path')
        assert (markers / 'fault-entered').read_text() == mode
    else:
        assert mode in ('full', 'resume'), 'fault did not fail its attempt'
    reaped(markers)
    with run_lock(run):
        pass
    assert 'torch' not in sys.modules
    (markers / 'result.json').write_text(json.dumps(result))
'''


def _setup(tmp_path):
    driver, config = _base._setup(tmp_path)
    (tmp_path / 'job_fixture.py').write_text(MODELS, encoding='utf-8')
    driver.write_text(DRIVER, encoding='utf-8')
    return driver, config


@pytest.mark.parametrize('mode', ['stage', 'rename', 'pointer', 'after-pointer', 'g-backward', 'ema'])
def test_shared_controller_fault_boundary_and_exact_fresh_recovery(tmp_path, mode):
    driver, config = _setup(tmp_path)
    _base._run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    failed = _base._run(driver, config, tmp_path / 'fault', tmp_path / 'fault-pids', mode)
    pointer = json.loads((tmp_path / 'fault/distributed-checkpoints/latest.json').read_text())
    selected_step = 2 if mode == 'after-pointer' else 1
    assert pointer['step'] == selected_step
    selected = tmp_path / 'fault/distributed-checkpoints' / pointer['checkpoint']
    assert json.loads((selected / 'manifest.json').read_text())['step'] == selected_step
    assert all((selected / f'rank-{rank:05d}.pt').is_file() for rank in range(2))
    events = [json.loads(line) for line in (tmp_path / 'fault/events.jsonl').read_text().splitlines()]
    assert [row['step'] for row in events if row['event'] == 'checkpoint'] == [0, 1]
    assert [row['step'] for row in events if row['event'] == 'train'] == list(range(1, failed['steps'] + 1))
    assert events[-1]['event'] == 'failed'
    if mode in ('g-backward', 'ema'):
        assert all((tmp_path / f'fault-pids/poisoned-rank-{rank}').read_text() == 'true' for rank in range(2))
    # Post-replace exception leaves a valid newer selection despite no acknowledged
    # checkpoint event. Takeover must use actual durable selection, never roll it back.
    recovered = _base._run(driver, config, tmp_path / 'fault', tmp_path / 'resume-pids', 'resume')
    assert recovered['status'] == 'complete' and recovered['attempt_index'] == 2
    assert recovered['attempt_id'] != failed['attempt_id']
    resumed_events = [json.loads(line) for line in (tmp_path / 'fault/events.jsonl').read_text().splitlines()
                      if json.loads(line)['attempt_id'] == recovered['attempt_id']]
    assert [row['step'] for row in resumed_events if row['event'] == 'train'] == list(range(selected_step + 1, 5))
    _base._artifacts(recovered)
    for actual, expected in zip(_base._states(tmp_path / 'fault'), _base._states(tmp_path / 'full')):
        _base._same(actual, expected)
