"""The run records its retention and launch tables, numbers attempts once, and
never re-saves unchanged trainer state, across resumed attempts (torch-free)."""
import json
from pathlib import Path

import pytest

from hypergan import cli
from hypergan.checkpoints import resolve_checkpoint
from hypergan.config import config_values, load_config, resume_compatible
from hypergan.model_file import normalize_checkpoint_policy
from hypergan.provenance import determine_attempt, write_launch_provenance
from hypergan.run_controller import (
    CompletedUpdate, ExecutionInfo, PreviewResult, Restored, run_resume, run_train,
)
from hypergan.summary import read_summary_json


POLICY = {'keep_last': 2, 'keep_every': 0,
          'keep_best': {'metric': 'score', 'mode': 'max', 'count': 1}}


class CheckpointingExecution:
    """Writes controller checkpoints under checkpoints/ like the native adapter."""
    inference_available = False

    def __init__(self, config, previews=False):
        self.step, self.count, self.previews = 0, 0, previews

    @staticmethod
    def environment():
        return {'runtime': {'runtime_checked': False}, 'source': {}}

    def start(self):
        return ExecutionInfo(self.step, {'dataset': 'fixture'}, [], {})

    def restore(self, run_dir, checkpoint, run_id, config_sha256):
        target = Path(checkpoint or json.loads((run_dir / 'manifest.json').read_text())['checkpoint_path'])
        self.step = json.loads((target / 'manifest.json').read_text())['step']
        return Restored(target, self.step)

    def update(self):
        self.step += 1
        return CompletedUpdate(self.step, {'loss': 0.25})

    def checkpoint(self, run_dir, metadata):
        self.count += 1
        path = run_dir / 'checkpoints' / f"{metadata['attempt_id']}-step-{self.step:08d}-{self.count:012x}"
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text(json.dumps(dict(metadata, step=self.step)))
        (path.parent / 'latest.json').write_text(json.dumps({'checkpoint': path.name, 'step': self.step}))
        return path

    def preview(self, run_dir, identity, *, keep):
        if not self.previews:
            raise AssertionError('previews are off')
        from hypergan.previews import publish_preview_payload
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity,
                   'name': identity['name'], 'step': self.step, 'count': 1,
                   'shape': [1, 1], 'samples': [[0.5]]}
        return PreviewResult(*publish_preview_payload(run_dir, payload, identity, self.step, keep=keep))

    def observe(self, callback, event):
        callback(event)

    def shutdown(self):
        pass


def write_receipt(run_dir, metric, step, value, owner):
    evaluation = f'{metric}-{step:08x}'
    directory = Path(run_dir) / 'metrics' / 'evaluations' / f'step-{step:08d}-{evaluation}'
    directory.mkdir(parents=True)
    (directory / 'receipt.json').write_text(json.dumps({
        'schema_version': 1, 'status': 'complete', 'evaluation_id': evaluation,
        'metric_id': metric, 'source_step': step,
        'run_id': owner['run_id'], 'attempt_id': owner['attempt_id'], 'result': {'value': value, 'step': step}}))
    (directory / 'stream.json').write_text('{}')  # Registered: recovery leaves it alone.


def manifest(run_dir):
    return json.loads((Path(run_dir) / 'manifest.json').read_text())


def test_configured_keep_best_applies_after_a_resume_from_the_manifest(tmp_path):
    # Review 2026-10-25 #9 / H1: the manifest did not record [checkpoints], so
    # a resumed attempt, summary.best and --checkpoint best used the default policy.
    root = tmp_path / 'run'
    first = run_train({'training': {'steps': 4}, 'checkpoints': POLICY}, root, checkpoint_every=1,
                      stop_after_steps=2, execution_factory=CheckpointingExecution)
    assert first['config']['checkpoints'] == normalize_checkpoint_policy(POLICY)
    # "score" declares no direction, so only keep_best.mode can rank it.
    write_receipt(root, 'score', 1, 9.0, first)
    write_receipt(root, 'score', 2, 1.0, first)
    # Resume without a configuration: the policy comes from the manifest alone.
    second = run_resume(root, execution_factory=CheckpointingExecution)
    assert second['status'] == 'complete' and second['attempt_index'] == 2
    assert manifest(root)['config']['checkpoints'] == normalize_checkpoint_policy(POLICY)
    best = resolve_checkpoint(root, 'best')
    assert json.loads((best / 'manifest.json').read_text())['step'] == 1
    summary = read_summary_json(root)
    assert (summary['best']['metric'], summary['best']['step']) == ('score', 1)
    # The resumed attempt pruned with this policy: step zero, the best (1), the
    # recovery parent (2) and the last two. The default would have dropped step 1.
    kept = sorted(json.loads((path / 'manifest.json').read_text())['step']
                  for path in (root / 'checkpoints').iterdir() if path.is_dir())
    assert kept == [0, 1, 2, 3, 4]


def test_retention_policy_is_observation_not_numerical_identity():
    base = load_config({'training': {'steps': 4}})
    configured = load_config({'training': {'steps': 4}, 'checkpoints': POLICY})
    defaults = load_config({'training': {'steps': 4}, 'checkpoints': {}})
    # Never part of the numerical comparison.
    assert resume_compatible(configured, config_values(base))
    # A run that recorded no policy adopts the configured one.
    assert resume_compatible(configured, config_values(base), include_observation=True)
    # A recorded policy is compared like metrics; omitting the table means the defaults.
    changed = resume_compatible(base, config_values(configured), include_observation=True)
    assert not changed and changed.differences == ['checkpoints.keep_best.count', 'checkpoints.keep_best.metric',
                                                   'checkpoints.keep_best.mode', 'checkpoints.keep_every',
                                                   'checkpoints.keep_last']
    assert resume_compatible(base, config_values(defaults), include_observation=True)


def test_repeat_train_reports_a_retention_change_as_an_observation_change(tmp_path, monkeypatch):
    from hypergan import execution
    from hypergan.model_file import ConfigError
    monkeypatch.setattr(execution, '_checkpoint', lambda *args: None)  # This fixture's checkpoints are not native.
    root = tmp_path / 'run'
    run_train({'training': {'steps': 2}, 'checkpoints': POLICY}, root, execution_factory=CheckpointingExecution)
    with pytest.raises(ConfigError, match=r'differs from the original run in checkpoints; `hypergan resume'):
        execution.prepare_train({'training': {'steps': 2}, 'checkpoints': {'keep_last': 5}}, root)
    assert execution.prepare_train({'training': {'steps': 2}, 'checkpoints': POLICY}, root).operation == 'resume'
    assert execution.prepare_resume(root, config_path={'training': {'steps': 2}, 'checkpoints': {'keep_last': 5}})
    # A run recorded before its policy was continues with the configured one.
    legacy = manifest(root)
    del legacy['config']['checkpoints']
    (root / 'manifest.json').write_text(json.dumps(legacy))
    assert execution.prepare_train({'training': {'steps': 2}, 'checkpoints': {'keep_last': 5}}, root).operation == 'resume'


def test_steps_override_keeps_retention_and_launch_tables(tmp_path):
    # A --steps override re-resolved config_values alone, dropping [run] (and,
    # before retention was an observation value, [checkpoints]).
    root = tmp_path / 'run'
    config = {'training': {'steps': 9}, 'checkpoints': POLICY, 'run': {'env': 'fixture-env'}}
    first = run_train(config, root, 3, stop_after_steps=1, execution_factory=CheckpointingExecution)
    assert first['total_steps'] == 3
    assert first['config']['checkpoints'] == normalize_checkpoint_policy(POLICY)
    launch = json.loads((root / 'launch' / '0001.json').read_text())
    assert launch['run'] == {'env': 'fixture-env'}
    second = run_resume(root, config_path={**config, 'training': {'steps': 3}}, steps=3,
                        execution_factory=CheckpointingExecution)
    assert second['status'] == 'complete' and second['total_steps'] == 3
    assert second['config']['checkpoints'] == normalize_checkpoint_policy(POLICY)
    launch = json.loads((root / 'launch' / '0002.json').read_text())
    assert launch['run'] == {'env': 'fixture-env'}


def test_reexecuted_resume_keeps_the_attempt_number_its_parent_reserved(tmp_path, monkeypatch):
    # Notes 02 §3: resume popped _HYPERGAN_REEXEC_ATTEMPT before reading it, so
    # the child wrote a second launch record under the next number.
    root = tmp_path / 'run'
    run_train({'training': {'steps': 3}}, root, stop_after_steps=1, execution_factory=CheckpointingExecution)
    reserved = cli._determine_attempt(root)
    assert reserved == determine_attempt(root) == 2
    write_launch_provenance(root, attempt=reserved, metadata={'executable': 'parent-python'})
    monkeypatch.setenv('_HYPERGAN_REEXEC', '1')
    monkeypatch.setenv('_HYPERGAN_REEXEC_ATTEMPT', f'{reserved:04d}')
    second = run_resume(root, stop_after_steps=1, execution_factory=CheckpointingExecution)
    assert second['attempt_index'] == 2 and second['attempt_id'].startswith('0002-')
    assert sorted(path.name for path in (root / 'launch').iterdir()) == ['0001.json', '0002.json']
    assert json.loads((root / 'launch' / '0002.json').read_text())['executable'] == 'parent-python'
    import os
    assert '_HYPERGAN_REEXEC_ATTEMPT' not in os.environ and '_HYPERGAN_REEXEC' not in os.environ


def test_attempt_and_launch_numbers_cannot_diverge(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    run_train({'training': {'steps': 4}}, root, stop_after_steps=1, execution_factory=CheckpointingExecution)
    # A run recorded before launch provenance existed has attempts but no launch records.
    for path in (root / 'launch').iterdir():
        path.unlink()
    second = run_resume(root, stop_after_steps=1, execution_factory=CheckpointingExecution)
    assert second['attempt_index'] == 2
    assert sorted(path.name for path in (root / 'launch').iterdir()) == ['0002.json']
    # A stale inherited number that an attempt already used is ignored.
    monkeypatch.setenv('_HYPERGAN_REEXEC_ATTEMPT', '0002')
    third = run_resume(root, stop_after_steps=1, execution_factory=CheckpointingExecution)
    assert third['attempt_index'] == 3
    assert sorted(path.name for path in (root / 'launch').iterdir()) == ['0002.json', '0003.json']
    attempts = sorted(int(path.name.split('-')[0]) for path in (root / 'attempts').iterdir())
    assert attempts == [1, 2, 3]


def test_publications_after_the_last_checkpoint_do_not_rewrite_it(tmp_path):
    # Notes 02 §4: a preview after the final periodic checkpoint re-saved the
    # unchanged trainer state as a second checkpoint at the same step.
    root = tmp_path / 'run'
    factory = lambda config: CheckpointingExecution(config, previews=True)
    result = run_train({'training': {'steps': 2}, 'checkpoints': {'keep_last': 9}}, root,
                       checkpoint_every=2, preview_every=2, execution_factory=factory)
    assert result['status'] == 'complete' and result['preview_count'] == 1
    steps = sorted(json.loads((path / 'manifest.json').read_text())['step']
                   for path in (root / 'checkpoints').iterdir() if path.is_dir())
    assert steps == [0, 2]
    saved = json.loads((Path(result['checkpoint_path']) / 'manifest.json').read_text())
    # The run's durable event frontier still covers the preview published after it.
    boundary = result['durable_event_boundary']
    assert boundary['sequence'] > saved['event_boundary']['sequence'] and boundary['step'] == 2
    events = (root / 'events.jsonl').read_bytes()[:boundary['offset']].decode().splitlines()
    assert any(json.loads(line)['event'] == 'preview' for line in events)
    # The checkpoint's own boundary remains a valid committed prefix for resume.
    resumed = run_resume(root, execution_factory=factory)
    assert resumed['status'] == 'complete' and resumed['recovery_parent']['step'] == 2
