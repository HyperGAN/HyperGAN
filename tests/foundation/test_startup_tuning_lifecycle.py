"""Startup calibration belongs before the first checkpoint, never in resume."""
import json

import pytest

from hypergan.execution import prepare_train
from hypergan.run_controller import run_train, run_resume
from .test_controller_fences import NativeExecution, setup
from .test_public_execution import project, stopped


def test_tune_options_validate_before_any_run_write(tmp_path):
    config = project(tmp_path)
    root = tmp_path / 'run'
    with pytest.raises(ValueError, match='single-process'):
        prepare_train(config, root, tune=True, profile='cpu-replicated-gloo')
    with pytest.raises(ValueError, match='boolean'):
        prepare_train(config, root, tune=1)
    assert not root.exists()
    assert prepare_train(config, root, tune=True).controls['tune'] is True
    assert 'tune_warmup_steps' not in prepare_train(config, root, tune=True).controls
    assert 'tune_warmup_steps' not in prepare_train(config, root).controls
    assert 'tune' not in prepare_train(config, root).controls


def test_repeat_train_with_tune_visibly_skips_search(tmp_path):
    path, root, checkpoint, _ = stopped(tmp_path, replicated=False)
    with pytest.warns(RuntimeWarning, match='tuning is skipped'):
        prepared = prepare_train(path, root, tune=True)
    assert prepared.operation == 'resume'
    assert prepared.checkpoint == checkpoint
    assert 'tune' not in prepared.controls


def test_tune_precedes_checkpoint_and_update_and_never_repeats_on_resume(tmp_path):
    path, root, _, options, trace, _ = setup(tmp_path, native=True)
    sequence = []
    events = []

    class Tunable(NativeExecution):
        def tune(self, run_dir, on_event=None):
            assert self.step == 0
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            assert manifest['status'] == 'tuning'
            assert manifest['initialization_tuning']['status'] == 'running'
            sequence.append('tune')
            on_event({'stage': 'measure', 'trial_step': 1, 'trial_steps': 8})
            return {'outcome': 'selected', 'selected_candidate': 'fixture'}

        def checkpoint(self, run_dir, metadata):
            sequence.append('checkpoint')
            assert metadata['initialization_tuning']['status'] == 'complete'
            return super().checkpoint(run_dir, metadata)

        def update(self):
            sequence.append('update')
            return super().update()

    factory = lambda config: Tunable(config, root, options, trace)
    result = run_train(path, root, steps=2, stop_after_steps=1, tune=True,
                       execution_factory=factory, on_event=events.append)
    assert sequence[:3] == ['tune', 'checkpoint', 'update']
    assert events[0]['event'] == 'start' and events[0]['sequence'] == 1
    assert sum(event['event'] == 'start' for event in events) == 1
    assert [e['tuning']['status'] for e in events if e['event'] == 'tuning'] == ['running', 'running', 'complete']
    assert result['initialization_tuning']['selected_candidate'] == 'fixture'
    assert run_resume(root, execution_factory=factory)['status'] == 'complete'
    assert sequence.count('tune') == 1


def test_failed_tuning_never_checkpoints_or_trains_partial_weights(tmp_path):
    path, root, _, options, trace, _ = setup(tmp_path, native=True)

    class Failing(NativeExecution):
        def tune(self, run_dir, on_event=None):
            raise ValueError('calibration failed')

        def checkpoint(self, *args, **kwargs):
            pytest.fail('partial initialization must not be checkpointed')

        def update(self):
            pytest.fail('partial initialization must not train')

    factory = lambda config: Failing(config, root, options, trace)
    with pytest.raises(ValueError, match='calibration failed'):
        run_train(path, root, tune=True, execution_factory=factory)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['initialization_tuning']['status'] == 'failed'
    assert manifest['last_durable_step'] is None
    assert not (root / 'fixture-checkpoints').exists()


@pytest.mark.parametrize('tune', [False, True])
@pytest.mark.parametrize('preview_every', [0, 100])
def test_initial_preview_follows_tuning_and_checkpoint_before_updates(tmp_path, tune, preview_every):
    path, root, _, options, trace, _ = setup(tmp_path, native=True)
    sequence = []

    class InitialPreview(NativeExecution):
        def tune(self, run_dir, on_event=None):
            sequence.append('tune')
            return {'outcome': 'selected'}

        def checkpoint(self, run_dir, metadata):
            sequence.append('checkpoint')
            return super().checkpoint(run_dir, metadata)

        def preview(self, run_dir, identity, *, keep):
            assert self.step == 0
            sequence.append('preview')

        def update(self):
            sequence.append('update')
            return super().update()

    factory = lambda config: InitialPreview(config, root, options, trace)
    run_train(path, root, steps=2, stop_after_steps=1, tune=tune,
              preview_every=preview_every, execution_factory=factory)
    expected = (['tune'] if tune else []) + ['checkpoint'] + (['preview'] if preview_every else []) + ['update']
    assert sequence[:len(expected)] == expected
    initial_preview_count = sequence.count('preview')
    # Restoring the saved step-zero checkpoint also must not repeat startup work.
    checkpoint = next((root / 'fixture-checkpoints').iterdir())
    run_resume(root, checkpoint=checkpoint, execution_factory=factory, stop_after_steps=1)
    assert sequence.count('preview') == initial_preview_count
    assert sequence.count('tune') == int(tune)


def test_unsupported_adapter_rejected_before_directory_creation(tmp_path):
    path, root, factory, _, _, _ = setup(tmp_path, native=True)
    with pytest.raises(ValueError, match='unsupported'):
        run_train(path, root, tune=True, execution_factory=factory)
    assert not root.exists()


def test_train_tuning_flags_are_explicit_and_mutually_exclusive():
    from hypergan.cli import _parser
    parser = _parser()
    arguments = ['train', 'config.toml', '--run-dir', 'run']
    assert parser.parse_args(arguments).tune is False
    assert parser.parse_args([*arguments, '--tune']).tune is True
    assert parser.parse_args([*arguments, '--no-tune']).tune is False
    with pytest.raises(SystemExit):
        parser.parse_args([*arguments, '--tune', '--tune-warmup-steps', '1000'])
    with pytest.raises(SystemExit):
        parser.parse_args([*arguments, '--tune', '--no-tune'])
