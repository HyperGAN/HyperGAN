"""Public routing rejects numerical/option conflicts before viewer or run writes."""
from copy import deepcopy
import json
import builtins

import pytest

from hypergan.config import config_values, fingerprint, load_config, write_default
from hypergan.execution import prepare_resume, prepare_train
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.run_state import EventJournal


def project(tmp_path, device='cpu'):
    return write_default(tmp_path / 'project', device=device)


def stopped(tmp_path, *, replicated=True):
    path = project(tmp_path)
    config = load_config(path)
    run = tmp_path / 'run'
    run.mkdir()
    manifest = {'schema_version': 1, 'run_id': 'fixture', 'config': config_values(config),
                'config_sha256': fingerprint(config), 'next_sample_sequence': 2,
                'checkpoint_every': 7, 'preview_every': 0, 'preview_keep': 3}
    identity = {'config_sha256': fingerprint(config)}
    if replicated:
        execution = resolve_execution_profile({'schema_version': 1, 'execution': {
            'name': 'cpu-replicated-gloo', 'accumulation_steps': 2}}, config)['execution']
        manifest['execution'] = execution
        identity['topology'] = execution
    root = run / ('distributed-checkpoints' if replicated else 'checkpoints')
    checkpoint = root / 'snapshot'
    checkpoint.mkdir(parents=True)
    journal = EventJournal(run)
    journal.append({'run_id': 'fixture', 'attempt_id': 'first', 'step': 0, 'sequence': 1, 'event': 'start'})
    info = {'schema_version': 1, 'run_id': 'fixture', 'attempt_id': 'first', 'step': 0,
            'event_boundary': journal.commit_boundary(),
            'kind': 'hypergan-distributed-training-checkpoint' if replicated else 'hypergan-training-checkpoint'}
    info.update({'identity': identity} if replicated else identity)
    (checkpoint / 'manifest.json').write_text(json.dumps(info))
    (root / 'latest.json').write_text(json.dumps({'schema_version': 1, 'checkpoint': 'snapshot', 'step': 0, 'kind': info['kind']}))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    return path, run, checkpoint, manifest


def test_native_default_preserves_configured_cuda(tmp_path):
    prepared = prepare_train(project(tmp_path, 'cuda'), tmp_path / 'run')
    assert prepared.config['training']['device'] == 'cuda'
    assert prepared.profile is None
    assert not prepared.run_dir.exists()


def test_explicit_cpu_and_replicated_profiles(tmp_path):
    path = project(tmp_path)
    native = prepare_train(path, tmp_path / 'native', profile='cpu-single')
    assert native.profile['execution']['name'] == 'cpu-single'
    replicated = prepare_train(path, tmp_path / 'distributed', profile='cpu-replicated-gloo',
                               service_policy={'command_timeout': 123})
    assert replicated.profile['execution']['world_size'] == 2
    assert replicated.service_policy['command_timeout'] == 123
    with pytest.raises(ValueError, match='training.device=cuda'):
        prepare_train(path, tmp_path / 'run', profile='cuda-replicated-nccl')
    with pytest.raises(ValueError, match='require a replicated'):
        prepare_train(path, tmp_path / 'run', service_policy={'command_timeout': 123})


def test_profile_file_and_deadline_validation(tmp_path):
    path = project(tmp_path)
    profile = tmp_path / 'execution.toml'
    profile.write_text('schema_version = 1\n[execution]\nname = "cpu-replicated-gloo"\naccumulation_steps = 2\n')
    prepared = prepare_train(path, tmp_path / 'run', profile=profile)
    assert prepared.profile['execution']['microbatch_size'] == 4
    with pytest.raises(ValueError, match='collective_timeout'):
        prepare_train(path, tmp_path / 'run', profile=profile, service_policy={'command_timeout': 1})
    with pytest.raises(ValueError, match='Unknown'):
        prepare_train(path, tmp_path / 'run', profile=profile, service_policy={'typo': 1})


def test_resume_infers_numerical_identity_and_accepts_deadlines(tmp_path):
    _, run, checkpoint, manifest = stopped(tmp_path)
    before = (run / 'manifest.json').read_bytes()
    prepared = prepare_resume(run, service_policy={'command_timeout': 120, 'total_timeout': 900})
    assert prepared.profile['execution'] == manifest['execution']
    assert prepared.checkpoint == checkpoint
    assert prepared.controls['checkpoint_every'] == 7
    assert prepared.service_policy['command_timeout'] == 120
    assert (run / 'manifest.json').read_bytes() == before
    assert not (run / 'attempts').exists()


def test_resume_native_and_explicit_cpu_identity(tmp_path):
    _, run, checkpoint, _ = stopped(tmp_path, replicated=False)
    assert prepare_resume(run).profile is None
    assert prepare_resume(run, profile='cpu-single').checkpoint == checkpoint
    with pytest.raises(ValueError, match='execution identity differs'):
        prepare_resume(run, profile='cpu-replicated-gloo')


@pytest.mark.parametrize('execution', [None, {}, {'name': 'unknown'}, [], 'cpu-single'])
def test_unknown_saved_execution_never_falls_back_to_native(tmp_path, execution):
    _, run, _, manifest = stopped(tmp_path)
    manifest['execution'] = execution
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='supported numerical execution'):
        prepare_resume(run)


def test_resume_rejects_topology_schedule_checkpoint_conflicts(tmp_path, monkeypatch):
    path, run, checkpoint, _ = stopped(tmp_path)
    with pytest.raises(ValueError, match='execution identity differs'):
        prepare_resume(run, profile='cpu-replicated-gloo')  # saved accumulation is 2
    config = load_config(path)
    changed = deepcopy(config_values(config))
    changed['training']['steps'] += 1
    from hypergan.config import resolve_config
    import hypergan.execution as execution
    with monkeypatch.context() as scoped:
        scoped.setattr(execution, 'load_config', lambda _: resolve_config(changed))
        with pytest.raises(ValueError, match='configuration differs'):
            prepare_resume(run, config_path=path)
    info = json.loads((checkpoint / 'manifest.json').read_text())
    info['kind'] = 'ema-inference'
    (checkpoint / 'manifest.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='checkpoint kind'):
        prepare_resume(run)
    assert not (run / 'attempts').exists()


def test_external_checkpoint_rejected(tmp_path):
    _, run, _, _ = stopped(tmp_path)
    other = tmp_path / 'foreign'
    other.mkdir()
    with pytest.raises(ValueError, match='inside this run'):
        prepare_resume(run, checkpoint=other)


def test_cli_conflicts_precede_viewer_and_numerical_imports(tmp_path, monkeypatch, capsys):
    from hypergan import cli
    path = project(tmp_path)
    def viewer(_):
        pytest.fail('viewer must not start on conflicting options')
    monkeypatch.setattr(cli, '_training_viewer', viewer)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'particlegan'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert cli.main(['train', str(path), '--run-dir', str(tmp_path / 'run'),
                     '--profile', 'cuda-replicated-nccl', '--server']) == 1
    assert 'training.device=cuda' in capsys.readouterr().err
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('value', [True, 2.0])
def test_saved_execution_uses_strict_json_types(tmp_path, value):
    _, run, _, manifest = stopped(tmp_path)
    manifest['execution']['world_size'] = value
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='positive integer'):
        prepare_resume(run)


def test_checkpoint_topology_and_step_are_strict(tmp_path):
    _, run, checkpoint, _ = stopped(tmp_path)
    path = checkpoint / 'manifest.json'
    info = json.loads(path.read_text())
    info['identity']['topology']['world_size'] = 2.0
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='execution identity differs'):
        prepare_resume(run)
    info['identity']['topology']['world_size'] = 2
    info['step'] = True
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='Checkpoint step'):
        prepare_resume(run)


@pytest.mark.parametrize('saved_config', [{}, {'training': None}, 'invalid'])
def test_explicit_config_supplies_validated_checkpoint_schedule(tmp_path, saved_config):
    config_path, run, checkpoint, manifest = stopped(tmp_path)
    manifest['config'] = saved_config
    (run / 'manifest.json').write_text(json.dumps(manifest))
    prepared = prepare_resume(run, config_path=config_path)
    assert prepared.checkpoint == checkpoint
    info_path = checkpoint / 'manifest.json'
    info = json.loads(info_path.read_text())
    info['step'] = prepared.config['training']['steps'] + 1
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='outside the original schedule'):
        prepare_resume(run, config_path=config_path)


@pytest.mark.parametrize('replicated', [False, True], ids=['native', 'replicated'])
def test_repeated_train_selects_latest_and_inherits_saved_controls(tmp_path, replicated):
    path, run, checkpoint, manifest = stopped(tmp_path, replicated=replicated)
    manifest.update(preview_every=2, preview_keep=5)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    prepared = prepare_train(path, run)
    assert prepared.operation == 'resume'
    assert prepared.checkpoint == checkpoint
    assert prepared.controls == dict(checkpoint_every=7, max_seconds=None, stop_after_steps=None,
                                     preview_every=2, preview_keep=5, preview_name='g')
    if replicated:
        assert prepared.profile['execution'] == manifest['execution']
    else:
        assert prepared.profile is None
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_repeated_train_allows_explicit_attempt_controls(tmp_path):
    path, run, checkpoint, manifest = stopped(tmp_path)
    manifest['preview_every'] = 2
    (run / 'manifest.json').write_text(json.dumps(manifest))
    prepared = prepare_train(path, run, checkpoint_every=2, preview_every=0,
                             preview_keep=5, preview_name='ema', stop_after_steps=1,
                             service_policy={'command_timeout': 120})
    assert prepared.checkpoint == checkpoint
    assert prepared.controls['checkpoint_every'] == 2
    assert prepared.controls['preview_every'] == 0
    assert prepared.controls['preview_keep'] == 5
    assert prepared.controls['preview_name'] == 'ema'
    assert prepared.controls['stop_after_steps'] == 1
    assert prepared.service_policy['command_timeout'] == 120


def test_repeated_train_effective_steps_must_match_original_schedule(tmp_path):
    from hypergan.config import resolve_config
    path, run, checkpoint, manifest = stopped(tmp_path, replicated=False)
    values = config_values(load_config(path))
    values['training']['steps'] = 9
    config_hash = fingerprint(resolve_config(values))
    manifest.update(config=values, config_sha256=config_hash)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    info = json.loads((checkpoint / 'manifest.json').read_text())
    info['config_sha256'] = config_hash
    (checkpoint / 'manifest.json').write_text(json.dumps(info))
    prepared = prepare_train(path, run, steps=9)
    assert prepared.operation == 'resume'
    assert prepared.config['training']['steps'] == 9
    for steps in (None, 8, 10):
        with pytest.raises(ValueError, match='configuration differs'):
            prepare_train(path, run, steps=steps)


def test_train_control_defaults_distinguish_omission_from_disable(tmp_path):
    from hypergan.cli import _parser
    args = _parser().parse_args(['train', 'config', '--run-dir', 'run'])
    assert args.checkpoint_every is args.preview_every is args.preview_keep is None
    assert args.preview_name is None
    disabled = _parser().parse_args(['train', 'config', '--run-dir', 'run', '--no-previews'])
    assert disabled.preview_every == 0
    prepared = prepare_train(project(tmp_path), tmp_path / 'run')
    assert prepared.controls['checkpoint_every'] == 100
    assert prepared.controls['preview_every'] == 0
    # A history slider needs more than a handful of retained previews.
    assert prepared.controls['preview_keep'] == 20
    assert prepared.controls['preview_name'] == 'g'


@pytest.mark.parametrize('conflict', ['config', 'metrics', 'schedule', 'profile', 'missing-checkpoint'])
def test_repeated_train_conflicts_precede_viewer_output_and_numerical_imports(
        tmp_path, monkeypatch, capsys, conflict):
    from hypergan import cli
    path, run, _, _ = stopped(tmp_path)
    extra = []
    if conflict == 'config':
        path.write_text(path.read_text().replace('steps = 5', 'steps = 6'))
    elif conflict == 'metrics':
        path.write_text(path.read_text() + '\n[metrics]\npreset = "none"\n')
    elif conflict == 'schedule':
        extra = ['--steps', '6']
    elif conflict == 'profile':
        extra = ['--profile', 'cpu-single']
    else:
        (run / 'distributed-checkpoints' / 'latest.json').unlink()
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(cli, '_training_viewer', lambda _: pytest.fail('viewer started on conflict'))
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'particlegan'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert cli.main(['train', str(path), '--run-dir', str(run), '--server',
                     '--progress-every', '2', *extra]) == 1
    error = capsys.readouterr().err
    assert 'error:' in error
    assert ('configuration differs' if conflict in {'config', 'metrics', 'schedule'} else
            'execution identity differs' if conflict == 'profile' else
            'No full training checkpoint') in error
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['empty-directory', 'file'])
def test_repeated_train_rejects_existing_paths_without_run_metadata(tmp_path, kind):
    path = project(tmp_path)
    run = tmp_path / 'run'
    if kind == 'file':
        run.write_bytes(b'preserve me')
    else:
        run.mkdir()
    with pytest.raises((OSError, ValueError)):
        prepare_train(path, run)
    if kind == 'file':
        assert run.read_bytes() == b'preserve me'
    else:
        assert list(run.iterdir()) == []


def test_repeated_train_revalidates_changed_config_before_dispatch(tmp_path):
    path, run, _, _ = stopped(tmp_path, replicated=False)
    prepared = prepare_train(path, run)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    path.write_text(path.read_text().replace('steps = 5', 'steps = 6'))
    with pytest.raises(ValueError, match='configuration differs'):
        prepared.run()
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_repeated_train_metrics_are_strict_but_explicit_resume_can_change_them(tmp_path):
    path, run, checkpoint, _ = stopped(tmp_path, replicated=False)
    path.write_text(path.read_text() + '\n[metrics]\npreset = "none"\n')
    with pytest.raises(ValueError, match='configuration differs'):
        prepare_train(path, run)
    prepared = prepare_resume(run, config_path=path)
    assert prepared.checkpoint == checkpoint
    assert prepared.config['metrics']['preset'] == 'none'


SNAPSHOT_METRIC = '''
[metrics.custom.fid]
factory = "hypergan.metric_examples:ColorMomentDistance"
mode = "snapshot"
{trigger}inputs = {{ generated = "evaluation.generated", reference = "evaluation.reference" }}
[metrics.custom.fid.evaluation]
device = "cpu"
sample_count = 8
batch_size = 4
seed = 5
[metrics.custom.fid.evaluation.data]
factory = "gaussian_grid"
args = {{ side = 4 }}
'''


def test_manual_run_keeps_its_recorded_schedule_when_the_default_changes(tmp_path):
    """A run recorded with trigger="manual" resumes under its own resolved config."""
    path, run, checkpoint, _ = stopped(tmp_path, replicated=False)
    path.write_text(path.read_text() + SNAPSHOT_METRIC.format(trigger='trigger = "manual"\n'))
    manifest = json.loads((run / 'manifest.json').read_text())
    config = load_config(path)
    assert config['metrics']['custom']['fid']['trigger'] == 'manual'
    manifest['config'] = config_values(config)
    manifest['config_sha256'] = fingerprint(config)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    info = json.loads((checkpoint / 'manifest.json').read_text())
    info['config_sha256'] = manifest['config_sha256']
    (checkpoint / 'manifest.json').write_text(json.dumps(info))
    prepared = prepare_train(path, run)
    assert prepared.operation == 'resume'
    assert prepared.config['metrics']['custom']['fid']['trigger'] == 'manual'
    # Dropping the explicit opt-out is a real schedule change, refused by name.
    path.write_text(path.read_text().replace('trigger = "manual"\n', ''))
    with pytest.raises(ValueError, match='differs from the original run in metrics'):
        prepare_train(path, run)
    prepared = prepare_resume(run, config_path=path)
    assert prepared.config['metrics']['custom']['fid']['trigger'] == 'interval'
    assert prepared.config['metrics']['custom']['fid']['every_steps'] == 10000


def test_repeated_train_accepts_equivalent_config_from_another_filename(tmp_path):
    path, run, checkpoint, _ = stopped(tmp_path, replicated=False)
    copied = tmp_path / 'renamed.toml'
    copied.write_text('# Only comments and the file location changed.\n' + path.read_text())
    prepared = prepare_train(copied, run)
    assert prepared.operation == 'resume' and prepared.checkpoint == checkpoint


def test_prepared_new_train_cannot_silently_switch_to_existing_run(tmp_path):
    path = project(tmp_path)
    prepared = prepare_train(path, tmp_path / 'run')
    # Another invocation creates the run while the prepared caller is opening
    # its viewer. The first caller must not silently take over that invocation.
    path.unlink()
    _, run, _, _ = stopped(tmp_path, replicated=False)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    with pytest.raises((OSError, ValueError)):
        prepared.run()
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('replicated', [False, True], ids=['native', 'replicated'])
@pytest.mark.parametrize('version', [0, 2, -1, True, 1.0, '1', None, {}])
def test_checkpoint_compatibility_rejects_before_viewer_and_imports(
        tmp_path, monkeypatch, capsys, version, replicated):
    from hypergan import cli
    path, run, checkpoint, _ = stopped(tmp_path, replicated=replicated)
    metadata_path = checkpoint / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    identity = metadata['identity'] if replicated else metadata
    identity['hypergan_checkpoint_version'] = version
    metadata_path.write_text(json.dumps(metadata))
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(cli, '_training_viewer', lambda _: pytest.fail('viewer started for incompatible checkpoint'))
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'particlegan'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    for args in (['train', str(path), '--run-dir', str(run)], ['resume', str(run)]):
        assert cli.main([*args, '--server']) == 1
        assert 'checkpoint compatibility version' in capsys.readouterr().err
        assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('replicated', [False, True], ids=['native', 'replicated'])
@pytest.mark.parametrize('explicit', [False, True], ids=['existing-unversioned', 'current-version'])
def test_checkpoint_compatibility_accepts_supported_version(tmp_path, explicit, replicated):
    path, run, checkpoint, _ = stopped(tmp_path, replicated=replicated)
    metadata_path = checkpoint / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    if explicit:
        identity = metadata['identity'] if replicated else metadata
        identity['hypergan_checkpoint_version'] = 1
        metadata_path.write_text(json.dumps(metadata))
    assert prepare_resume(run).checkpoint == checkpoint
    assert prepare_train(path, run).checkpoint == checkpoint
