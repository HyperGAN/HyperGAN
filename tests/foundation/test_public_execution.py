"""Public routing rejects numerical/option conflicts before viewer or run writes."""
from copy import deepcopy
import json
import builtins

import pytest

from hypergan.config import config_values, fingerprint, load_config, write_default
from hypergan.execution import prepare_resume, prepare_train
from hypergan.execution_profiles import resolve_execution_profile


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
    info = {'schema_version': 1, 'run_id': 'fixture', 'step': 0,
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
