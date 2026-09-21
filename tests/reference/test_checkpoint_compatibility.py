"""Supported HyperGAN changes preserve full state while external identity stays strict."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import pytest

from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.execution import prepare_train, train
from hypergan.training import resume

from .test_recovery import equal


def files(run):
    return {path.relative_to(run): path.read_bytes() for path in run.rglob('*') if path.is_file()}


def old_hypergan_provenance(metadata):
    metadata['runtime']['hypergan'] = '0.0.0-previous-build'
    for name in metadata['implementation']:
        if name == 'hypergan' or name.startswith('hypergan.'):
            metadata['implementation'][name] = '0' * 64


@pytest.mark.parametrize('explicit', [True], ids=['current-version'])
def test_native_previous_hypergan_build_restores_exact_state_and_earlier_snapshot(tmp_path, explicit):
    config = write_default(tmp_path / 'config', device='cpu')
    baseline, run = tmp_path / 'baseline', tmp_path / 'run'
    train(config, baseline)
    stopped = train(config, run, stop_after_steps=2, checkpoint_every=1)
    checkpoint = Path(stopped['checkpoint_path'])
    metadata_path = checkpoint / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['hypergan_checkpoint_version'] == 2
    old_hypergan_provenance(metadata)
    if not explicit:
        metadata.pop('hypergan_checkpoint_version')
    metadata_path.write_text(json.dumps(metadata))
    historical = files(checkpoint)
    result = train(config, run)
    assert result['status'] == 'complete' and result['steps'] == 5
    equal(read_checkpoint(baseline)[2], read_checkpoint(run)[2])
    assert files(checkpoint) == historical
    # Selecting this preserved earlier snapshot remains valid after later
    # checkpoints have been written by the current HyperGAN build.
    replay = resume(run, checkpoint=checkpoint)
    assert replay['status'] == 'complete' and replay['attempt_index'] == 3
    equal(read_checkpoint(baseline)[2], read_checkpoint(run)[2])
    assert files(checkpoint) == historical


@pytest.mark.parametrize('kind', ['dependency', 'runtime', 'data', 'digest', 'implementation-container', 'runtime-container'])
def test_hypergan_provenance_differences_do_not_bypass_other_validation(tmp_path, kind):
    config = write_default(tmp_path / 'config', device='cpu')
    run = tmp_path / 'run'
    stopped = train(config, run, stop_after_steps=1)
    metadata_path = Path(stopped['checkpoint_path']) / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    old_hypergan_provenance(metadata)
    if kind == 'dependency':
        key = next(name for name in metadata['implementation'] if name.startswith('particlegan.'))
        metadata['implementation'][key] = 'f' * 64
        error = 'implementation'
    elif kind == 'runtime':
        metadata['runtime']['torch'] = 'different-pytorch'
        error = 'runtime'
    elif kind == 'data':
        metadata['data_contract']['supported'] = False
        error = 'data identity'
    elif kind == 'digest':
        metadata['state_sha256'] = '0' * 64
        error = 'digest mismatch'
    elif kind == 'implementation-container':
        metadata['implementation'] = ['hypergan.training']
        error = 'implementation'
    else:
        metadata['runtime'] = ['hypergan']
        error = 'runtime'
    metadata_path.write_text(json.dumps(metadata))
    before = files(run)
    with pytest.raises(ValueError, match=error):
        train(config, run)
    assert files(run) == before


def test_changed_external_factory_source_still_rejects_resume(tmp_path, monkeypatch):
    module_name = 'compatibility_external_generator'
    source = tmp_path / f'{module_name}.py'
    source.write_text('from hypergan.recipes import MLP\n\ndef make(**kwargs):\n    return MLP(**kwargs)\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text().replace('factory = "hndl"', f'factory = "{module_name}:make"', 1))
    run = tmp_path / 'run'
    train(config, run, stop_after_steps=1)
    source.write_text(source.read_text() + '\n# external factory source changed\n')
    before = files(run)
    with pytest.raises(ValueError, match='implementation differs'):
        train(config, run)
    assert files(run) == before


FIRST_CARD = '548116b7-9dbe-de58-b3d9-a6e27b0f74ce'
SECOND_CARD = 'ed080e41-3193-3755-6756-f3d46c433331'


def one_of_two_identical_cards(monkeypatch, selected):
    """Record a synthetic pair of identical GPUs so `selected` names the current one.

    Both cards report the same model and capability; only the physical identity
    moves, which is what an unpinned CUDA enumeration order does on a restart.
    """
    import hypergan.single_execution as single
    from hypergan.training import runtime_info as original

    def runtime_info(device='cpu'):
        return dict(original(device),
                    cuda={'version': '13.0', 'cudnn': 91200, 'name': 'NVIDIA RTX A6000',
                          'capability': [8, 6], 'uuid': selected[0],
                          'visible_devices': [selected[0]],
                          'cudnn_benchmark': False, 'cudnn_deterministic': True,
                          'cublas_workspace_config': ':4096:8',
                          'deterministic_warn_only': False, 'matmul_precision': 'highest',
                          'matmul_allow_tf32': False, 'cudnn_allow_tf32': False})

    monkeypatch.setattr(single, 'runtime_info', runtime_info)


def test_resume_on_the_other_identical_card_warns_and_completes(tmp_path, monkeypatch):
    config = write_default(tmp_path / 'config', device='cpu')
    baseline, run = tmp_path / 'baseline', tmp_path / 'run'
    selected = [FIRST_CARD]
    one_of_two_identical_cards(monkeypatch, selected)
    train(config, baseline)
    train(config, run, stop_after_steps=2, checkpoint_every=1)
    # The restart enumerates the other identical card as device 0.
    selected[0] = SECOND_CARD
    with pytest.warns(RuntimeWarning, match='different physical GPU'):
        result = train(config, run)
    assert result['status'] == 'complete' and result['steps'] == 5
    # Resuming onto an equivalent card is a warning, not a different result.
    equal(read_checkpoint(baseline)[2], read_checkpoint(run)[2])
    recorded = [warning for warning in result['warnings'] if 'different physical GPU' in warning]
    assert len(recorded) == 1 and FIRST_CARD in recorded[0] and SECOND_CARD in recorded[0]
    assert result['resume_warnings'] == recorded
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    resumed = next(event for event in events if event['event'] == 'resume')
    assert resumed['warnings'] == recorded


def test_resume_on_a_different_card_model_names_the_rejected_fields(tmp_path, monkeypatch):
    config = write_default(tmp_path / 'config', device='cpu')
    run = tmp_path / 'run'
    selected = [FIRST_CARD]
    one_of_two_identical_cards(monkeypatch, selected)
    train(config, run, stop_after_steps=1)
    import hypergan.single_execution as single
    replaced = single.runtime_info

    def other_model(device='cpu'):
        runtime = replaced(device)
        runtime['cuda'] = dict(runtime['cuda'], uuid=SECOND_CARD, visible_devices=[SECOND_CARD],
                               name='NVIDIA GeForce RTX 4090', capability=[8, 9])
        return runtime

    monkeypatch.setattr(single, 'runtime_info', other_model)
    before = files(run)
    with pytest.raises(ValueError) as error:
        train(config, run)
    message = str(error.value)
    assert 'cuda.name: saved "NVIDIA RTX A6000", current "NVIDIA GeForce RTX 4090"' in message
    assert 'cuda.capability: saved [8, 6], current [8, 9]' in message
    assert files(run) == before


@pytest.mark.parametrize('version', [1, 3, True])
def test_checkpoint_compatibility_rechecked_under_lock_before_numerical_loading(tmp_path, monkeypatch, version):
    import hypergan.run_controller as controller
    import hypergan.single_execution as single
    import torch
    config = write_default(tmp_path / 'config', device='cpu')
    run = tmp_path / 'run'
    stopped = train(config, run, stop_after_steps=1)
    metadata_path = Path(stopped['checkpoint_path']) / 'manifest.json'
    prepared = prepare_train(config, run)
    original_lock = controller.run_lock
    expected = None

    @contextmanager
    def change_after_public_preparation(run_dir):
        nonlocal expected
        with original_lock(run_dir):
            metadata = json.loads(metadata_path.read_text())
            metadata['hypergan_checkpoint_version'] = version
            metadata_path.write_text(json.dumps(metadata))
            expected = files(run)
            yield

    monkeypatch.setattr(controller, 'run_lock', change_after_public_preparation)
    monkeypatch.setattr(single, 'ReferenceTrainer',
                        lambda _: pytest.fail('trainer constructed before compatibility rejection'))
    monkeypatch.setattr(torch, 'load',
                        lambda *args, **kwargs: pytest.fail('tensor loading preceded compatibility rejection'))
    with pytest.raises(ValueError, match='checkpoint compatibility version'):
        prepared.run()
    assert expected is not None and files(run) == expected
