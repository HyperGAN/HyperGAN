"""Resume runtime validation names what differs and qualifies a card swap.

Two identical GPUs in one machine enumerate in an unstable order unless
CUDA_DEVICE_ORDER pins it, so a restart can restore onto the other physical
card. That changes which device holds the state, not what the state means, so
it warns and continues. Every other runtime difference stays a rejection, and a
rejection now names each differing key path with both values.
"""
import copy
import warnings

import pytest

from hypergan.checkpoint_compatibility import (
    DEVICE_IDENTITY_KEYS,
    HARDWARE_TOLERANCE_KEYS,
    HyperGANHardwareWarning,
    validate_runtime,
)


FIRST_CARD = '548116b7-9dbe-de58-b3d9-a6e27b0f74ce'
SECOND_CARD = 'ed080e41-3193-3755-6756-f3d46c433331'


def cuda_runtime(uuid=FIRST_CARD, visible=(FIRST_CARD, SECOND_CARD)):
    """The shape `hypergan.training.runtime_info` records for a CUDA device."""
    return {'python': '3.12.9', 'torch': '2.14.0+cu130', 'numpy': '2.2.1',
            'platform': 'Linux', 'machine': 'x86_64', 'threads': 1,
            'particlegan': '0.5.0', 'hypergan': '2.0.0a1', 'device': 'cuda:0',
            'dtype': 'float32', 'world_size': 1, 'default_dtype': 'torch.float32',
            'deterministic_algorithms': True,
            'cuda': {'version': '13.0', 'cudnn': 91200, 'name': 'NVIDIA RTX A6000',
                     'capability': [8, 6], 'uuid': uuid, 'visible_devices': list(visible),
                     'cudnn_benchmark': False, 'cudnn_deterministic': True,
                     'cublas_workspace_config': ':4096:8', 'deterministic_warn_only': False,
                     'matmul_precision': 'highest', 'matmul_allow_tf32': False,
                     'cudnn_allow_tf32': False}}


def rejection(saved, current):
    with pytest.raises(ValueError) as error:
        validate_runtime(saved, current)
    message = str(error.value)
    assert message.startswith('Resume runtime/topology differs from checkpoint: ')
    return message


def test_identical_runtime_passes_without_warnings():
    saved = cuda_runtime()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert validate_runtime(saved, copy.deepcopy(saved)) == []


def test_hypergan_release_alone_still_passes_without_warnings():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['hypergan'] = '2.0.0b1'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert validate_runtime(saved, current) == []


def test_other_identical_card_warns_names_both_uuids_and_resumes():
    saved = cuda_runtime()
    current = cuda_runtime(uuid=SECOND_CARD, visible=(SECOND_CARD, FIRST_CARD))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        messages = validate_runtime(saved, current)
    assert len(messages) == 1 and len(captured) == 1
    assert issubclass(captured[0].category, RuntimeWarning) and captured[0].category is HyperGANHardwareWarning
    assert str(captured[0].message) == messages[0]
    message = messages[0]
    assert FIRST_CARD in message and SECOND_CARD in message
    assert f'cuda.uuid: saved "{FIRST_CARD}", current "{SECOND_CARD}"' in message
    assert 'cuda.visible_devices: saved' in message
    # The owner needs the model named and a way to pin the card next time.
    assert 'NVIDIA RTX A6000' in message and 'CUDA_DEVICE_ORDER' in message


def test_warning_sink_receives_the_message_instead_of_the_warnings_module():
    saved = cuda_runtime()
    current = cuda_runtime(uuid=SECOND_CARD, visible=(SECOND_CARD, FIRST_CARD))
    recorded = []
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        messages = validate_runtime(saved, current, warn=recorded.append)
    assert recorded == messages and len(recorded) == 1


def test_only_the_declared_identity_keys_are_qualified():
    assert HARDWARE_TOLERANCE_KEYS == {
        'cuda.name', 'cuda.capability', 'cuda.rank_devices', 'world_size', 'device',
        'platform', 'machine', 'cuda.uuid', 'cuda.visible_devices', 'cuda.nccl',
        'cuda.gcnArchName'
    }


@pytest.mark.parametrize('path,value', [
    ('version', '12.4'),
    ('cudnn', 90100),
    ('cudnn_benchmark', True),
    ('cudnn_deterministic', False),
    ('matmul_precision', 'high'),
    ('matmul_allow_tf32', True),
    ('cudnn_allow_tf32', True),
    ('deterministic_warn_only', True),
    ('cublas_workspace_config', None),
])
def test_a_different_cuda_runtime_setting_still_rejects(path, value):
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['cuda'][path] = value
    message = rejection(saved, current)
    assert f'cuda.{path}: saved ' in message


@pytest.mark.parametrize('key,value', [
    ('dtype', 'bfloat16'),
    ('default_dtype', 'torch.float64'),
    ('torch', '2.13.0+cu130'),
    ('python', '3.13.1'),
    ('numpy', '1.26.4'),
    ('threads', 4),
    ('particlegan', '0.6.0'),
    ('deterministic_algorithms', False),
])
def test_a_different_top_level_runtime_value_still_rejects(key, value):
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current[key] = value
    message = rejection(saved, current)
    assert f'{key}: saved ' in message


def test_gpu_model_change_warns_with_hypergan_hardware_warning_and_resumes():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['cuda']['name'] = 'NVIDIA GeForce RTX 4090'
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        messages = validate_runtime(saved, current)
    assert len(messages) == 1 and len(captured) == 1
    assert captured[0].category is HyperGANHardwareWarning
    assert 'cuda.name' in messages[0]
    assert 'NVIDIA GeForce RTX 4090' in messages[0]


def test_gpu_capability_change_warns_and_resumes():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['cuda']['capability'] = [8, 9]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        messages = validate_runtime(saved, current)
    assert len(messages) == 1 and len(captured) == 1
    assert captured[0].category is HyperGANHardwareWarning
    assert 'cuda.capability' in messages[0]


def test_world_size_change_warns_and_resumes():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['world_size'] = 2
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        messages = validate_runtime(saved, current)
    assert len(messages) == 1 and len(captured) == 1
    assert captured[0].category is HyperGANHardwareWarning
    assert 'world_size' in messages[0]


def test_platform_and_machine_change_warns_and_resumes():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['platform'] = 'Darwin'
    current['machine'] = 'aarch64'
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        messages = validate_runtime(saved, current)
    assert len(messages) == 1 and len(captured) == 1
    assert captured[0].category is HyperGANHardwareWarning
    assert 'platform' in messages[0] and 'machine' in messages[0]


def test_rejection_lists_every_differing_key_path_with_both_values():
    saved = cuda_runtime()
    current = cuda_runtime(uuid=SECOND_CARD, visible=(SECOND_CARD, FIRST_CARD))
    current['default_dtype'] = 'torch.float64'
    current['cuda']['version'] = '12.4'
    message = rejection(saved, current)
    assert 'cuda.version: saved "13.0", current "12.4"' in message
    assert 'default_dtype: saved "torch.float32", current "torch.float64"' in message
    # A card swap does not excuse the incompatible differences beside it, and the
    # identity/hardware keys are not reported as the reason for the rejection.
    assert 'cuda.uuid' not in message and 'cuda.visible_devices' not in message
    assert 'start a new run directory' in message



def test_missing_and_added_key_paths_are_named_as_absent():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['cuda'].pop('matmul_precision')
    current['cuda']['new_setting'] = 'added'
    message = rejection(saved, current)
    assert 'cuda.matmul_precision: saved "highest", current absent' in message
    assert 'cuda.new_setting: saved absent, current "added"' in message


def test_a_cpu_checkpoint_resumed_on_cuda_rejects_and_names_the_device():
    saved = {'python': '3.12.9', 'torch': '2.14.0+cu130', 'numpy': '2.2.1',
             'platform': 'Linux', 'machine': 'x86_64', 'threads': 1,
             'particlegan': '0.5.0', 'device': 'cpu', 'dtype': 'float32',
             'world_size': 1, 'default_dtype': 'torch.float32',
             'deterministic_algorithms': True}
    message = rejection(saved, cuda_runtime())
    assert 'cuda.version: saved absent, current "13.0"' in message
    assert 'cuda.cudnn: saved absent, current 91200' in message
    # Hardware tolerance keys never explain a rejection, even when they appear from nothing.
    assert 'cuda.uuid' not in message and 'cuda.name' not in message and 'device' not in message


@pytest.mark.parametrize('value', [None, ['runtime'], 'runtime', 7])
def test_runtime_metadata_must_be_a_dictionary(value):
    with pytest.raises(ValueError, match='Invalid resume runtime metadata'):
        validate_runtime(value, cuda_runtime())
    with pytest.raises(ValueError, match='Invalid resume runtime metadata'):
        validate_runtime(cuda_runtime(), value)


def test_a_long_value_is_truncated_rather_than_flooding_the_message():
    saved = cuda_runtime()
    current = copy.deepcopy(saved)
    current['cuda']['version'] = 'x' * 4096
    message = rejection(saved, current)
    assert '...' in message and len(message) < 1024
