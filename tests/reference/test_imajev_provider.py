"""Native HNDL imajev_4b provider: shape_fn resolution, lightweight state_dict, and differentiable readouts."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
from hndl import HNDLError

from hypergan.hndl_networks import build_network
import hypergan.pretrained_providers as pp


REAL_BASE_MODEL = Path(
    '/nvme/dev/imajev/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/'
    'snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)
REAL_ADAPTER = Path('/nvme/dev/imajev/artifacts/imajev-4b')
REAL_READOUT_PT = REAL_ADAPTER / 'decision_readout.pt'
REAL_READOUT_SHA256 = '7a98c7aac7c0f648ac2fdfd563dc6ccce4cdf4e81b47c6a6a0dae1725b409326'


@pytest.fixture
def dummy_readout_artifact(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        weight = torch.randn(256, 2560, dtype=torch.float32) * 0.02
    path = tmp_path / 'decision_readout.pt'
    torch.save({'weight': weight}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, weight


class _MockQwenInnerModel(torch.nn.Module):
    def __init__(self, hidden_size=2560, num_visual_tokens=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens
        self.vision_proj = torch.nn.Linear(1536, hidden_size, bias=False)
        self.lm_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            torch.manual_seed(19)
            self.vision_proj.weight.normal_(0.0, 0.02)
            self.lm_proj.weight.copy_(torch.eye(hidden_size))
        self.requires_grad_(False)

    def get_image_features(self, pixel_values, image_grid_thw=None, return_dict=True):
        batch = image_grid_thw.shape[0]
        num_patches = pixel_values.shape[0] // batch
        num_visual = num_patches // 4
        projected = self.vision_proj(pixel_values.float()).view(batch, num_patches, self.hidden_size)
        pooled = projected.view(batch, num_visual, 4, self.hidden_size).mean(dim=2)
        out = tuple(pooled[i] for i in range(batch))

        class _VisionOut:
            pooler_output = out

        return _VisionOut()

    def language_model(self, inputs_embeds, position_ids=None, use_cache=False):
        pooled = inputs_embeds.float().mean(dim=1, keepdim=True)
        last = self.lm_proj(pooled).expand(-1, inputs_embeds.shape[1], -1)

        class _LMOut:
            last_hidden_state = last

        return _LMOut()


class _MockQwenTopModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _MockQwenInnerModel()


def _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu')):
    image_size = int(provider_cfg.get('image_size', 256))
    grid_h = image_size // 16
    num_visual = (grid_h // 2) ** 2
    seq_len = 16 + num_visual
    hidden_size = 2560
    base_embeds_1 = torch.zeros(1, seq_len, hidden_size, dtype=torch.float32, device=device)
    image_mask_1 = torch.zeros(1, seq_len, hidden_size, dtype=torch.bool, device=device)
    image_mask_1[:, 4:4 + num_visual, :] = True
    positions_1 = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len).contiguous()
    grid_thw_1 = torch.tensor([[1, grid_h, grid_h]], device=device, dtype=torch.long)
    num_candidates = len(provider_cfg['options']) + 1
    readout_indices = torch.arange(num_candidates, device=device, dtype=torch.long)

    mock_runtime = {
        'model': _MockQwenTopModel().to(device).eval(),
        'processor': None,
        'manifest': {'version': 1, 'codes': [{'code': c, 'token_id': i} for i, c in enumerate(pp._imajev_labels_for_count(255))]},
        'device': device,
        'prompts': {},
    }
    prompt_, labels, candidates, state_str = pp._imajev_format_prompt(
        provider_cfg['question'], provider_cfg['options'], provider_cfg.get('state', {})
    )
    prompt_key = (
        (provider_cfg['question'], tuple(provider_cfg['options']), state_str)
        if image_size == 256
        else (provider_cfg['question'], tuple(provider_cfg['options']), state_str, image_size)
    )
    mock_runtime['prompts'][prompt_key] = {
        'prompt': prompt_,
        'rendered': prompt_ + '<think>\n\n</think>\n\n',
        'labels': tuple(labels),
        'candidates': tuple(candidates),
        'base_embeds_1': base_embeds_1,
        'positions_1': positions_1,
        'grid_thw_1': grid_thw_1,
        'image_mask_1': image_mask_1,
        'readout_indices': readout_indices,
        'image_size': image_size,
    }
    cache_key = (str(provider_cfg['base_model_path']), str(provider_cfg['adapter_path']), str(device))
    monkeypatch.setitem(pp._IMAJEV_RUNTIME_CACHE, cache_key, mock_runtime)
    return mock_runtime


@pytest.mark.parametrize('readout', ['decision_logits', 'decision_probs'])
def test_imajev_provider_shape_resolution_and_lightweight_deepcopy(dummy_readout_artifact, readout):
    path, digest, expected_weight = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'question': 'Which digit is shown in the image?',
        'options': ['0 — digit zero', '1 — digit one', '2 — digit two'],
    }
    source = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="{readout}", trainable=False, name="imajev")'
    )
    model = build_network(
        source,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 4),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    backbone = model['imajev'].model
    assert dict(backbone.named_parameters()).keys() == {'weight'}
    assert not backbone.weight.requires_grad
    assert not backbone.training
    torch.testing.assert_close(backbone.weight, expected_weight, rtol=0, atol=0)

    copied = deepcopy(model)
    copied_bytes = sum(p.numel() * p.element_size() for p in copied.parameters())
    assert copied_bytes == 256 * 2560 * 4


def test_imajev_provider_rejects_invalid_shape_or_readout(dummy_readout_artifact):
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'question': 'Is this valid?',
        'options': ['yes', 'no'],
    }
    with pytest.raises(HNDLError):
        build_network(
            f'pretrained({str(path)!r}, provider="imajev_4b", sha256="{digest}", readout="decision_logits")',
            input_shape=('B', 1, 32, 32),
            output_shape=('B', 3),
            pretrained_providers={'imajev_4b': provider_cfg},
        )
    with pytest.raises(HNDLError):
        build_network(
            f'pretrained({str(path)!r}, provider="imajev_4b", sha256="{digest}", readout="unknown_readout")',
            input_shape=('B', 3, 32, 32),
            output_shape=('B', 3),
            pretrained_providers={'imajev_4b': provider_cfg},
        )


def test_imajev_mocked_forward_and_backward_gradients(dummy_readout_artifact, monkeypatch):
    path, digest, expected_weight = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'question': 'What color is the object?',
        'options': ['red', 'green', 'blue'],
        'state': {'scene': 'studio'},
        'temperature': 1.5,
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    logits_src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="decision_logits", trainable=False, name="imajev")'
    )
    probs_src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="decision_probs", trainable=False, name="imajev")'
    )
    logits_net = build_network(
        logits_src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 4),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    probs_net = build_network(
        probs_src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 4),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    x = torch.linspace(-1.2, 1.2, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32).requires_grad_(True)
    logits = logits_net(x)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)

    loss = (logits[:, 0] - logits[:, 1]).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert logits_net['imajev'].model.weight.grad is None
    torch.testing.assert_close(logits_net['imajev'].model.weight, expected_weight, rtol=0, atol=0)

    x2 = x.detach().clone().requires_grad_(True)
    probs = probs_net(x2)
    assert probs.shape == (2, 4)
    torch.testing.assert_close(probs, logits.detach().softmax(dim=-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
    probs[:, 0].sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all() and x2.grad.abs().sum() > 0


def test_imajev_custom_image_size_and_compile_flag(dummy_readout_artifact, monkeypatch):
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'question': 'What is the object?',
        'options': ['apple', 'orange'],
        'image_size': 128,
        'compile': True,
    }
    mock_runtime = _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))
    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="decision_logits", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 128, 128),
        output_shape=('B', 3),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    logits = net(x)
    assert logits.shape == (2, 3)
    assert mock_runtime.get('compiled') is True
    logits[:, 0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.skipif(
    not (torch.cuda.is_available() and REAL_BASE_MODEL.is_dir() and REAL_READOUT_PT.is_file()),
    reason='Requires CUDA and local imajev-4b artifacts',
)
def test_real_imajev_4b_gpu_scores_and_gradients():
    device = None
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes >= 12 * 1024**3:
            device = f'cuda:{idx}'
            break
    if device is None:
        pytest.skip('No CUDA device has at least 12 GiB free memory')

    provider_cfg = {
        'base_model_path': str(REAL_BASE_MODEL),
        'adapter_path': str(REAL_ADAPTER),
        'question': 'What color is the image?',
        'options': ['red — mostly red pixels', 'blue — mostly blue pixels'],
    }
    source = (
        f'pretrained({str(REAL_READOUT_PT)!r}, provider="imajev_4b", '
        f'sha256="{REAL_READOUT_SHA256}", readout="decision_probs", trainable=False, name="imajev")'
    )
    net = build_network(
        source,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 3),
        pretrained_providers={'imajev_4b': provider_cfg},
    ).to(device)

    x = torch.full((2, 3, 64, 64), -0.9, device=device)
    x[0, 0] = 0.9  # image 0: red
    x[1, 2] = 0.9  # image 1: blue
    x.requires_grad_(True)

    probs = net(x)
    assert probs.shape == (2, 3)
    assert torch.isfinite(probs).all()
    assert probs[0, 0] > 0.90
    assert probs[1, 1] > 0.90

    (probs[0, 0] + probs[1, 1]).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert net['imajev'].model.weight.grad is None

    # Verify CPU-input fallback (e.g. ema_graph.cpu() during preview rendering) uses the cached GPU runtime.
    cpu_net = deepcopy(net).cpu()
    x_cpu = x.detach().cpu().requires_grad_(True)
    probs_cpu = cpu_net(x_cpu)
    assert probs_cpu.device.type == 'cpu'
    torch.testing.assert_close(probs_cpu, probs.detach().cpu(), atol=1e-4, rtol=1e-4)
    probs_cpu[:, 0].sum().backward()
    assert x_cpu.grad is not None and torch.isfinite(x_cpu.grad).all() and x_cpu.grad.abs().sum() > 0
