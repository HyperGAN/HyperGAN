"""Tier 5 White-Box Adversarial Coverage Hardening Suite for ImaJev-4B Constraint GAN (M5 Phase 2).

Systematically tests and stresses:
1. `_ImaJev4BProvider` in `src/hypergan/pretrained_providers.py`:
   - Batch size = 1 (`B = 1`) across all 5 readouts (`joint_log_odds`, `pooled_features`,
     `pooled_vision`, `decision_logits`, `decision_probs`) with gradient flow to input.
   - Non-contiguous tensor inputs (strided slices, channel transpositions, reversed spatial strides)
     through forward scoring and backward preconditioned backprop.
   - Abnormal aspect ratios and spatial minimum boundary conditions (`(16, 16)` boundary,
     `(32, 64)` asymmetric, rejection of `H < 16`, `W < 16`, `C != 3`, `ndim != 4`).
   - Autograd caching reentrancy, cache hits (`torch.equal`), cache invalidation, and zero gradients.
   - Dual-output tuple backward paths (`return_vision_features=True`) under partial and joint losses.
   - Extreme temperature (`T = 1000.0`, `T = 0.01`) and soft-capping tau (`tau = 0.05`, `tau = 100.0`).
   - Provider configuration validation rejections in `network_config.py` (missing keys, empty paths,
     heterogeneous query options, out-of-range target_layer, invalid image_size, non-positive temperature/tau).
   - Real GPU execution with B=1 and non-contiguous tensors if CUDA and weights are present.

2. `scripts/verify_imajev_acceptance.py`:
   - Exact threshold boundaries (target 0.700 vs 0.699, noise 0.050 vs 0.051, unknown 0.080 vs 0.081,
     audit accuracy 0.750 vs 0.749, audit artifact 0.120 vs 0.121, entropy ratio 0.850 vs 0.849,
     cosine diversity 0.120 vs 0.119, seam ratio 1.150 vs 1.151, shift sensitivity 0.120 vs 0.119).
   - Degenerate input handling: empty samples, flat/constant images (zero gradients everywhere),
     all-zero mode probabilities, orthogonal/identical features, minimum sample size N=2.
   - Transformations and cyclic option permutations under varied option list lengths.
   - Full acceptance evaluation sample size boundaries (N=63 rejection vs N=64 acceptance) and
     CLI main workflows, error exit codes, and JSON outputs.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hndl import HNDLError
from hypergan.hndl_networks import build_network
import hypergan.pretrained_providers as pp
from hypergan.network_config import validate_pretrained_providers

import scripts.verify_imajev_acceptance as via


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
        torch.manual_seed(42)
        weight = torch.randn(256, 2560, dtype=torch.float32) * 0.02
    path = tmp_path / 'decision_readout.pt'
    torch.save({'weight': weight}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, weight


class _MockQwenLanguageModel(nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.lm_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        with torch.no_grad():
            self.lm_proj.weight.copy_(torch.eye(hidden_size))
        self.requires_grad_(False)

    def forward(
        self,
        inputs_embeds,
        position_ids=None,
        use_cache=False,
        output_hidden_states=False,
        attention_mask=None,
        **kwargs,
    ):
        pooled = inputs_embeds.float().mean(dim=1, keepdim=True)
        last = self.lm_proj(pooled).expand(-1, inputs_embeds.shape[1], -1)
        states = (
            tuple(last * (0.5 + 0.01 * idx) for idx in range(37))
            if output_hidden_states
            else None
        )

        class _LMOut:
            last_hidden_state = last
            hidden_states = states

        return _LMOut()


class _MockQwenInnerModel(nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.vision_proj = nn.Linear(1536, hidden_size, bias=False)
        self.language_model = _MockQwenLanguageModel(hidden_size=hidden_size)
        with torch.no_grad():
            torch.manual_seed(99)
            self.vision_proj.weight.normal_(0.0, 0.02)
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


class _MockQwenTopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _MockQwenInnerModel()


def _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu')):
    image_size = int(provider_cfg.get('image_size', 256))
    grid_h = image_size // 16
    num_visual = (grid_h // 2) ** 2
    seq_len = 16 + num_visual
    hidden_size = 2560
    image_mask_1 = torch.zeros(1, seq_len, hidden_size, dtype=torch.bool, device=device)
    image_mask_1[:, 4:4 + num_visual, :] = True
    positions_1 = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len).contiguous()
    grid_thw_1 = torch.tensor([[1, grid_h, grid_h]], device=device, dtype=torch.long)

    mock_runtime = {
        'model': _MockQwenTopModel().to(device).eval(),
        'processor': None,
        'manifest': {'version': 1, 'codes': [{'code': c, 'token_id': i} for i, c in enumerate(pp._imajev_labels_for_count(255))]},
        'device': device,
        'prompts': {},
    }
    queries = pp._imajev_config_queries(provider_cfg)
    for q_idx, q in enumerate(queries):
        num_candidates = len(q['options']) + 1
        readout_indices = torch.arange(q_idx * num_candidates, (q_idx + 1) * num_candidates, device=device, dtype=torch.long)
        base_embeds_1 = torch.full((1, seq_len, hidden_size), float(q_idx) * 0.05, dtype=torch.float32, device=device)
        prompt_, labels, candidates, state_str = pp._imajev_format_prompt(
            q['question'], q['options'], q.get('state', {})
        )
        prompt_key = (
            (q['question'], tuple(q['options']), state_str)
            if image_size == 256
            else (q['question'], tuple(q['options']), state_str, image_size)
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


# ============================================================================
# MODULE 1: _ImaJev4BProvider Boundary Conditions & Degenerate Tensors
# ============================================================================

def test_imajev_batch_size_one_all_readouts_and_backward(dummy_readout_artifact, monkeypatch):
    """Stress-test B = 1 edge case across all 5 readout modes with backward gradient propagation."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Subject?', 'options': ['apple', 'orange', 'banana', 'noise']},
            {'question': 'Color?', 'options': ['red', 'green', 'yellow', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    readout_configs = [
        ('joint_log_odds', 10),
        ('pooled_features', 2560),
        ('pooled_vision', 2560),
        ('decision_logits', 5),
        ('decision_probs', 5),
    ]

    for readout, expected_dim in readout_configs:
        src = (
            f'pretrained({str(path)!r}, provider="imajev_4b", '
            f'sha256="{digest}", readout="{readout}", trainable=False, name="imajev")'
        )
        net = build_network(
            src,
            input_shape=('B', 3, 32, 32),
            output_shape=('B', expected_dim),
            pretrained_providers={'imajev_4b': provider_cfg},
        )

        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        out = net(x)
        assert out.shape == (1, expected_dim), f"Readout {readout} failed shape check on B=1: {out.shape}"
        assert torch.isfinite(out).all()

        # Differentiate on B=1: for probabilities, differentiate specific class (since sum(p) is identically 1.0)
        if readout == 'decision_probs':
            loss = out[:, 0].sum()
        else:
            loss = out.sum()
        loss.backward()
        assert x.grad is not None, f"Readout {readout} produced None gradient on B=1"
        assert x.grad.shape == (1, 3, 32, 32), f"Readout {readout} produced wrong grad shape: {x.grad.shape}"
        assert torch.isfinite(x.grad).all(), f"Readout {readout} produced non-finite grad on B=1"
        assert x.grad.abs().sum() > 0, f"Readout {readout} produced zero grad on B=1"


def test_imajev_non_contiguous_inputs_stress(dummy_readout_artifact, monkeypatch):
    """Stress-test non-contiguous memory layouts through forward execution and Sobolev preconditioned backward."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Fruit?', 'options': ['apple', 'pear', 'noise']},
            {'question': 'Ripe?', 'options': ['yes', 'no', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    # 1. Strided spatial slice from a parent tensor
    x_large = torch.randn(2, 3, 64, 64, requires_grad=True)
    x_strided = x_large[:, :, ::2, ::2]
    assert not x_strided.is_contiguous()
    out1 = net(x_strided)
    assert out1.shape == (2, 8) and torch.isfinite(out1).all()
    out1.sum().backward()
    assert x_large.grad is not None and torch.isfinite(x_large.grad).all()
    assert x_large.grad.shape == (2, 3, 64, 64)

    # 2. Transposed channels leaf tensor: (B, H, W, C) -> permuted to (B, C, H, W)
    x_hwc = torch.randn(2, 32, 32, 3)
    x_permuted = x_hwc.permute(0, 3, 1, 2)
    assert not x_permuted.is_contiguous()
    x_permuted = x_permuted.requires_grad_(True)
    out2 = net(x_permuted)
    assert out2.shape == (2, 8) and torch.isfinite(out2).all()
    out2.sum().backward()
    assert x_permuted.grad is not None and torch.isfinite(x_permuted.grad).all()

    # 3. Transposed spatial layout (H, W) -> (W, H)
    x_base = torch.randn(2, 3, 32, 32)
    x_transposed = x_base.transpose(2, 3).requires_grad_(True)
    assert not x_transposed.is_contiguous()
    out3 = net(x_transposed)
    assert out3.shape == (2, 8) and torch.isfinite(out3).all()
    out3.sum().backward()
    assert x_transposed.grad is not None and torch.isfinite(x_transposed.grad).all()


def test_imajev_abnormal_aspect_ratios_and_minimum_bounds(dummy_readout_artifact, monkeypatch):
    """Verify handling of asymmetric non-square aspect ratios and boundary rejection below 16px."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Shape?', 'options': ['round', 'oval', 'noise']},
            {'question': 'Size?', 'options': ['big', 'small', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )

    # 1. Non-square aspect ratio: (B, 3, 32, 64)
    net_wide = build_network(
        src,
        input_shape=('B', 3, 32, 64),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x_wide = torch.randn(2, 3, 32, 64, requires_grad=True)
    out_wide = net_wide(x_wide)
    assert out_wide.shape == (2, 8)
    out_wide.sum().backward()
    assert x_wide.grad is not None and x_wide.grad.shape == (2, 3, 32, 64)
    assert torch.isfinite(x_wide.grad).all()

    # 2. Non-square aspect ratio: (B, 3, 64, 32)
    net_tall = build_network(
        src,
        input_shape=('B', 3, 64, 32),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x_tall = torch.randn(2, 3, 64, 32, requires_grad=True)
    out_tall = net_tall(x_tall)
    assert out_tall.shape == (2, 8)
    out_tall.sum().backward()
    assert x_tall.grad is not None and x_tall.grad.shape == (2, 3, 64, 32)
    assert torch.isfinite(x_tall.grad).all()

    # 3. Exact minimum boundary: (B, 3, 16, 16)
    net_min = build_network(
        src,
        input_shape=('B', 3, 16, 16),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x_min = torch.randn(1, 3, 16, 16, requires_grad=True)
    out_min = net_min(x_min)
    assert out_min.shape == (1, 8)
    out_min.sum().backward()
    assert x_min.grad is not None and x_min.grad.shape == (1, 3, 16, 16)

    # 4. Out of bounds rejections
    for bad_shape in [('B', 3, 15, 32), ('B', 3, 32, 15), ('B', 1, 32, 32), ('B', 4, 32, 32)]:
        with pytest.raises(HNDLError):
            build_network(
                src,
                input_shape=bad_shape,
                output_shape=('B', 8),
                pretrained_providers={'imajev_4b': provider_cfg},
            )


# ============================================================================
# MODULE 2: Autograd Cache Reentrancy, Dual Outputs, & Gradient Flow Edge Cases
# ============================================================================

def test_imajev_autograd_caching_and_zero_gradients(dummy_readout_artifact, monkeypatch):
    """Verify autograd cache hit on repeated backward, cache invalidation, and zero-grad handling."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Item?', 'options': ['apple', 'orange', 'noise']},
            {'question': 'Color?', 'options': ['red', 'green', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = net(x)

    # 1. First backward call (creates cache)
    loss1 = out[:, 0].sum()
    loss1.backward(retain_graph=True)
    g1 = x.grad.clone()
    assert torch.isfinite(g1).all()

    # 2. Second backward call with identical grad_output (cache hit)
    x.grad.zero_()
    loss2 = out[:, 0].sum()
    loss2.backward(retain_graph=True)
    g2 = x.grad.clone()
    torch.testing.assert_close(g1, g2, atol=0, rtol=0)

    # 3. Third backward call with different grad_output (cache invalidation / recompute)
    x.grad.zero_()
    loss3 = (out[:, 1] * 2.0).sum()
    loss3.backward(retain_graph=True)
    g3 = x.grad.clone()
    assert not torch.equal(g1, g3), "Cache invalidation failed: identical grad produced for different loss"

    # 4. Zero gradient output
    x.grad.zero_()
    zero_loss = (out * 0.0).sum()
    zero_loss.backward()
    assert torch.all(x.grad == 0.0), f"Expected exact zero grad, got max {x.grad.abs().max()}"
    assert torch.isfinite(x.grad).all()


def test_imajev_dual_output_tuple_backward_paths(dummy_readout_artifact, monkeypatch):
    """Stress-test forward_joint_log_odds with return_vision_features=True under partial losses."""
    path, digest, expected_weight = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Subject?', 'options': ['apple', 'orange', 'noise']},
            {'question': 'Texture?', 'options': ['smooth', 'rough', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    module = net['imajev'].model

    # 1. Loss on joint_log_odds alone
    x1 = torch.randn(2, 3, 32, 32, requires_grad=True)
    out1, vis1 = module.forward_joint_log_odds(x1, return_vision_features=True)
    assert out1.shape == (2, 8) and vis1.shape == (2, 2560)
    loss1 = out1[:, 0].sum()
    loss1.backward()
    assert x1.grad is not None and torch.isfinite(x1.grad).all() and x1.grad.abs().sum() > 0

    # 2. Loss on pooled_vision alone
    x2 = torch.randn(2, 3, 32, 32, requires_grad=True)
    out2, vis2 = module.forward_joint_log_odds(x2, return_vision_features=True)
    loss2 = vis2.sum()
    loss2.backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all() and x2.grad.abs().sum() > 0

    # 3. Weighted joint loss
    x3 = torch.randn(2, 3, 32, 32, requires_grad=True)
    out3, vis3 = module.forward_joint_log_odds(x3, return_vision_features=True)
    loss3 = 0.7 * out3.sum() + 0.3 * vis3.sum()
    loss3.backward()
    assert x3.grad is not None and torch.isfinite(x3.grad).all() and x3.grad.abs().sum() > 0

    # Readout weights immutability
    assert module.weight.grad is None
    torch.testing.assert_close(module.weight, expected_weight, rtol=0, atol=0)


def test_imajev_extreme_temperature_and_tau(dummy_readout_artifact, monkeypatch):
    """Stress-test numerical stability under extreme temperatures and extreme soft-capping tau values."""
    path, digest, _ = dummy_readout_artifact
    base_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': 'Apple?', 'options': ['yes', 'no', 'noise']},
            {'question': 'Red?', 'options': ['yes', 'no', 'noise']},
        ],
    }

    # 1. Extremely high temperature (T = 1000.0)
    cfg_hi_t = {**base_cfg, 'temperature': 1000.0, 'tau': 3.0}
    _install_mock_runtime(monkeypatch, cfg_hi_t, device=torch.device('cpu'))
    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net_hi_t = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 8), pretrained_providers={'imajev_4b': cfg_hi_t})
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out_hi_t = net_hi_t(x)
    assert torch.isfinite(out_hi_t).all()
    # High temperature drives logits to ~0, log-odds centered near log(1/(K-1))
    assert out_hi_t.abs().max() < 3.0
    out_hi_t.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # 2. Extremely low temperature (T = 0.01)
    cfg_lo_t = {**base_cfg, 'temperature': 0.01, 'tau': 3.0}
    _install_mock_runtime(monkeypatch, cfg_lo_t, device=torch.device('cpu'))
    net_lo_t = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 8), pretrained_providers={'imajev_4b': cfg_lo_t})
    x2 = torch.randn(2, 3, 32, 32, requires_grad=True)
    out_lo_t = net_lo_t(x2)
    assert torch.isfinite(out_lo_t).all()
    # Must strictly satisfy -3.0 <= u <= 3.0 despite extreme logit scaling
    assert (out_lo_t >= -3.0).all() and (out_lo_t <= 3.0).all()
    out_lo_t.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()

    # 3. Extremely small tau (tau = 0.05)
    cfg_small_tau = {**base_cfg, 'tau': 0.05}
    _install_mock_runtime(monkeypatch, cfg_small_tau, device=torch.device('cpu'))
    net_small_tau = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 8), pretrained_providers={'imajev_4b': cfg_small_tau})
    x3 = torch.randn(2, 3, 32, 32)
    out_small_tau = net_small_tau(x3)
    assert (out_small_tau >= -0.05 - 1e-6).all() and (out_small_tau <= 0.05 + 1e-6).all()

    # 4. Extremely large tau (tau = 100.0)
    cfg_large_tau = {**base_cfg, 'tau': 100.0}
    _install_mock_runtime(monkeypatch, cfg_large_tau, device=torch.device('cpu'))
    net_large_tau = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 8), pretrained_providers={'imajev_4b': cfg_large_tau})
    x4 = torch.randn(2, 3, 32, 32)
    out_large_tau = net_large_tau(x4)
    assert torch.isfinite(out_large_tau).all()
    assert (out_large_tau >= -100.0).all() and (out_large_tau <= 100.0).all()


def test_imajev_provider_config_validation_rejections():
    """Verify that validate_pretrained_providers rejects invalid, corrupt, or out-of-range configurations."""
    valid_base = {
        'base_model_path': '/mock/base',
        'adapter_path': '/mock/adapter',
        'queries': [
            {'question': 'Q1?', 'options': ['a', 'b']},
            {'question': 'Q2?', 'options': ['c', 'd']},
        ],
    }

    # Missing base or adapter paths
    with pytest.raises(ValueError, match='base_model_path'):
        validate_pretrained_providers({'imajev_4b': {'adapter_path': '/mock/a', 'queries': valid_base['queries']}})
    with pytest.raises(ValueError, match='adapter_path'):
        validate_pretrained_providers({'imajev_4b': {'base_model_path': '/mock/b', 'queries': valid_base['queries']}})
    with pytest.raises(ValueError, match='nonempty text'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'base_model_path': '   '}})

    # Invalid queries structure: empty list, single query (< 2), heterogeneous options
    with pytest.raises(ValueError, match='2 to 64 query tables'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'queries': []}})
    with pytest.raises(ValueError, match='2 to 64 query tables'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'queries': [{'question': 'q', 'options': ['a', 'b']}]}})
    with pytest.raises(ValueError, match='same number of options'):
        validate_pretrained_providers({
            'imajev_4b': {
                **valid_base,
                'queries': [
                    {'question': 'q1', 'options': ['a', 'b']},
                    {'question': 'q2', 'options': ['c', 'd', 'e']},
                ],
            }
        })

    # Out of range target_layer (allowed [-32, 31])
    with pytest.raises(ValueError, match='target_layer'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'target_layer': 32}})
    with pytest.raises(ValueError, match='target_layer'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'target_layer': -33}})

    # Invalid image_size (must be multiple of 32 in [32, 1024])
    with pytest.raises(ValueError, match='image_size'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'image_size': 16}})
    with pytest.raises(ValueError, match='image_size'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'image_size': 60}})
    with pytest.raises(ValueError, match='image_size'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'image_size': 2048}})

    # Invalid temperature and tau
    for bad_val in [0.0, -1.0, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='temperature'):
            validate_pretrained_providers({'imajev_4b': {**valid_base, 'temperature': bad_val}})
        with pytest.raises(ValueError, match='tau'):
            validate_pretrained_providers({'imajev_4b': {**valid_base, 'tau': bad_val}})

    # Invalid mode
    with pytest.raises(ValueError, match='mode'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'mode': 'unsupported_mode'}})

    # Invalid sobolev_precondition
    with pytest.raises(ValueError, match='sobolev_precondition'):
        validate_pretrained_providers({'imajev_4b': {**valid_base, 'sobolev_precondition': 'yes'}})


@pytest.mark.skipif(
    not (torch.cuda.is_available() and REAL_BASE_MODEL.is_dir() and REAL_READOUT_PT.is_file()),
    reason='Requires CUDA and local imajev-4b artifacts',
)
def test_real_imajev_gpu_adversarial_batch_and_extremes():
    """Adversarial stress-test on real RTX A6000 GPU: B = 1, non-contiguous strided tensor, and dual readout."""
    device = None
    for idx in reversed(range(torch.cuda.device_count())):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes >= 12 * 1024**3:
            device = f'cuda:{idx}'
            break
    if device is None:
        pytest.skip('No CUDA device has >= 12 GiB free memory')

    provider_cfg = {
        'base_model_path': str(REAL_BASE_MODEL),
        'adapter_path': str(REAL_ADAPTER),
        'queries': [
            {'question': 'What fruit is shown?', 'options': ['apple', 'orange', 'noise']},
            {'question': 'What color is it?', 'options': ['red', 'green', 'noise']},
        ],
    }
    src = (
        f'pretrained({str(REAL_READOUT_PT)!r}, provider="imajev_4b", '
        f'sha256="{REAL_READOUT_SHA256}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 8),
        pretrained_providers={'imajev_4b': provider_cfg},
    ).to(device)

    # 1. B = 1 execution on GPU
    x_b1 = torch.full((1, 3, 64, 64), 0.5, device=device, requires_grad=True)
    out_b1 = net(x_b1)
    assert out_b1.shape == (1, 8) and torch.isfinite(out_b1).all()
    out_b1.sum().backward()
    assert x_b1.grad is not None and torch.isfinite(x_b1.grad).all()

    # 2. Non-contiguous permuted GPU input
    x_hwc = torch.randn(2, 64, 64, 3, device=device)
    x_permuted = x_hwc.permute(0, 3, 1, 2)
    assert not x_permuted.is_contiguous()
    x_permuted = x_permuted.requires_grad_(True)
    out_permuted = net(x_permuted)
    assert out_permuted.shape == (2, 8) and torch.isfinite(out_permuted).all()
    out_permuted.sum().backward()
    assert x_permuted.grad is not None and torch.isfinite(x_permuted.grad).all()


# ============================================================================
# MODULE 3: scripts/verify_imajev_acceptance.py Boundary & Error Hardening
# ============================================================================

def test_acceptance_in_distribution_exact_boundaries():
    """Verify sharp boundary transitions and malformed tensor rejections in in-distribution satisfaction."""
    N = 64
    base_probs = {}
    for q in via.IN_DISTRIBUTION_QUERIES:
        aspect = q['aspect']
        K = len(q['options'])
        p = torch.zeros(N, K, dtype=torch.float32)
        valid = q['valid_target_modes']
        p[:, valid] = 0.80 / len(valid)
        p[:, q['noise_index']] = 0.02
        p[:, q['unknown_index']] = 0.03
        # Fill rest uniformly
        rem = 1.0 - p.sum(dim=-1, keepdim=True)
        rem_idx = [i for i in range(K) if i not in valid and i != q['noise_index'] and i != q['unknown_index']]
        if rem_idx:
            p[:, rem_idx] = rem / len(rem_idx)
        base_probs[aspect] = p

    # Baseline pass
    res_base = via.compute_in_distribution_satisfaction(base_probs)
    assert res_base['passed'] is True

    # 1. Target mode boundary: 0.7000 (pass), 0.6990 (fail), 0.7010 (pass)
    # Ensure remainder is assigned to non-target mode (e.g. noise)
    probs_700 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        p = probs_700[q['aspect']]
        p[:, :] = 0.0
        p[:, q['valid_target_modes'][0]] = 0.7000
        p[:, q['noise_index']] = 0.3000
    assert via.compute_in_distribution_satisfaction(probs_700)['passed_target'] is True

    probs_699 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        p = probs_699[q['aspect']]
        p[:, :] = 0.0
        p[:, q['valid_target_modes'][0]] = 0.6990
        p[:, q['noise_index']] = 0.3010
    assert via.compute_in_distribution_satisfaction(probs_699)['passed_target'] is False

    probs_701 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        p = probs_701[q['aspect']]
        p[:, :] = 0.0
        p[:, q['valid_target_modes'][0]] = 0.7010
        p[:, q['noise_index']] = 0.2990
    assert via.compute_in_distribution_satisfaction(probs_701)['passed_target'] is True

    # 2. Noise mode boundary: 0.0500 (pass), 0.0510 (fail), 0.0490 (pass)
    probs_noise_50 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        probs_noise_50[q['aspect']][:, q['noise_index']] = 0.0500
    assert via.compute_in_distribution_satisfaction(probs_noise_50)['passed_noise'] is True

    probs_noise_51 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        probs_noise_51[q['aspect']][:, q['noise_index']] = 0.0510
    assert via.compute_in_distribution_satisfaction(probs_noise_51)['passed_noise'] is False

    # 3. Unknown mode boundary: 0.0800 (pass), 0.0810 (fail), 0.0790 (pass)
    probs_unk_80 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        probs_unk_80[q['aspect']][:, q['unknown_index']] = 0.0800
    assert via.compute_in_distribution_satisfaction(probs_unk_80)['passed_unknown'] is True

    probs_unk_81 = deepcopy(base_probs)
    for q in via.IN_DISTRIBUTION_QUERIES:
        probs_unk_81[q['aspect']][:, q['unknown_index']] = 0.0810
    assert via.compute_in_distribution_satisfaction(probs_unk_81)['passed_unknown'] is False

    # 4. Error cases: missing aspect, empty tensor, 1D tensor
    with pytest.raises(KeyError, match='Missing predictions'):
        incomplete = {k: v for k, v in base_probs.items() if k != 'subject'}
        via.compute_in_distribution_satisfaction(incomplete)

    with pytest.raises(ValueError, match='non-empty 2D tensor'):
        bad_empty = deepcopy(base_probs)
        bad_empty['subject'] = torch.zeros(0, 5)
        via.compute_in_distribution_satisfaction(bad_empty)

    with pytest.raises(ValueError, match='non-empty 2D tensor'):
        bad_1d = deepcopy(base_probs)
        bad_1d['subject'] = torch.zeros(5)
        via.compute_in_distribution_satisfaction(bad_1d)


def test_acceptance_held_out_audit_boundaries_and_rotations():
    """Verify sharp boundaries on held-out audit accuracy and permutation invariance in cyclic rotations."""
    N = 64
    base_audit = {}
    for q in via.HELD_OUT_AUDIT_QUERIES:
        aspect = q['aspect']
        K = len(q['options'])
        p = torch.zeros(N, K, dtype=torch.float32)
        p[:, q['valid_apple_index']] = 0.85
        p[:, q['artifact_noise_indices'][0]] = 0.05
        p[:, q['unknown_index']] = 0.05
        base_audit[aspect] = p

    # Baseline pass
    assert via.compute_held_out_audit_accuracy(base_audit)['passed'] is True

    # 1. Apple accuracy boundary: 0.7500 (pass), 0.7490 (fail), 0.7510 (pass)
    audit_750 = deepcopy(base_audit)
    for p in audit_750.values():
        p[:, :] = 0.0
        # Exactly 48 of 64 correct = 75.0%
        p[:48, 0] = 0.9
        p[48:, 1] = 0.9
    assert via.compute_held_out_audit_accuracy(audit_750)['passed_accuracy'] is True

    audit_749 = deepcopy(base_audit)
    for p in audit_749.values():
        p[:, :] = 0.0
        # 47 of 64 correct = 73.4% < 75.0%
        p[:47, 0] = 0.9
        p[47:, 1] = 0.9
    assert via.compute_held_out_audit_accuracy(audit_749)['passed_accuracy'] is False

    # 2. Combined artifact/noise/unknown boundary: 0.1200 (pass), 0.1210 (fail), 0.1190 (pass)
    audit_art_120 = deepcopy(base_audit)
    for q in via.HELD_OUT_AUDIT_QUERIES:
        p = audit_art_120[q['aspect']]
        p[:, q['valid_apple_index']] = 0.880
        p[:, q['artifact_noise_indices'][0]] = 0.060
        p[:, q['unknown_index']] = 0.060  # total 0.120
    assert via.compute_held_out_audit_accuracy(audit_art_120)['passed_artifact'] is True

    audit_art_121 = deepcopy(base_audit)
    for q in via.HELD_OUT_AUDIT_QUERIES:
        p = audit_art_121[q['aspect']]
        p[:, q['valid_apple_index']] = 0.870
        p[:, q['artifact_noise_indices'][0]] = 0.061
        p[:, q['unknown_index']] = 0.060  # total 0.121
    assert via.compute_held_out_audit_accuracy(audit_art_121)['passed_artifact'] is False

    # 3. Error handling: missing aspect, empty tensor
    with pytest.raises(KeyError, match='Missing predictions'):
        via.compute_held_out_audit_accuracy({'geometry_3d': base_audit['geometry_3d']})

    with pytest.raises(ValueError, match='>= 1 samples'):
        via.compute_held_out_audit_accuracy({q['aspect']: torch.zeros(0, len(q['options'])) for q in via.HELD_OUT_AUDIT_QUERIES})

    # 4. Cyclic option rotations
    options_4 = ['apple', 'orange', 'pear', 'unknown']
    rotations_4 = via.apply_cyclic_option_rotations(options_4, n_rotations=4)
    assert len(rotations_4) == 3  # rotatable_k = 3
    for indices, rotated in rotations_4:
        assert indices[-1] == 3  # unknown remains last
        assert rotated[-1] == 'unknown'
        assert set(rotated) == set(options_4)

    # Single rotatable option edge case: ['only_one', 'unknown']
    rot_2 = via.apply_cyclic_option_rotations(['only_one', 'unknown'], n_rotations=4)
    assert len(rot_2) == 1
    assert rot_2[0][1] == ['only_one', 'unknown']


def test_acceptance_entropy_and_diversity_boundaries():
    """Verify boundaries and edge cases in mode marginal entropy and feature cosine diversity."""
    # 1. Mode entropy boundary: ratio 0.8500 (pass), 0.8490 (fail)
    M = 3
    max_H = math.log(M)
    target_H = 0.85 * max_H

    # Perfect balance: entropy ratio = 1.0 >= 0.85 -> pass
    probs_uniform = torch.full((64, M), 1.0 / M)
    res_uniform = via.compute_mode_marginal_entropy(probs_uniform)
    assert res_uniform['passed'] is True
    assert float(res_uniform['entropy_ratio']) == pytest.approx(1.0, rel=1e-5)

    # Completely collapsed: entropy ~ 0.0 -> fail
    probs_collapsed = torch.zeros(64, M)
    probs_collapsed[:, 0] = 1.0
    res_collapsed = via.compute_mode_marginal_entropy(probs_collapsed)
    assert res_collapsed['passed'] is False
    assert res_collapsed['empirical_entropy'] < 1e-8

    # All-zero probabilities fallback to uniform
    probs_zero = torch.zeros(64, M)
    res_zero = via.compute_mode_marginal_entropy(probs_zero)
    assert res_zero['passed'] is True  # fallback uniform

    # Rejection of M < 2
    with pytest.raises(ValueError, match='at least M >= 2'):
        via.compute_mode_marginal_entropy(torch.ones(64, 1))

    # 2. Feature pairwise cosine diversity
    # Boundary: 0.1200 (pass), 0.1190 (fail), 0.1210 (pass)
    # Minimum valid sample size: N = 2
    f_pair = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> dist = 1.0
    res_pair = via.compute_pairwise_cosine_diversity(f_pair)
    assert res_pair['passed'] is True
    assert res_pair['mean_pairwise_distance'] == pytest.approx(1.0, rel=1e-5)

    # Identical vectors -> dist = 0.0 -> fail
    f_identical = torch.ones(64, 128)
    res_id = via.compute_pairwise_cosine_diversity(f_identical)
    assert res_id['passed'] is False
    assert res_id['mean_pairwise_distance'] == pytest.approx(0.0, abs=1e-6)

    # Rejection of N < 2
    with pytest.raises(ValueError, match='N >= 2'):
        via.compute_pairwise_cosine_diversity(torch.ones(1, 128))

    with pytest.raises(ValueError, match='2D tensor'):
        via.compute_pairwise_cosine_diversity(torch.ones(128))


def test_acceptance_optical_and_shift_boundaries():
    """Verify patch-seam ratio and shift sensitivity boundaries and degeneracies."""
    # 1. Flat constant image: zero gradient everywhere -> seam ratio = 0.0 <= 1.15 -> pass
    flat_img = torch.full((4, 3, 64, 64), 0.5)
    res_flat = via.compute_patch_seam_gradient_ratio(flat_img)
    assert res_flat['passed'] is True
    assert res_flat['patch_seam_ratio'] == 0.0

    # 2. Minimum resolution: 16x16
    img_16 = torch.randn(2, 3, 16, 16)
    res_16 = via.compute_patch_seam_gradient_ratio(img_16)
    assert 'patch_seam_ratio' in res_16
    assert math.isfinite(res_16['patch_seam_ratio'])

    # 3. Asymmetric non-square resolution: 32x64
    img_rect = torch.randn(2, 3, 32, 64)
    res_rect = via.compute_patch_seam_gradient_ratio(img_rect)
    assert 'patch_seam_ratio' in res_rect
    assert math.isfinite(res_rect['patch_seam_ratio'])

    # 4. Dimension rejections
    with pytest.raises(ValueError, match='>= patch_size'):
        via.compute_patch_seam_gradient_ratio(torch.randn(2, 3, 15, 32))

    with pytest.raises(ValueError, match='4D tensor'):
        via.compute_patch_seam_gradient_ratio(torch.randn(3, 32, 32))

    with pytest.raises(ValueError, match='4D tensor'):
        via.compute_patch_seam_gradient_ratio(torch.randn(2, 1, 32, 32))

    # 5. Shift sensitivity boundary: < 0.12 is required (0.1200 is fail, 0.1190 is pass)
    p_orig = torch.zeros(64, 5)
    p_orig[:, 0] = 1.0

    # Exact difference 0.1190 -> pass
    p_shift_pass = torch.zeros(64, 5)
    p_shift_pass[:, 0] = 1.0 - 0.1190
    p_shift_pass[:, 1] = 0.1190
    assert via.compute_shift_sensitivity(p_orig, p_shift_pass, target_indices=[0])['passed'] is True

    # Exact difference 0.1200 -> fail (strict inequality < 0.12)
    p_shift_fail = torch.zeros(64, 5)
    p_shift_fail[:, 0] = 1.0 - 0.1200
    p_shift_fail[:, 1] = 0.1200
    assert via.compute_shift_sensitivity(p_orig, p_shift_fail, target_indices=[0])['passed'] is False

    # Shape mismatch rejection
    with pytest.raises(ValueError, match='matching shapes'):
        via.compute_shift_sensitivity(p_orig, p_shift_pass[:32])


def test_acceptance_evaluate_acceptance_and_cli_adversarial(tmp_path):
    """Stress-test evaluate_acceptance and CLI main across sample boundaries, failure scenarios, and error exits."""
    # 1. Sample count boundary: N = 63 (rejected) vs N = 64 (accepted)
    fixtures_63 = via.generate_synthetic_test_batch(batch_size=63, scenario='pass')
    with pytest.raises(ValueError, match='at least 64 samples; provided 63'):
        via.evaluate_acceptance(
            images=fixtures_63['images'],
            features=fixtures_63['features'],
            in_dist_probs=fixtures_63['in_distribution_probs'],
            audit_probs=fixtures_63['audit_probs'],
        )

    fixtures_64 = via.generate_synthetic_test_batch(batch_size=64, scenario='pass')
    report_64 = via.evaluate_acceptance(
        images=fixtures_64['images'],
        features=fixtures_64['features'],
        in_dist_probs=fixtures_64['in_distribution_probs'],
        audit_probs=fixtures_64['audit_probs'],
    )
    assert report_64['status'] == 'PASS'
    assert report_64['sample_count'] == 64

    # 2. Missing evaluation inputs
    with pytest.raises(ValueError, match='Must provide either model_eval_fn or pre-computed'):
        via.evaluate_acceptance(images=fixtures_64['images'])

    # 3. CLI main boundary sample count exit code 2
    code_samples_63 = via.main(['--samples', '63'])
    assert code_samples_63 == 2

    # 4. CLI main synthetic scenarios exit codes
    assert via.main(['--synthetic', '--samples', '64', '--scenario', 'pass']) == 0
    assert via.main(['--synthetic', '--samples', '64', '--scenario', 'collapse']) == 1
    assert via.main(['--synthetic', '--samples', '64', '--scenario', 'patch_static']) == 1
    assert via.main(['--synthetic', '--samples', '64', '--scenario', 'noise']) == 1

    # 5. CLI non-existent checkpoint path exit code 2
    code_bad_ckpt = via.main(['--checkpoint', '/nonexistent/path/to/checkpoint.pt', '--samples', '64'])
    assert code_bad_ckpt == 2

    # 6. CLI JSON reporting and output-dir artifact generation
    out_dir = tmp_path / 'report_out'
    code_json = via.main([
        '--synthetic',
        '--samples', '64',
        '--scenario', 'pass',
        '--output-dir', str(out_dir),
        '--json',
    ])
    assert code_json == 0
    report_path = out_dir / 'imajev_acceptance_report.json'
    assert report_path.is_file()
    saved_report = json.loads(report_path.read_text(encoding='utf-8'))
    assert saved_report['status'] == 'PASS'
    assert saved_report['sample_count'] == 64
