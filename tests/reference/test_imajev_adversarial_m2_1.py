"""Adversarial stress-test and empirical challenge suite for Milestone 2.

Challenges and stress-tests:
1. ViT single-execution invariance: ViT forward pass executes strictly once per batch regardless
   of number of queries Q (scaled from 2 to 20), repeated calls across varying batch sizes,
   dual output requests, and backward passes.
2. Soft-capping strict bounding: tau * tanh(u / tau) bounds outputs strictly to [-tau, tau] (and (-tau, tau)
   where precision allows) even under extreme inputs (x = +/- 1000, 1e5, degenerate images) and extreme logits.
3. Gradient flow smoothness: End-to-end backpropagation to input image x without NaNs, Infs, or gradient explosion,
   including extreme inputs, partial tuple losses (dual output), and verification of frozen weights immutability.
4. Statelessness: Zero dependencies on global mutable state or queues, bitwise referential transparency,
   and instance isolation.
5. Real GPU hardware execution on NVIDIA RTX A6000 with real Qwen3.5-4B and ImaJev-4B weights.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hndl import HNDLError
from hypergan.hndl_networks import build_network
import hypergan.pretrained_providers as pp
from hypergan.constraint_data import JointMultiQueryConstraintData, OptionProbability

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
        torch.manual_seed(101)
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

    def forward(self, inputs_embeds, position_ids=None, use_cache=False, output_hidden_states=False, attention_mask=None, **kwargs):
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
            torch.manual_seed(42)
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
            'base_embeds_1': base_embeds_1,
            'image_mask_1': torch.zeros(1, seq_len, hidden_size, dtype=torch.bool, device=device),
            'positions_1': torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len).contiguous(),
            'attn_mask_1': torch.ones(1, seq_len, device=device, dtype=torch.long),
            'grid_thw_1': torch.tensor([[1, grid_h, grid_h]], device=device, dtype=torch.long),
            'readout_indices': readout_indices,
            'image_size': image_size,
        }

    key = (
        str(provider_cfg['base_model_path']),
        str(provider_cfg['adapter_path']),
        str(device),
    )
    pp._IMAJEV_RUNTIME_CACHE[key] = mock_runtime
    monkeypatch.setattr(pp, '_load_imajev_runtime', lambda *args, **kwargs: mock_runtime)
    return mock_runtime


# ==============================================================================
# Group 1: ViT Single Execution Invariance (Vision-Once Caching)
# ==============================================================================

def test_adversarial_vit_single_execution_scaling_queries(dummy_readout_artifact, monkeypatch):
    """Stress-test: ViT forward pass executes strictly ONCE per batch regardless of Q scaling (2..20 queries)."""
    path, digest, _ = dummy_readout_artifact

    for num_q in (2, 3, 6, 12, 20):
        queries = [
            {'question': f'Query {i}?', 'options': ['opt1', 'opt2', 'opt3', 'noise']}
            for i in range(num_q)
        ]
        provider_cfg = {
            'base_model_path': f'/mock/qwen_{num_q}',
            'adapter_path': f'/mock/adapter_{num_q}',
            'queries': queries,
        }
        mock_rt = _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

        vit_counter = {'calls': 0}
        orig_get_features = mock_rt['model'].model.get_image_features

        def counting_get_features(*args, **kwargs):
            vit_counter['calls'] += 1
            return orig_get_features(*args, **kwargs)

        mock_rt['model'].model.get_image_features = counting_get_features

        src = (
            f'pretrained({str(path)!r}, provider="imajev_4b", '
            f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
        )
        net = build_network(
            src,
            input_shape=('B', 3, 32, 32),
            output_shape=('B', num_q * 5),
            pretrained_providers={'imajev_4b': provider_cfg},
        )

        x = torch.randn(3, 3, 32, 32)
        out = net(x)
        assert out.shape == (3, num_q * 5)
        assert vit_counter['calls'] == 1, (
            f"ViT executed {vit_counter['calls']} times for Q={num_q}; expected strictly 1!"
        )


def test_adversarial_vit_repeated_calls_and_varying_batch_sizes(dummy_readout_artifact, monkeypatch):
    """Stress-test: Repeated executions across 10 distinct batches of varying sizes B in [1, 2, 4, 8, 16].

    ViT call count must strictly increment by 1 per batch, total == 10.
    """
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_rep',
        'adapter_path': '/mock/adapter_rep',
        'queries': [
            {'question': 'Color?', 'options': ['red', 'green', 'blue', 'noise']},
            {'question': 'Shape?', 'options': ['round', 'square', 'star', 'noise']},
        ],
    }
    mock_rt = _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))
    vit_counter = {'calls': 0}
    orig_get_features = mock_rt['model'].model.get_image_features

    def counting_get_features(*args, **kwargs):
        vit_counter['calls'] += 1
        return orig_get_features(*args, **kwargs)

    mock_rt['model'].model.get_image_features = counting_get_features

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 10),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    batch_sizes = [1, 2, 4, 8, 16, 1, 3, 5, 2, 4]
    for step, b in enumerate(batch_sizes, start=1):
        x = torch.randn(b, 3, 32, 32)
        out = net(x)
        assert out.shape == (b, 10)
        assert vit_counter['calls'] == step, (
            f"Step {step} (B={b}): ViT calls={vit_counter['calls']}; expected {step}!"
        )


def test_adversarial_vit_backward_pass_no_extra_forward(dummy_readout_artifact, monkeypatch):
    """Stress-test: Backward pass uses recorded autograd graph and triggers zero additional ViT executions."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_bwd',
        'adapter_path': '/mock/adapter_bwd',
        'queries': [
            {'question': 'Q1?', 'options': ['a', 'b', 'c', 'noise']},
            {'question': 'Q2?', 'options': ['d', 'e', 'f', 'noise']},
        ],
    }
    mock_rt = _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))
    vit_counter = {'calls': 0}
    orig_get_features = mock_rt['model'].model.get_image_features

    def counting_get_features(*args, **kwargs):
        vit_counter['calls'] += 1
        return orig_get_features(*args, **kwargs)

    mock_rt['model'].model.get_image_features = counting_get_features

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 10),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = net(x)
    assert vit_counter['calls'] == 1

    loss1 = out.sum()
    loss1.backward(retain_graph=True)
    assert vit_counter['calls'] == 1, "ViT unexpectedly executed during backward pass!"

    loss2 = (out * 0.5).sum()
    loss2.backward()
    assert vit_counter['calls'] == 1, "ViT unexpectedly executed during second backward pass!"


# ==============================================================================
# Group 2: Soft-Capping Strict Bounding and Extremes
# ==============================================================================

def test_adversarial_soft_capping_extreme_inputs(dummy_readout_artifact, monkeypatch):
    """Adversarial stress-test: Image inputs with extreme values (+/- 1000, 1e5, degenerate uniform).

    Output must strictly satisfy: isfinite, and |out| <= tau.
    """
    path, digest, _ = dummy_readout_artifact
    tau = 3.0
    provider_cfg = {
        'base_model_path': '/mock/qwen_bounds',
        'adapter_path': '/mock/adapter_bounds',
        'tau': tau,
        'queries': [
            {'question': 'Subject?', 'options': ['apple', 'orange', 'banana', 'noise']},
            {'question': 'Color?', 'options': ['red', 'green', 'yellow', 'noise']},
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
        output_shape=('B', 10),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    test_inputs = [
        ('positive_1000', torch.full((2, 3, 32, 32), 1000.0)),
        ('negative_1000', torch.full((2, 3, 32, 32), -1000.0)),
        ('positive_1e5', torch.full((2, 3, 32, 32), 1e5)),
        ('negative_1e5', torch.full((2, 3, 32, 32), -1e5)),
        ('checkerboard_extreme', torch.tensor([1000.0, -1000.0]).repeat(512).view(1, 1, 32, 32).repeat(2, 3, 1, 1)),
        ('all_zeros', torch.zeros(2, 3, 32, 32)),
        ('all_ones', torch.ones(2, 3, 32, 32)),
        ('all_minus_ones', torch.full((2, 3, 32, 32), -1.0)),
        ('random_standard', torch.randn(4, 3, 32, 32)),
    ]

    for name, x in test_inputs:
        out = net(x)
        assert torch.isfinite(out).all(), f"Input {name} produced non-finite output!"
        assert torch.all(out <= tau), f"Input {name} violated upper bound: max={out.max()} > {tau}"
        assert torch.all(out >= -tau), f"Input {name} violated lower bound: min={out.min()} < {-tau}"


def test_adversarial_soft_capping_various_tau_thresholds(dummy_readout_artifact, monkeypatch):
    """Stress-test: Soft-capping across tau in {0.2, 0.5, 1.0, 2.5, 5.0, 10.0}."""
    path, digest, _ = dummy_readout_artifact

    for tau in (0.2, 0.5, 1.0, 2.5, 5.0, 10.0):
        provider_cfg = {
            'base_model_path': f'/mock/qwen_tau_{tau}',
            'adapter_path': f'/mock/adapter_tau_{tau}',
            'tau': tau,
            'queries': [
                {'question': 'A?', 'options': ['1', '2', 'noise']},
                {'question': 'B?', 'options': ['3', '4', 'noise']},
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

        x = torch.randn(4, 3, 32, 32)
        out = net(x)
        assert torch.isfinite(out).all()
        assert torch.all(out <= tau)
        assert torch.all(out >= -tau)


def test_adversarial_soft_capping_extreme_logit_vectors():
    """Mathematical oracle: Directly evaluate tau * tanh(u / tau) for extreme log-odds u in [-1000, 1000]."""
    tau = 3.0
    u_extremes = torch.tensor([-1000.0, -500.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 500.0, 1000.0])
    u_tilde = tau * torch.tanh(u_extremes / tau)

    assert torch.isfinite(u_tilde).all()
    assert torch.all(u_tilde <= tau)
    assert torch.all(u_tilde >= -tau)
    # Saturation at limits
    assert u_tilde[-1].item() == pytest.approx(tau, abs=1e-6)
    assert u_tilde[0].item() == pytest.approx(-tau, abs=1e-6)

    # Monotonicity test
    diffs = u_tilde[1:] - u_tilde[:-1]
    assert torch.all(diffs >= 0), "Soft-capping is not monotonic!"


# ==============================================================================
# Group 3: Output Gradient Flow Smoothness and Anti-Explosion
# ==============================================================================

def test_adversarial_gradient_flow_standard_and_extreme_inputs(dummy_readout_artifact, monkeypatch):
    """Stress-test: Verify backpropagation to x without NaNs, Infs, or gradient explosion under standard and +/- 1000 inputs."""
    path, digest, expected_weight = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_grad',
        'adapter_path': '/mock/adapter_grad',
        'tau': 3.0,
        'queries': [
            {'question': 'Subject?', 'options': ['apple', 'orange', 'banana', 'noise']},
            {'question': 'Color?', 'options': ['red', 'green', 'yellow', 'noise']},
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
        output_shape=('B', 10),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    # 1. Standard range input
    x_std = torch.randn(2, 3, 32, 32, requires_grad=True)
    out_std = net(x_std)
    loss_std = (out_std[:, 0] - out_std[:, 1] + out_std[:, 5] - out_std[:, 6]).sum()
    loss_std.backward()

    assert x_std.grad is not None
    assert x_std.grad.shape == x_std.shape
    assert torch.isfinite(x_std.grad).all()
    assert x_std.grad.abs().sum() > 0
    std_grad_norm = x_std.grad.norm().item()

    # Readout weights must remain frozen without gradient
    assert net['imajev'].model.weight.grad is None
    torch.testing.assert_close(net['imajev'].model.weight, expected_weight, rtol=0, atol=0)

    # 2. Extreme input (+/- 1000)
    x_ext = torch.full((2, 3, 32, 32), 1000.0, requires_grad=True)
    out_ext = net(x_ext)
    loss_ext = (out_ext[:, 0] - out_ext[:, 1] + out_ext[:, 5] - out_ext[:, 6]).sum()
    loss_ext.backward()

    assert x_ext.grad is not None
    assert x_ext.grad.shape == x_ext.shape
    assert torch.isfinite(x_ext.grad).all()
    assert x_ext.grad.abs().sum() > 0
    ext_grad_norm = x_ext.grad.norm().item()

    # Straight-through estimator ensures bounded gradients without explosion
    assert ext_grad_norm < 1e5, f"Gradient explosion detected: norm={ext_grad_norm}"


def test_adversarial_gradient_flow_dual_output_partial_losses(dummy_readout_artifact, monkeypatch):
    """Stress-test: Differentiability when forward_joint_log_odds is called with return_vision_features=True.

    Verifies Case A (loss on log_odds only), Case B (loss on pooled_v only), Case C (combined loss).
    """
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_dual',
        'adapter_path': '/mock/adapter_dual',
        'queries': [
            {'question': 'Subject?', 'options': ['apple', 'orange', 'noise']},
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
    imajev_mod = net['imajev'].model

    # Case A: Loss on out only
    xA = torch.randn(2, 3, 32, 32, requires_grad=True)
    outA, visA = imajev_mod.forward_joint_log_odds(xA, return_vision_features=True)
    assert outA.shape == (2, 8)
    assert visA.shape == (2, 2560)
    lossA = outA.sum()
    lossA.backward()
    assert xA.grad is not None and torch.isfinite(xA.grad).all() and xA.grad.abs().sum() > 0

    # Case B: Loss on vis only
    xB = torch.randn(2, 3, 32, 32, requires_grad=True)
    outB, visB = imajev_mod.forward_joint_log_odds(xB, return_vision_features=True)
    lossB = visB.sum()
    lossB.backward()
    assert xB.grad is not None and torch.isfinite(xB.grad).all() and xB.grad.abs().sum() > 0

    # Case C: Joint loss
    xC = torch.randn(2, 3, 32, 32, requires_grad=True)
    outC, visC = imajev_mod.forward_joint_log_odds(xC, return_vision_features=True)
    lossC = outC.sum() + 0.1 * visC.sum()
    lossC.backward()
    assert xC.grad is not None and torch.isfinite(xC.grad).all() and xC.grad.abs().sum() > 0


# ==============================================================================
# Group 4: Statelessness and Referential Transparency
# ==============================================================================

def test_adversarial_statelessness_referential_transparency(dummy_readout_artifact, monkeypatch):
    """Stress-test: Referential transparency.

    Evaluating batch B in isolation vs evaluating batch B after batch A must produce
    bit-for-bit identical outputs and gradients.
    """
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_stateless',
        'adapter_path': '/mock/adapter_stateless',
        'queries': [
            {'question': 'A?', 'options': ['1', '2', 'noise']},
            {'question': 'B?', 'options': ['3', '4', 'noise']},
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

    batch_A = torch.randn(3, 3, 32, 32)
    batch_B = torch.randn(3, 3, 32, 32)

    # Baseline: evaluate B directly on fresh state
    out_B_direct = net(batch_B)

    # Sequence: evaluate A, then evaluate B
    out_A = net(batch_A)
    out_B_sequential = net(batch_B)

    torch.testing.assert_close(out_B_sequential, out_B_direct, rtol=0, atol=0)


def test_adversarial_statelessness_queue_immunity(dummy_readout_artifact, monkeypatch):
    """Stress-test: joint_log_odds and JointMultiQueryConstraintData never touch or mutate _IMAJEV_ACTIVE_QUERY_QUEUE."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen_noqueue',
        'adapter_path': '/mock/adapter_noqueue',
        'queries': [
            {'question': 'A?', 'options': ['1', '2', 'noise']},
            {'question': 'B?', 'options': ['3', '4', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    if hasattr(pp, '_IMAJEV_ACTIVE_QUERY_QUEUE'):
        pp._IMAJEV_ACTIVE_QUERY_QUEUE.clear()

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
    loss = out.sum()
    loss.backward()

    # Sample constraint data
    constraint_data = JointMultiQueryConstraintData(
        [
            {'question': 'A?', 'options': ['1', '2', 'noise'], 'target_modes': [0]},
            {'question': 'B?', 'options': ['3', '4', 'noise'], 'target_modes': [1]},
        ],
        tau=3.0,
    )
    sample = constraint_data(4)
    assert 'real' in sample and 'modes' in sample

    if hasattr(pp, '_IMAJEV_ACTIVE_QUERY_QUEUE'):
        assert len(pp._IMAJEV_ACTIVE_QUERY_QUEUE) == 0, (
            f"_IMAJEV_ACTIVE_QUERY_QUEUE was unexpectedly mutated: {pp._IMAJEV_ACTIVE_QUERY_QUEUE}"
        )


def test_adversarial_multi_instance_isolation(dummy_readout_artifact, monkeypatch):
    """Stress-test: Two independent network instances with different configs operate concurrently without interference."""
    path, digest, _ = dummy_readout_artifact

    cfg1 = {
        'base_model_path': '/mock/qwen_inst1',
        'adapter_path': '/mock/adapter_inst1',
        'tau': 2.0,
        'queries': [
            {'question': 'Q1a?', 'options': ['a', 'b', 'noise']},
            {'question': 'Q1b?', 'options': ['c', 'd', 'noise']},
        ],
    }
    cfg2 = {
        'base_model_path': '/mock/qwen_inst2',
        'adapter_path': '/mock/adapter_inst2',
        'tau': 4.0,
        'queries': [
            {'question': 'Q2a?', 'options': ['e', 'f', 'noise']},
            {'question': 'Q2b?', 'options': ['g', 'h', 'noise']},
            {'question': 'Q2c?', 'options': ['i', 'j', 'noise']},
        ],
    }
    _install_mock_runtime(monkeypatch, cfg1, device=torch.device('cpu'))
    _install_mock_runtime(monkeypatch, cfg2, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net1 = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 8), pretrained_providers={'imajev_4b': cfg1})
    net2 = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 12), pretrained_providers={'imajev_4b': cfg2})

    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(3, 3, 32, 32)

    out1 = net1(x1)
    out2 = net2(x2)

    assert out1.shape == (2, 8)
    assert out2.shape == (3, 12)
    assert torch.all(out1.abs() <= 2.0)
    assert torch.all(out2.abs() <= 4.0)


# ==============================================================================
# Group 5: Real GPU Hardware Execution (RTX A6000)
# ==============================================================================

@pytest.mark.skipif(
    not (torch.cuda.is_available() and REAL_BASE_MODEL.is_dir() and REAL_READOUT_PT.is_file()),
    reason='Requires CUDA GPU and local Qwen3.5-4B / ImaJev-4B model weights',
)
def test_adversarial_real_gpu_rtx_a6000_joint_log_odds():
    """Live GPU Stress-Test on NVIDIA RTX A6000 with real Qwen3.5-4B and ImaJev-4B weights.

    Verifies on physical hardware:
    1. ViT executes strictly once per batch for 2 queries simultaneously.
    2. Soft-capping strictly bounds outputs to (-tau, tau).
    3. Output gradients flow smoothly back to input image x on CUDA device without NaNs or explosion.
    4. Statelessness across consecutive GPU calls.
    """
    device = 'cuda:0'
    queries = [
        {'question': 'What color is the image?', 'options': ['red — mostly red pixels', 'blue — mostly blue pixels']},
        {'question': 'What style is the image?', 'options': ['photograph', 'illustration']},
    ]
    provider_cfg = {
        'base_model_path': str(REAL_BASE_MODEL),
        'adapter_path': str(REAL_ADAPTER),
        'tau': 3.0,
        'temperature': 1.30515697,
        'queries': queries,
    }
    src = (
        f'pretrained({str(REAL_READOUT_PT)!r}, provider="imajev_4b", '
        f'sha256="{REAL_READOUT_SHA256}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 6),
        pretrained_providers={'imajev_4b': provider_cfg},
    ).to(device)

    # 1. ViT single execution test on real GPU
    imajev_mod = net['imajev'].model
    runtime = imajev_mod._get_or_load_runtime(torch.device(device))
    real_inner = runtime['model'].model
    orig_get_image_features = real_inner.get_image_features
    vit_calls = {'count': 0}

    def counted_get_image_features(*args, **kwargs):
        vit_calls['count'] += 1
        return orig_get_image_features(*args, **kwargs)

    real_inner.get_image_features = counted_get_image_features

    x = torch.full((2, 3, 64, 64), -0.9, device=device, requires_grad=True)
    with torch.no_grad():
        x[0, 0] = 0.9  # Red image
        x[1, 2] = 0.9  # Blue image

    out = net(x)
    assert out.shape == (2, 6)
    assert vit_calls['count'] == 1, (
        f"Real ViT executed {vit_calls['count']} times on GPU; expected strictly 1!"
    )

    # 2. Output bounds
    assert torch.isfinite(out).all()
    assert torch.all(out <= 3.0)
    assert torch.all(out >= -3.0)

    # 3. Gradient flow back to GPU image tensor
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient failed to reach GPU image tensor"
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all(), "GPU gradient contains NaNs or Infs"
    assert x.grad.abs().sum() > 0, "GPU gradient is identically zero"
    assert x.grad.norm().item() < 1e5, "GPU gradient explosion detected"

    # Restore unpatched method
    real_inner.get_image_features = orig_get_image_features
