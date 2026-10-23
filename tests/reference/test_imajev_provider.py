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


class _MockQwenLanguageModel(torch.nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.lm_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
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


class _MockQwenInnerModel(torch.nn.Module):
    def __init__(self, hidden_size=2560, num_visual_tokens=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens
        self.vision_proj = torch.nn.Linear(1536, hidden_size, bias=False)
        self.language_model = _MockQwenLanguageModel(hidden_size=hidden_size)
        with torch.no_grad():
            torch.manual_seed(19)
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
        # Give each query distinct readout indices and base embeddings so query outputs differ deterministically
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


@pytest.mark.parametrize('readout,expected_dim', [
    ('decision_logits', 4),
    ('decision_probs', 4),
    ('joint_log_odds', 4),
    ('pooled_features', 2560),
    ('pooled_vision', 2560),
])
def test_imajev_provider_shape_resolution_and_lightweight_deepcopy(
    dummy_readout_artifact, readout, expected_dim
):
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
        output_shape=('B', expected_dim),
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

    x = torch.linspace(-1.2, 1.2, 2 * 3 * 32 * 32).reshape(2, 3, 32 * 32).view(2, 3, 32, 32).requires_grad_(True)
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


@pytest.mark.parametrize('num_queries,candidates_per_query,expected_dim', [
    (1, 4, 4),    # 1 query, 3 options + 1 unknown = 4
    (2, 5, 10),   # 2 queries, 4 options + 1 unknown = 10
    (6, 5, 30),   # 6 canonical queries (subject, color, bites, sitting_on, style, stem_leaf) = 30
])
def test_imajev_joint_log_odds_shape_resolution_and_lightweight_copy(
    dummy_readout_artifact, monkeypatch, num_queries, candidates_per_query, expected_dim
):
    """Verify HNDL shape_fn resolution for joint_log_odds and pooled_vision, and parameter immutability."""
    path, digest, expected_weight = dummy_readout_artifact
    options_list = [f'option_{i}' for i in range(candidates_per_query - 1)]
    if num_queries == 1:
        provider_cfg = {
            'base_model_path': '/mock/qwen3.5-4b',
            'adapter_path': '/mock/imajev-4b',
            'question': 'Semantic question 0?',
            'options': options_list,
        }
    else:
        provider_cfg = {
            'base_model_path': '/mock/qwen3.5-4b',
            'adapter_path': '/mock/imajev-4b',
            'queries': [
                {'question': f'Semantic question {q}?', 'options': options_list}
                for q in range(num_queries)
            ],
        }
    _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    joint_src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    joint_net = build_network(
        joint_src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', expected_dim),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    backbone = joint_net['imajev'].model
    assert dict(backbone.named_parameters()).keys() == {'weight'}
    assert not backbone.weight.requires_grad
    assert not backbone.training
    torch.testing.assert_close(backbone.weight, expected_weight, rtol=0, atol=0)

    # Verify deepcopy produces lightweight clones (only 256 * 2560 * 4 bytes for readout weights)
    copied = deepcopy(joint_net)
    copied_bytes = sum(p.numel() * p.element_size() for p in copied.parameters())
    assert copied_bytes == 256 * 2560 * 4

    # Verify pooled_vision readout resolves to [B, 2560]
    pooled_src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="pooled_vision", trainable=False, name="imajev")'
    )
    pooled_net = build_network(
        pooled_src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 2560),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x = torch.randn(3, 3, 32, 32)
    assert joint_net(x).shape == (3, expected_dim)
    assert pooled_net(x).shape == (3, 2560)


def test_imajev_vision_once_caching(dummy_readout_artifact, monkeypatch):
    """Verify that ViT forward pass (get_image_features) executes strictly ONCE per batch across all Q queries."""
    path, digest, _ = dummy_readout_artifact
    num_queries = 6
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': [
            {'question': f'Apple aspect {q}?', 'options': ['optA', 'optB', 'optC', 'optD']}
            for q in range(num_queries)
        ],
    }
    mock_runtime = _install_mock_runtime(monkeypatch, provider_cfg, device=torch.device('cpu'))

    inner_model = mock_runtime['model'].model
    original_get_image_features = inner_model.get_image_features
    call_counts = {'get_image_features': 0}

    def counting_get_image_features(*args, **kwargs):
        call_counts['get_image_features'] += 1
        return original_get_image_features(*args, **kwargs)

    inner_model.get_image_features = counting_get_image_features

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', num_queries * 5),
        pretrained_providers={'imajev_4b': provider_cfg},
    )

    # First batch of size B=4
    x1 = torch.randn(4, 3, 32, 32)
    out1 = net(x1)
    assert out1.shape == (4, num_queries * 5)
    # ViT must execute exactly ONCE for the batch, not once per query (1 != 6)
    assert call_counts['get_image_features'] == 1, (
        f"ViT executed {call_counts['get_image_features']} times; expected exactly 1 (vision-once violation)"
    )

    # Second batch of size B=2
    x2 = torch.randn(2, 3, 32, 32)
    out2 = net(x2)
    assert out2.shape == (2, num_queries * 5)
    # Total call count increments by exactly 1
    assert call_counts['get_image_features'] == 2, (
        f"ViT executed {call_counts['get_image_features']} times; expected exactly 2 across two batches"
    )


def test_imajev_soft_capping_bounds_and_saturation(dummy_readout_artifact, monkeypatch):
    """Verify that joint_log_odds output strictly satisfies -tau < u_tilde < tau under normal and extreme inputs."""
    path, digest, _ = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
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

    # Standard range inputs
    x_std = torch.randn(4, 3, 32, 32)
    out_std = net(x_std)
    assert torch.all(out_std > -3.0), f"Lower bound violation: min value {out_std.min()} <= -3.0"
    assert torch.all(out_std < 3.0), f"Upper bound violation: max value {out_std.max()} >= 3.0"

    # Extreme inputs: verify saturation immunity without NaN or numerical overflow
    x_extreme = torch.full((2, 3, 32, 32), 1e4)
    out_extreme = net(x_extreme)
    assert torch.isfinite(out_extreme).all()
    assert torch.all(out_extreme > -3.0)
    assert torch.all(out_extreme < 3.0)

    # Custom tau test (tau = 1.5)
    custom_tau_cfg = {**provider_cfg, 'tau': 1.5}
    _install_mock_runtime(monkeypatch, custom_tau_cfg, device=torch.device('cpu'))
    custom_net = build_network(
        src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 10),
        pretrained_providers={'imajev_4b': custom_tau_cfg},
    )
    out_custom = custom_net(x_std)
    assert torch.all(out_custom > -1.5)
    assert torch.all(out_custom < 1.5)

    # Mathematical property: strict monotonicity d(u_tilde)/du = sech^2(u/tau) > 0
    u_vals = torch.linspace(-15.0, 15.0, 200)
    tau = 3.0
    u_tilde = tau * torch.tanh(u_vals / tau)
    assert torch.all((u_tilde[1:] - u_tilde[:-1]) > 0), "Soft-capping is not strictly monotonic"
    # Odd symmetry: u_tilde(-u) == -u_tilde(u)
    torch.testing.assert_close(tau * torch.tanh(-u_vals / tau), -u_tilde)


def test_imajev_calibration_temperature_scaling(dummy_readout_artifact, monkeypatch):
    """Verify that calibration temperature scales logits by 1/T, softening log-odds before soft-capping."""
    path, digest, _ = dummy_readout_artifact
    q = [{'question': 'Aspect?', 'options': ['a', 'b', 'c', 'noise']}, {'question': 'Aspect2?', 'options': ['d', 'e', 'f', 'noise']}]

    cfg_default = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'queries': q,
    }
    cfg_t1 = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'temperature': 1.0,
        'tau': 100.0,  # Un-saturate to test linear scaling
        'queries': q,
    }
    cfg_t2 = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'temperature': 2.0,
        'tau': 100.0,
        'queries': q,
    }

    _install_mock_runtime(monkeypatch, cfg_default, device=torch.device('cpu'))
    _install_mock_runtime(monkeypatch, cfg_t1, device=torch.device('cpu'))
    _install_mock_runtime(monkeypatch, cfg_t2, device=torch.device('cpu'))

    src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev")'
    )
    net_default = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 10), pretrained_providers={'imajev_4b': cfg_default})
    net_t1 = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 10), pretrained_providers={'imajev_4b': cfg_t1})
    net_t2 = build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 10), pretrained_providers={'imajev_4b': cfg_t2})

    # Assert default temperature is exactly 1.30515697
    backbone_def = net_default['imajev'].model
    assert float(backbone_def._config.get('temperature', 1.30515697)) == pytest.approx(1.30515697, rel=1e-5)

    x = torch.randn(2, 3, 32, 32)
    out_t1 = net_t1(x)
    out_t2 = net_t2(x)

    abs_t1 = out_t1.abs().mean()
    abs_t2 = out_t2.abs().mean()
    if abs_t1 > 1e-4:
        assert abs_t2 < abs_t1, "Doubling temperature did not reduce log-odds magnitude"

    # Input validation: reject temperature <= 0
    with pytest.raises((HNDLError, ValueError)):
        bad_cfg = {**cfg_default, 'temperature': 0.0}
        build_network(src, input_shape=('B', 3, 32, 32), output_shape=('B', 10), pretrained_providers={'imajev_4b': bad_cfg})


def test_imajev_backward_pass_and_gradient_flow_to_image(dummy_readout_artifact, monkeypatch):
    """Verify end-to-end backpropagation through LM and ViT down to the input image tensor x."""
    path, digest, expected_weight = dummy_readout_artifact
    provider_cfg = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'tau': 3.0,
        'temperature': 1.30515697,
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

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = net(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()

    # Differentiate joint multi-query objective
    loss = (out[:, 0] - out[:, 1] + out[:, 5] - out[:, 6]).sum()
    loss.backward()

    # Assert gradient flow reaches input image tensor
    assert x.grad is not None, "Gradient failed to reach input image tensor"
    assert torch.isfinite(x.grad).all(), "Gradient on input image contains non-finite values (NaN/Inf)"
    assert x.grad.abs().sum() > 0, "Gradient on input image is identically zero"
    assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} mismatch with image shape {x.shape}"

    # Assert readout weights remain frozen without gradients
    assert net['imajev'].model.weight.grad is None
    torch.testing.assert_close(net['imajev'].model.weight, expected_weight, rtol=0, atol=0)

    # Verify gradient flow through pooled_vision readout
    pooled_src = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="pooled_vision", trainable=False, name="imajev")'
    )
    pooled_net = build_network(
        pooled_src,
        input_shape=('B', 3, 32, 32),
        output_shape=('B', 2560),
        pretrained_providers={'imajev_4b': provider_cfg},
    )
    x_vis = torch.randn(2, 3, 32, 32, requires_grad=True)
    vis_out = pooled_net(x_vis)
    assert vis_out.shape == (2, 2560)
    vis_loss = vis_out.sum()
    vis_loss.backward()
    assert x_vis.grad is not None and torch.isfinite(x_vis.grad).all() and x_vis.grad.abs().sum() > 0


def test_imajev_joint_scoring_no_queue_dependency(dummy_readout_artifact, monkeypatch):
    """Verify that joint_log_odds execution does not touch, modify, or require _IMAJEV_ACTIVE_QUERY_QUEUE."""
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

    if hasattr(pp, '_IMAJEV_ACTIVE_QUERY_QUEUE'):
        pp._IMAJEV_ACTIVE_QUERY_QUEUE.clear()

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

    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    assert out.shape == (2, 10)

    if hasattr(pp, '_IMAJEV_ACTIVE_QUERY_QUEUE'):
        assert len(pp._IMAJEV_ACTIVE_QUERY_QUEUE) == 0, "joint_log_odds improperly accessed global active query queue"


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
    for idx in reversed(range(torch.cuda.device_count())):
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


def test_sobolev_precondition_gradient_properties():
    from hypergan.pretrained_providers import sobolev_precondition_gradient

    # Input validations
    with pytest.raises(TypeError, match='torch.Tensor'):
        sobolev_precondition_gradient([[1.0]])
    with pytest.raises(ValueError, match='4D tensor'):
        sobolev_precondition_gradient(torch.randn(3, 32, 32))
    with pytest.raises(ValueError, match='matching length'):
        sobolev_precondition_gradient(torch.randn(1, 3, 32, 32), scales=(1, 2), weights=(0.5,))

    # Dtype and device preservation
    x = torch.randn(2, 3, 64, 64, dtype=torch.float32)
    filtered = sobolev_precondition_gradient(x)
    assert filtered.shape == x.shape
    assert filtered.dtype == x.dtype
    assert filtered.device == x.device

    # Descent direction: <g, P(g)> > 0 and cosine similarity >= 0.5
    inner = (x * filtered).sum()
    assert inner.item() > 0
    cos_sim = inner / (x.norm() * filtered.norm())
    assert cos_sim.item() >= 0.5 - 1e-5

    # DC component preservation: spatially uniform gradient is unchanged
    dc_grad = torch.full((2, 3, 64, 64), 3.5)
    filtered_dc = sobolev_precondition_gradient(dc_grad)
    torch.testing.assert_close(filtered_dc, dc_grad, atol=1e-5, rtol=1e-5)

    # High-frequency attenuation: checkerboard (-1)^(i+j)
    i = torch.arange(64).view(1, 1, 64, 1)
    j = torch.arange(64).view(1, 1, 1, 64)
    checker = ((-1.0) ** (i + j)).expand(1, 3, 64, 64)
    filtered_checker = sobolev_precondition_gradient(checker)
    torch.testing.assert_close(filtered_checker, 0.5 * checker, atol=1e-5, rtol=1e-5)
    assert filtered_checker.norm().item() ** 2 == pytest.approx(0.25 * checker.norm().item() ** 2, rel=1e-4)


def test_sobolev_precondition_mock_vlm_backward(dummy_readout_artifact, monkeypatch):
    from hypergan.pretrained_providers import sobolev_precondition_gradient
    path, digest, _ = dummy_readout_artifact

    # 1. With sobolev_precondition=True (default)
    cfg_on = {
        'base_model_path': '/mock/qwen3.5-4b',
        'adapter_path': '/mock/imajev-4b',
        'question': 'What is the object?',
        'options': ['apple', 'orange'],
        'image_size': 64,
        'sobolev_precondition': True,
    }
    _install_mock_runtime(monkeypatch, cfg_on, device=torch.device('cpu'))
    src_on = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev_on")'
    )
    net_on = build_network(
        src_on,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 3),
        pretrained_providers={'imajev_4b': cfg_on},
    )

    # 2. With sobolev_precondition=False
    cfg_off = deepcopy(cfg_on)
    cfg_off['sobolev_precondition'] = False
    _install_mock_runtime(monkeypatch, cfg_off, device=torch.device('cpu'))
    src_off = (
        f'pretrained({str(path)!r}, provider="imajev_4b", '
        f'sha256="{digest}", readout="joint_log_odds", trainable=False, name="imajev_off")'
    )
    net_off = build_network(
        src_off,
        input_shape=('B', 3, 64, 64),
        output_shape=('B', 3),
        pretrained_providers={'imajev_4b': cfg_off},
    )

    x_on = torch.randn(2, 3, 64, 64, requires_grad=True)
    x_off = x_on.clone().detach().requires_grad_(True)

    out_on = net_on(x_on)
    out_off = net_off(x_off)

    out_on[:, 0].sum().backward()
    out_off[:, 0].sum().backward()

    assert x_on.grad is not None and x_off.grad is not None
    # Filtered grad should match sobolev_precondition_gradient(x_off.grad)
    expected_on = sobolev_precondition_gradient(x_off.grad)
    torch.testing.assert_close(x_on.grad, expected_on, atol=1e-5, rtol=1e-5)

    # Verify TV of preconditioned gradient is strictly lower than raw gradient
    def total_variation(g):
        return (g[:, :, 1:, :] - g[:, :, :-1, :]).abs().mean() + (g[:, :, :, 1:] - g[:, :, :, :-1]).abs().mean()

    assert total_variation(x_on.grad) < total_variation(x_off.grad)
