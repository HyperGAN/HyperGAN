"""Feature-only CIFAR critic retains input gradients through frozen ResNet."""
import hashlib
from pathlib import Path
import tomllib

import pytest
import torch
from particlegan.grad_regularizers import GradientPenalty
from torch.nn import functional as F
from torchvision.models import resnet18

from hypergan.hndl_networks import build_network

NETWORKS = Path(__file__).parents[2] / 'examples/networks'


@pytest.mark.parametrize('side', [32, 128])
def test_feature_only_scores_freezing_and_penalty(tmp_path, side):
    if side == 32:
        source = (NETWORKS / 'resnet18-features-discriminator-32.hndl').read_text()
    else:
        config = tomllib.loads((NETWORKS.parent / 'logos-tiny-transformer-resnet-features-128.toml').read_text())
        source = config['components']['discriminator']['args']['source']
    with torch.random.fork_rng(devices=[]):
        path = tmp_path / 'resnet18.pth'
        torch.save(resnet18(weights=None).state_dict(), path)
        model = build_network(
            source, input_shape=('B', 3, side, side), output_shape=('B', 1),
            parameters={'weights_path': str(path),
                        'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    model.train()
    backbone = model['resnet'].model
    assert all(not m.training for m in backbone.modules())
    assert all(not p.requires_grad for p in backbone.parameters())
    frozen = {k: v.clone() for k, v in backbone.state_dict().items()}
    captured = {}

    def capture(name):
        def hook(module, args, output):
            captured[name] = output
        return hook

    handles = [model[name].register_forward_hook(capture(name))
               for name in ('normalized', 'resnet', 'feature1_score',
                            'feature2_score', 'feature3_score')]
    # Vary both spatial axes to catch an unintended resize of the 128px input.
    x = torch.arange(2 * 3 * side * side).float().sin().reshape(2, 3, side, side).requires_grad_()
    score = model(x)
    for handle in handles:
        handle.remove()
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    backbone_input = F.interpolate(x, size=64, mode='bilinear', align_corners=False) if side == 32 else x
    expected = (backbone_input * .5 + .5 - mean) / std
    torch.testing.assert_close(captured['normalized'][:2], expected)
    feature_size = 64 if side == 32 else 128
    assert [tuple(v.shape) for v in captured['resnet']] == [
        (4, channels, feature_size // divisor, feature_size // divisor)
        for channels, divisor in ((64, 4), (128, 8), (256, 16))]
    torch.testing.assert_close(score, sum(captured[f'feature{i}_score'] for i in (1, 2, 3)) / 3**.5)
    assert all('pixel' not in n.id for n in model.plan.nodes)

    # Force an active penalty (R1 plus a zero fake-gradient cap) to exercise second derivatives and head gradients.
    penalty = GradientPenalty(kappa=0)(model, x.detach(), -x.detach())
    loss = score.square().mean() + penalty
    loss.backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    assert torch.isfinite(penalty) and penalty > 0
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert all(p.grad is None for p in backbone.parameters())
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)

    # Freezing D for G's update must leave the image-to-score derivative intact.
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    generated = x.detach().clone().requires_grad_()
    model(generated).sum().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
