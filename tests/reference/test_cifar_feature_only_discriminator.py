"""Feature-only CIFAR critic retains input gradients through frozen ResNet."""
import hashlib
from pathlib import Path

import torch
from particlegan import GradientPenalty
from torch.nn import functional as F
from torchvision.models import resnet18

from hypergan.hndl_networks import build_network

NETWORKS = Path(__file__).parents[2] / 'examples/networks'


def test_feature_only_scores_freezing_and_penalty(tmp_path):
    with torch.random.fork_rng(devices=[]):
        path = tmp_path / 'resnet18.pth'
        torch.save(resnet18(weights=None).state_dict(), path)
        model = build_network(
            (NETWORKS / 'resnet18-features-discriminator-32.hndl').read_text(),
            input_shape=('B', 3, 32, 32), output_shape=('B', 1),
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
    x = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32).requires_grad_()
    score = model(x)
    for handle in handles:
        handle.remove()
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    expected = (F.interpolate(x, size=64, mode='bilinear', align_corners=False) * .5 + .5 - mean) / std
    torch.testing.assert_close(captured['normalized'][:2], expected)
    assert [tuple(v.shape) for v in captured['resnet']] == [
        (4, 64, 16, 16), (4, 128, 8, 8), (4, 256, 4, 4)]
    torch.testing.assert_close(score, sum(captured[f'feature{i}_score'] for i in (1, 2, 3)) / 3**.5)
    assert all('pixel' not in n.id for n in model.plan.nodes)

    # Force an active b_cap penalty to exercise second derivatives and head gradients.
    penalty = GradientPenalty(arm='b_cap', kappa=0)(model, x.detach(), -x.detach())
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
