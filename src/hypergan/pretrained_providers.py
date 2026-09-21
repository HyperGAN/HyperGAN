"""Trusted external checkpoint architectures exposed through native HNDL.

Providers construct published library models without downloading weights.
HNDL verifies and loads the configured local artifact and selects its output.
"""


def _torchvision_resnet18():
    # Keep torchvision optional for every network that does not request it.
    try:
        from torchvision.models import resnet18
        from torch import nn
    except ImportError as exc:
        raise ImportError('The torchvision_resnet18 provider requires hypergan[cifar]') from exc
    model = resnet18(weights=None)
    # Intermediate features participate in gradient penalties. Avoid mutation
    # of tensors retained by autograd or selected by the pretrained operator.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def register_providers(registry):
    """Register host-approved builders on this registry, without global state."""
    if 'torchvision_resnet18' not in registry.pretrained_providers:
        registry.pretrained_provider('torchvision_resnet18', _torchvision_resnet18)
    return registry
