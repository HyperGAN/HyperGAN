"""Small CPU checks for the unconditioned 256px pixel discriminator control."""
import pytest
import torch
from particlegan.grad_regularizers import GradientPenalty

from hypergan.colorization_components import DCGANDiscriminator256


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('spectral_norm', [False, True])
def test_rgb_scores_are_sample_independent_and_differentiate_inputs(spectral_norm):
    torch.manual_seed(42)
    model = DCGANDiscriminator256(width=2, spectral_norm=spectral_norm).eval()
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    scores = model(x)
    assert scores.shape == (2, 1)
    torch.testing.assert_close(scores[:1], model(x[:1]))
    input_gradient = torch.autograd.grad(scores[0].sum(), x)[0]
    assert input_gradient[0].abs().sum() > 0
    assert torch.count_nonzero(input_gradient[1]) == 0


@pytest.mark.parametrize('spectral_norm', [False, True])
def test_training_and_bcap_double_backward_reach_parameters(spectral_norm):
    torch.manual_seed(43)
    model = DCGANDiscriminator256(width=2, spectral_norm=spectral_norm).train()
    real, fake = torch.randn(2, 3, 256, 256).tanh(), torch.randn(2, 3, 256, 256).tanh()
    # Zero cap guarantees an active penalty, exercising double backward even
    # when the small randomly initialized critic has gradients below one.
    penalty = GradientPenalty(kappa=0)(model, real, fake)
    penalty.backward()
    weight_gradients = [p.grad for p in model.parameters() if p.ndim > 1]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in weight_gradients)
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.softplus(model(fake) - model(real)).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_invalid_shape_and_options_are_rejected():
    model = DCGANDiscriminator256(width=1)
    for shape in [(2, 1, 256, 256), (2, 3, 128, 128), (3, 256, 256)]:
        with pytest.raises(ValueError, match=r'\[batch,3,256,256\]'):
            model(torch.zeros(shape))
    with pytest.raises(ValueError, match='width must be a positive integer'):
        DCGANDiscriminator256(width=0)
    with pytest.raises(ValueError, match='spectral_norm must be a boolean'):
        DCGANDiscriminator256(spectral_norm=1)
