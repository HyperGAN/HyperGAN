"""Streaming collapse diagnostic against explicit pairwise distances."""
import pytest
import torch
from torch.nn import functional as F

from hypergan.colorization_metrics import SampleDiversityRatio


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_matches_pairwise_oracle_and_is_invariant_to_batch_partition():
    rng = torch.Generator().manual_seed(72)
    reference = torch.rand(7, 3, 8, 8, generator=rng) * 2 - 1
    generated = reference * .3 + .2
    metric = SampleDiversityRatio(pool_size=2)
    pooled = [F.adaptive_avg_pool2d(x.double(), 2).flatten(1)
              for x in (generated, reference)]
    expected = (torch.pdist(pooled[0]).square().mean()
                / torch.pdist(pooled[1]).square().mean()).sqrt().item()
    for partitions in ([7], [1, 1, 1, 1, 1, 1, 1], [2, 4, 1]):
        generated_batches, reference_batches = (x.split(partitions) for x in (generated, reference))
        batches = ({'generated': g, 'reference': r}
                   for g, r in zip(generated_batches, reference_batches))
        assert metric.evaluate(batches=batches, context={}) == pytest.approx(expected, abs=1e-14)


def test_identical_outputs_zero_matching_spread_one_and_no_variance_cancellation():
    reference = torch.tensor([-1., 0., 1.])[:, None, None, None].expand(3, 3, 4, 4)
    metric = SampleDiversityRatio(pool_size=2)
    assert metric.evaluate(batches=[{'generated': reference, 'reference': reference}], context={}) == 1
    assert metric.evaluate(batches=[{'generated': torch.zeros_like(reference), 'reference': reference}], context={}) == 0
    # Small variation atop a large common mean should survive centered moments.
    near_constant = reference.double() * 1e-10 + .9
    value = metric.evaluate(batches=[{'generated': near_constant, 'reference': reference}], context={})
    assert value == pytest.approx(1e-10, rel=1e-5)
    assert metric.describe()['direction'] == 'none'


@pytest.mark.parametrize('count', [0, 1])
def test_insufficient_samples_rejected(count):
    batches = [] if count == 0 else [{'generated': torch.zeros(1, 3, 4, 4),
                                     'reference': torch.zeros(1, 3, 4, 4)}]
    with pytest.raises(ValueError, match='at least two'):
        SampleDiversityRatio(pool_size=2).evaluate(batches=batches, context={})


def test_zero_reference_spread_and_invalid_inputs_rejected():
    metric = SampleDiversityRatio(pool_size=2)
    rgb = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError, match='nonzero reference'):
        metric.evaluate(batches=[{'generated': rgb, 'reference': rgb}], context={})
    with pytest.raises(ValueError, match='at least pool_size'):
        metric.evaluate(batches=[{'generated': rgb[:, :, :1], 'reference': rgb}], context={})
    with pytest.raises(ValueError, match='finite'):
        metric.evaluate(batches=[{'generated': rgb + float('nan'), 'reference': rgb}], context={})
    for value in (0, 257, True, 2.5):
        with pytest.raises(ValueError, match='pool_size'):
            SampleDiversityRatio(pool_size=value)
