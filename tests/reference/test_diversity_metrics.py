"""Pairwise numerical oracles and undefined-case handling for sample diversity."""
import builtins
import importlib.util
import random

import pytest
import torch

import hypergan.diversity_metrics as diversity_metrics
from hypergan.diversity_metrics import DiversityMoments, SampleDiversity, batch_diversity


def pairwise_rms(samples):
    coordinates = samples.double().reshape(len(samples), -1)
    return (torch.pdist(coordinates).square().mean() / coordinates.shape[1]).sqrt().item()


@pytest.mark.parametrize('shape', [(), (3,), (2, 3), (1, 4, 4), (5, 2, 3)])
def test_streaming_moments_match_distinct_pair_oracle_across_partitions(shape):
    width = 1
    for size in shape:
        width *= size
    reference = torch.arange(7 * width, dtype=torch.float64).reshape(7, *shape).sin() * 11 - 5
    generated = reference * .3 + 41
    expected = pairwise_rms(reference)
    for partitions in ([7], [1] * 7, [2, 4, 1]):
        moments = DiversityMoments()
        batches = []
        offset = 0
        for count in partitions:
            moments.update(reference[offset:offset + count])
            batches.append({'generated': generated[offset:offset + count],
                            'reference': reference[offset:offset + count]})
            offset += count
        assert moments.count == 7 and moments.sample_shape == shape
        assert moments.rms() == pytest.approx(expected, rel=2e-14, abs=1e-14)
        for statistic, expected_value in [('generated_rms', expected * .3), ('reference_rms', expected), ('ratio', .3)]:
            value = SampleDiversity(statistic=statistic).evaluate(batches=iter(batches), context={'sample_count': 7})
            assert value == pytest.approx(expected_value, rel=2e-14, abs=1e-14)


def test_centered_moments_preserve_small_variation_at_large_offset():
    samples = (torch.tensor([0, 1, 4, 9, 16, 25, 36], dtype=torch.float64) / 1024 + 1e12)[:, None]
    expected = pairwise_rms(samples)
    for chunks in ([7], [1] * 7, [3, 4]):
        moments = DiversityMoments()
        for batch in samples.split(chunks):
            moments.update(batch)
        assert moments.rms() == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize('channels', [1, 3, 5])
def test_image_pooling_matches_explicit_pairwise_oracle(channels):
    images = torch.arange(7 * channels * 8 * 12, dtype=torch.float64).reshape(7, channels, 8, 12).cos()
    pooled = torch.nn.functional.adaptive_avg_pool2d(images, 4)
    result = batch_diversity(images * 2 + 17, images, pool_size=4)
    assert result['unavailable'] == {}
    assert result['metrics']['reference_rms'] == pytest.approx(pairwise_rms(pooled), rel=1e-13)
    assert result['metrics']['ratio'] == pytest.approx(2, rel=1e-13)


def test_identical_outputs_zero_and_constant_reference_only_disables_ratio():
    diverse = torch.tensor([[2., 8.], [5., 12.], [11., 16.]])
    constant = torch.full_like(diverse, 100)
    result = batch_diversity(constant, diverse)
    assert result['metrics']['generated_rms'] == result['metrics']['ratio'] == 0
    result = batch_diversity(diverse, constant)
    assert result['metrics'] == {'generated_rms': pytest.approx(pairwise_rms(diverse)), 'reference_rms': 0}
    assert 'nonzero reference' in result['unavailable']['ratio']
    with pytest.raises(ValueError, match='nonzero reference'):
        SampleDiversity().evaluate(batches=[{'generated': diverse, 'reference': constant}], context={})
    assert SampleDiversity(statistic='reference_rms').evaluate(batches=[{'reference': constant}], context={}) == 0


def test_repeated_textured_outputs_have_zero_diversity_despite_spatial_variation():
    texture = (torch.arange(64, dtype=torch.float64).reshape(1, 1, 8, 8) % 2) * 2 - 1
    generated = texture.expand(5, 3, 8, 8)
    reference = generated + torch.arange(5, dtype=torch.float64).reshape(5, 1, 1, 1)
    assert generated.std() > .9
    for pool_size in (None, 4):
        result = batch_diversity(generated, reference, pool_size=pool_size)
        assert result['unavailable'] == {}
        assert result['metrics']['generated_rms'] == result['metrics']['ratio'] == 0
        assert result['metrics']['reference_rms'] > 0


def test_independent_counts_are_not_truncated_or_reference_repeated():
    generated = torch.tensor([[0.], [1.], [2.], [7.], [13.]])
    reference = torch.tensor([[0.], [4.], [17.]])
    expected = pairwise_rms(generated) / pairwise_rms(reference)
    result = batch_diversity(generated, reference)
    assert result['metrics']['ratio'] == pytest.approx(expected)
    batches = [{'generated': generated[:1], 'reference': reference[:2]},
               {'generated': generated[1:], 'reference': reference[2:]}]
    assert SampleDiversity().evaluate(batches=batches, context={}) == pytest.approx(expected)
    with pytest.raises(ValueError, match='reference count 3.*sample_count 5'):
        SampleDiversity().evaluate(batches=batches, context={'sample_count': 5})


@pytest.mark.parametrize('invalid, reason', [
    (torch.zeros(0, 2), 'at least two'),
    (torch.zeros(1, 2), 'at least two'),
    (torch.ones(2, 2, dtype=torch.int64), 'floating'),
    (torch.ones(2, 2, dtype=torch.bool), 'floating'),
    (torch.ones(2, 2, dtype=torch.complex64), 'floating'),
    (torch.tensor([[float('nan'), 0.], [1., 2.]]), 'finite'),
    (torch.tensor([[float('inf'), 0.], [1., 2.]]), 'finite'),
    (torch.zeros(2, 0), 'coordinates'),
    (torch.tensor(4.), 'sample dimension'),
    ([1., 2.], 'tensor'),
    (torch.empty(2, 2, device='meta'), 'materialized'),
])
def test_unsupported_side_preserves_other_absolute_metric(invalid, reason):
    valid = torch.tensor([[1., 2.], [3., 4.]])
    result = batch_diversity(valid, invalid)
    assert result['metrics'] == {'generated_rms': pytest.approx(pairwise_rms(valid))}
    assert reason in result['unavailable']['reference_rms']
    assert 'ratio' in result['unavailable']
    result = batch_diversity(invalid, valid)
    assert result['metrics'] == {'reference_rms': pytest.approx(pairwise_rms(valid))}
    assert reason in result['unavailable']['generated_rms']


def test_different_shapes_do_not_become_comparable_by_flattening_or_pooling():
    left = torch.arange(24.).reshape(4, 2, 3)
    right = left.reshape(4, 3, 2)
    result = batch_diversity(left, right)
    assert set(result['metrics']) == {'generated_rms', 'reference_rms'}
    assert 'matching' in result['unavailable']['ratio']
    left = torch.arange(128.).reshape(2, 1, 8, 8)
    right = left[:, :, :4, :4]
    result = batch_diversity(left, right, pool_size=2)
    assert set(result['metrics']) == {'generated_rms', 'reference_rms'}
    assert 'matching' in result['unavailable']['ratio']


@pytest.mark.parametrize('pool_size', [0, -1, True, 2.5, '4'])
def test_invalid_pool_configuration_returns_reasons_or_strict_constructor_error(pool_size):
    result = batch_diversity(torch.ones(2, 3), torch.ones(2, 3), pool_size=pool_size)
    assert not result['metrics'] and len(result['unavailable']) == 3
    assert all('pool_size' in reason for reason in result['unavailable'].values())
    with pytest.raises(ValueError, match='pool_size'):
        SampleDiversity(pool_size=pool_size)


def test_pooling_undefined_for_vectors_or_upsampling_preserves_supported_side():
    images = torch.arange(32.).reshape(2, 1, 4, 4)
    result = batch_diversity(images, torch.ones(2, 4), pool_size=2)
    assert set(result['metrics']) == {'generated_rms'}
    assert 'NCHW' in result['unavailable']['reference_rms']
    result = batch_diversity(images, images, pool_size=5)
    assert not result['metrics']
    assert 'at least pool_size' in result['unavailable']['generated_rms']


def test_rejected_stream_update_preserves_state_and_shape_contract():
    moments = DiversityMoments().update(torch.tensor([[1., 2.], [4., 8.]]))
    before = moments.rms()
    with pytest.raises(ValueError, match='shape changed'):
        moments.update(torch.zeros(2, 1, 2))
    with pytest.raises(ValueError, match='finite'):
        moments.update(torch.full((2, 2), float('nan')))
    assert moments.count == 2 and moments.rms() == before
    with pytest.raises(ValueError, match='at least two'):
        DiversityMoments().rms()


def test_numerical_overflow_is_visible_instead_of_publishing_nonfinite_values():
    extreme = torch.tensor([[-1e308], [1e308]], dtype=torch.float64)
    valid = torch.tensor([[1.], [2.]])
    result = batch_diversity(extreme, valid)
    assert set(result['metrics']) == {'reference_rms'}
    assert 'overflowed' in result['unavailable']['generated_rms']


def test_observation_preserves_inputs_gradients_rng_and_deterministic_policy():
    generated = torch.arange(48., dtype=torch.float64).reshape(3, 1, 4, 4).requires_grad_()
    reference = generated.detach().flip(0).clone()
    generated.grad = torch.ones_like(generated)
    before, grad = generated.detach().clone(), generated.grad.clone()
    reference_before = reference.clone()
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    policy = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    result = batch_diversity(generated, reference, pool_size=2)
    assert result['metrics']['ratio'] == pytest.approx(1)
    assert torch.equal(generated, before) and torch.equal(reference, reference_before)
    assert torch.equal(generated.grad, grad) and generated.grad_fn is None
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
    assert policy == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def test_module_import_and_metric_description_do_not_import_torch(monkeypatch):
    path = diversity_metrics.__file__
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] == 'torch':
            raise AssertionError('eager torch import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    spec = importlib.util.spec_from_file_location('standalone_diversity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SampleDiversity().describe()['direction'] == 'none'
    assert module.DiversityMoments().count == 0


def test_snapshot_configuration_and_missing_batches_fail_clearly():
    with pytest.raises(ValueError, match='statistic'):
        SampleDiversity(statistic='quality')
    with pytest.raises(ValueError, match='at least two'):
        SampleDiversity().evaluate(batches=[], context={})
    with pytest.raises(ValueError, match='supply'):
        SampleDiversity().evaluate(batches=[{'generated': torch.ones(2, 1)}], context={})
    with pytest.raises(ValueError, match='sample_count'):
        SampleDiversity().evaluate(batches=[], context={'sample_count': True})
