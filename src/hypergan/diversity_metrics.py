"""Coordinate sample diversity, without range or RGB assumptions.

Imports stay torch-free. These measurements quantify variation, not realism or
semantic mode coverage; independent noise can have high diversity too.
"""
import math
from collections.abc import Mapping


_STATISTICS = ('generated_rms', 'reference_rms', 'ratio')


def _pool_size(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError('Diversity pool_size must be None or a positive integer')
    return value


class DiversityMoments:
    """Streaming float64 CPU moments with one coordinate per sample feature.

    ``update`` accepts a nonempty real floating tensor shaped ``[N, ...]``;
    ``[N]`` represents scalar samples. Shape is fixed across updates. Optional
    image pooling accepts NCHW tensors with any positive channel count.

    Centering relative to the first sample avoids subtracting large raw second
    moments and preserves small variations around a large common offset. Chan
    merging combines centered batch moments without storing previous samples.
    Input tensors, their gradients, RNGs and backend settings are untouched.
    """
    def __init__(self, pool_size=None):
        self.pool_size = _pool_size(pool_size)
        self.count = 0
        self.sample_shape = None
        self._origin = self._mean = self._m2 = None

    def update(self, samples):
        import torch
        if not isinstance(samples, torch.Tensor):
            raise ValueError('Sample diversity requires a torch tensor')
        if not samples.is_floating_point() or samples.is_complex():
            raise ValueError('Sample diversity requires real floating-point dtype')
        if samples.layout != torch.strided or samples.is_nested or samples.device.type == 'meta':
            raise ValueError('Sample diversity requires a materialized dense tensor')
        if samples.ndim < 1:
            raise ValueError('Sample diversity requires a leading sample dimension [N, ...]')
        if samples.shape[0] == 0:
            raise ValueError('Sample diversity requires at least two samples in total; received an empty batch')
        shape = tuple(samples.shape[1:])
        if any(size == 0 for size in shape):
            raise ValueError('Sample diversity requires nonempty sample coordinates')
        if self.sample_shape is not None and shape != self.sample_shape:
            raise ValueError('Sample diversity sample shape changed between batches')
        values = samples.detach().to(device='cpu', dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Sample diversity requires finite samples')
        if self.pool_size is not None:
            if values.ndim != 4:
                raise ValueError('Diversity pooling requires NCHW image tensors')
            if min(values.shape[-2:]) < self.pool_size:
                raise ValueError('Diversity image height and width must be at least pool_size')
            from torch.nn.functional import adaptive_avg_pool2d
            values = adaptive_avg_pool2d(values, self.pool_size)
        values = values.reshape(len(values), -1)
        origin = values[0].clone() if self._origin is None else self._origin
        centered = values - origin
        mean = centered.mean(0)
        m2 = (centered - mean).square().sum(0)
        count = self.count + len(values)
        if self.count:
            difference = mean - self._mean
            m2 = self._m2 + m2 + difference.square() * (self.count * len(values) / count)
            mean = self._mean + difference * (len(values) / count)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(m2).all()):
            raise ValueError('Sample diversity moments overflowed float64; rescale sample coordinates')
        # Commit only after validation, so a rejected update leaves state intact.
        self._origin, self._mean, self._m2 = origin, mean, m2
        self.count, self.sample_shape = count, shape
        return self

    def rms(self):
        """RMS distance over distinct sample pairs, normalized by coordinates."""
        if self.count < 2:
            raise ValueError('Sample diversity requires at least two samples')
        variance = float((self._m2 / (self.count - 1)).mean())
        if not math.isfinite(variance) or variance < 0:
            raise ValueError('Sample diversity variance is not finite and nonnegative')
        value = math.sqrt(variance) * math.sqrt(2.0)
        if not math.isfinite(value):
            raise ValueError('Sample diversity RMS is not finite')
        return value


def _results(moments, errors=None):
    values, unavailable = {}, dict(errors or {})
    for name, accumulator in moments.items():
        key = name + '_rms'
        if key not in unavailable:
            try:
                values[key] = accumulator.rms()
            except ValueError as exc:
                unavailable[key] = str(exc)
    missing = [key for key in ('generated_rms', 'reference_rms') if key not in values]
    if missing:
        unavailable['ratio'] = '; '.join(key + ': ' + unavailable.get(key, 'samples were not supplied')
                                         for key in missing)
    elif moments['generated'].sample_shape != moments['reference'].sample_shape:
        unavailable['ratio'] = 'Diversity ratio requires matching generated/reference sample shapes'
    elif values['reference_rms'] == 0:
        unavailable['ratio'] = 'Diversity ratio requires nonzero reference sample spread'
    else:
        ratio = values['generated_rms'] / values['reference_rms']
        if math.isfinite(ratio):
            values['ratio'] = ratio
        else:
            unavailable['ratio'] = 'Diversity ratio is not finite'
    return {'metrics': values, 'unavailable': unavailable}


def batch_diversity(generated, reference, *, pool_size=None):
    """Measure one observation without throwing for undefined measurements.

    Returns ``{'metrics': {name: finite_float}, 'unavailable': {name: reason}}``
    for generated_rms, reference_rms and ratio. Each absolute spread remains
    available when the other tensor is unsupported, too small, or a different
    sample shape. A constant reference has RMS zero and an unavailable ratio.
    Sample counts may differ; reference samples are never repeated to match N.
    """
    try:
        _pool_size(pool_size)
    except ValueError as exc:
        return {'metrics': {}, 'unavailable': {key: str(exc) for key in _STATISTICS}}
    moments, errors = {}, {}
    for name, samples in (('generated', generated), ('reference', reference)):
        moments[name] = DiversityMoments(pool_size)
        try:
            moments[name].update(samples)
        except ValueError as exc:
            errors[name + '_rms'] = str(exc)
    return _results(moments, errors)


class SampleDiversity:
    """Snapshot scalar factory for absolute diversity or its reference ratio."""
    def __init__(self, statistic='ratio', pool_size=None):
        if statistic not in _STATISTICS:
            raise ValueError('Diversity statistic must be generated_rms, reference_rms, or ratio')
        self.statistic, self.pool_size = statistic, _pool_size(pool_size)

    def describe(self):
        pooling = 'all coordinates' if self.pool_size is None else f'images average-pooled to {self.pool_size}x{self.pool_size}'
        return {'kind': 'scalar', 'label': 'Sample diversity ' + self.statistic.replace('_', ' '),
                'unit': 'ratio' if self.statistic == 'ratio' else 'sample_units', 'direction': 'none',
                'description': f'Distinct-pair RMS over {pooling}; measures variation, not realism or semantic coverage. Noise can score highly.'}

    def evaluate(self, *, batches, context):
        names = ('generated', 'reference') if self.statistic == 'ratio' else (self.statistic.removesuffix('_rms'),)
        moments = {name: DiversityMoments(self.pool_size) for name in names}
        expected = context.get('sample_count') if isinstance(context, Mapping) else None
        if expected is not None and (type(expected) is not int or expected < 2):
            raise ValueError('Diversity declared sample_count must be an integer of at least two')
        for batch in batches:
            if not isinstance(batch, Mapping) or any(name not in batch for name in names):
                raise ValueError('Diversity evaluation batches must supply ' + ', '.join(names))
            for name in names:
                moments[name].update(batch[name])
        for name, accumulator in moments.items():
            if expected is not None and accumulator.count != expected:
                raise ValueError(f'Diversity {name} count {accumulator.count} differs from declared sample_count {expected}')
        if self.statistic != 'ratio':
            return moments[names[0]].rms()
        result = _results(moments)
        if 'ratio' not in result['metrics']:
            raise ValueError(result['unavailable']['ratio'])
        return result['metrics']['ratio']
