"""Small ordinary Python metric factories; no implicit data or weight downloads."""


class ScalarRatio:
    """A primitive transform useful for testing explicit detached scalar bindings."""
    def describe(self):
        return {'kind': 'scalar', 'label': 'Scalar ratio', 'unit': 'ratio', 'direction': 'none'}

    def evaluate(self, *, numerator, denominator, context):
        if denominator == 0:
            raise ValueError('Ratio denominator is zero')
        return numerator / denominator


def _rgb(value, low, high):
    import torch
    if (value.ndim != 4 or value.shape[1] != 3 or not torch.isfinite(value).all()
            or value.numel() == 0 or value.min() < low or value.max() > high):
        raise ValueError('Color metrics require finite NCHW RGB tensors inside the declared range')
    return ((value.double() - low) / (high - low)).movedim(1, 0).reshape(3, -1)


class ColorMomentDistance:
    """Pixel-weighted RGB mean/spread distance, not a semantic quality measure."""
    def __init__(self, statistic='mean', low=-1.0, high=1.0, color_space='rgb'):
        import math
        if statistic not in ('mean', 'spread') or color_space != 'rgb':
            raise ValueError('Color moments support statistic mean/spread and explicit RGB only')
        if type(low) not in (int, float) or type(high) not in (int, float) or not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Color range must have finite low < high')
        self.statistic, self.low, self.high = statistic, low, high

    def describe(self):
        return {'kind': 'scalar', 'label': f'RGB {self.statistic} distance', 'unit': 'normalized_rgb',
                'direction': 'minimize', 'description': 'Mean absolute per-channel difference, weighted by pixels; not semantic or spatial fidelity.'}

    def evaluate(self, *, batches, context):
        import torch
        sums, squares, counts = [None, None], [None, None], [0, 0]
        for batch in batches:
            for i, name in enumerate(('generated', 'reference')):
                value = _rgb(batch[name], self.low, self.high)
                total, squared = value.sum(dim=1), value.square().sum(dim=1)
                sums[i] = total if sums[i] is None else sums[i] + total
                squares[i] = squared if squares[i] is None else squares[i] + squared
                counts[i] += value.shape[1]
        if not all(counts):
            raise ValueError('Color moments require nonempty generated and reference samples')
        means = [sums[i] / counts[i] for i in range(2)]
        values = means if self.statistic == 'mean' else [(squares[i] / counts[i] - means[i].square()).clamp_min(0).sqrt() for i in range(2)]
        return float((values[0] - values[1]).abs().mean())


class ColorHistogramDifference:
    """Absolute difference of pooled RGB-bin probabilities in a fixed range."""
    def __init__(self, bins=32, low=-1.0, high=1.0, color_space='rgb'):
        ColorMomentDistance(low=low, high=high, color_space=color_space)
        if type(bins) is not int or not 1 <= bins <= 512:
            raise ValueError('Color histogram bins must be between 1 and 512')
        self.bins, self.low, self.high = bins, low, high

    def describe(self):
        return {'kind': 'histogram', 'label': 'RGB histogram probability difference',
                'unit': 'probability_difference', 'direction': 'none',
                'description': 'Absolute generated/reference pooled RGB-bin probability differences; pixel weighted, no spatial correspondence.'}

    def evaluate(self, *, batches, context):
        import torch
        histograms = [None, None]
        for batch in batches:
            for i, name in enumerate(('generated', 'reference')):
                value = _rgb(batch[name], self.low, self.high)
                histogram = torch.histc(value, bins=self.bins, min=0, max=1)
                histograms[i] = histogram if histograms[i] is None else histograms[i] + histogram
        if any(value is None or value.sum() == 0 for value in histograms):
            raise ValueError('Color histograms require nonempty samples')
        value = (histograms[0] / histograms[0].sum() - histograms[1] / histograms[1].sum()).abs()
        return {'edges': [index / self.bins for index in range(self.bins + 1)], 'counts': value.tolist()}
