"""Held-out diagnostics, independent of the optimized colorization objectives."""
import math


def _rgb(value):
    import torch
    if (not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != 3
            or value.numel() == 0 or not torch.isfinite(value).all()
            or value.min() < -1 or value.max() > 1):
        raise ValueError('Colorization metrics require finite nonempty RGB NCHW tensors in [-1,1]')
    return value.detach().to(device='cpu', dtype=torch.float64).add(1).div(2)


def _luma(rgb):
    return rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114


class ChromaDistributionDistance:
    """Sliced Wasserstein-1 approximation using fixed projected chroma bins.

    Fixed YCbCr chroma projections preserve joint color information. Averaging
    integrated empirical CDF differences estimates transport distance without
    storing every pixel. All pixels count, including the white background.
    """
    def __init__(self, bins=128, projections=16):
        if type(bins) is not int or not 8 <= bins <= 512:
            raise ValueError('Chroma bins must be an integer between 8 and 512')
        if type(projections) is not int or not 2 <= projections <= 64:
            raise ValueError('Chroma projections must be an integer between 2 and 64')
        self.bins, self.projections = bins, projections

    def describe(self):
        return {'kind': 'scalar', 'label': 'Held-out chroma distribution distance',
                'unit': 'normalized_chroma', 'direction': 'minimize',
                'description': 'Fixed-projection histogram approximation to chroma sliced Wasserstein-1; pixel weighted including background, no spatial or semantic guarantee.'}

    def evaluate(self, *, batches, context):
        import torch
        histograms = torch.zeros((2, self.projections, self.bins), dtype=torch.float64)
        bound = math.sqrt(.5)
        for batch in batches:
            for i, name in enumerate(('generated', 'reference')):
                rgb = _rgb(batch[name])
                y = _luma(rgb)[:, 0]
                cb = (rgb[:, 2] - y) / 1.772
                cr = (rgb[:, 0] - y) / 1.402
                for p in range(self.projections):
                    angle = math.pi * p / self.projections
                    projected = cb * math.cos(angle) + cr * math.sin(angle)
                    histograms[i, p] += torch.histc(projected, bins=self.bins, min=-bound, max=bound)
        totals = histograms.sum(-1, keepdim=True)
        if (totals == 0).any():
            raise ValueError('Chroma distribution requires nonempty generated/reference samples')
        cdf = (histograms / totals).cumsum(-1)
        return float((cdf[0] - cdf[1]).abs().sum(-1).mean() * (2 * bound / self.bins))


class GrayscaleStructureDistance:
    """Paired luminance-edge MAE; compares location/shape without color matching."""
    def describe(self):
        return {'kind': 'scalar', 'label': 'Held-out grayscale edge discrepancy',
                'unit': 'normalized_luminance_gradient', 'direction': 'minimize',
                'description': 'Mean absolute difference of paired horizontal/vertical luminance differences; independent of training RGB reconstruction and sensitive to shape placement.'}

    def evaluate(self, *, batches, context):
        total, count = 0., 0
        for batch in batches:
            generated, reference = (_luma(_rgb(batch[name])) for name in ('generated', 'reference'))
            if generated.shape != reference.shape or min(generated.shape[-2:]) < 2:
                raise ValueError('Structure metric requires paired equal shapes with height/width at least two')
            for axis in (-1, -2):
                error = (generated.diff(dim=axis) - reference.diff(dim=axis)).abs()
                total += float(error.sum())
                count += error.numel()
        if not count:
            raise ValueError('Structure metric requires nonempty paired samples')
        return total / count


class RepeatedConditionChromaDiversity:
    """Mean pairwise chroma difference across consecutive repeated conditions.

    Requires ColorizationData(shuffle=False, repeats=K) and sample_count divisible
    by K. Grouping is validated using the complete reference RGB image; groups
    may straddle batches. High diversity alone is not evidence of good color.
    """
    def __init__(self, repeats=4):
        if type(repeats) is not int or not 2 <= repeats <= 16:
            raise ValueError('Repeated-condition diversity requires 2 to 16 repeats')
        self.repeats = repeats

    def describe(self):
        return {'kind': 'scalar', 'label': 'Held-out repeated-condition chroma diversity',
                'unit': 'normalized_chroma', 'direction': 'none',
                'description': 'Mean pairwise absolute chroma difference across independent samples of each repeated grayscale condition; higher alone is not better quality.'}

    def evaluate(self, *, batches, context):
        import torch
        group, condition = [], None
        total, pairs = 0., 0
        for batch in batches:
            generated, reference = (_rgb(batch[name]) for name in ('generated', 'reference'))
            if generated.shape != reference.shape:
                raise ValueError('Repeated-condition diversity requires paired equal shapes')
            for output, source in zip(generated, reference):
                if not group:
                    condition = source.clone()
                elif not torch.equal(source, condition):
                    raise ValueError('Repeated-condition reference differs inside a group; configure data repeats')
                y = _luma(output[None])[0, 0]
                chroma = torch.stack(((output[2] - y) / 1.772, (output[0] - y) / 1.402))
                for previous in group:
                    total += float((chroma - previous).abs().mean())
                    pairs += 1
                group.append(chroma)
                if len(group) == self.repeats:
                    group = []
        if group:
            raise ValueError('Repeated-condition diversity has an incomplete group; sample_count must be divisible by repeats')
        if not pairs:
            raise ValueError('Repeated-condition diversity requires nonempty repeated groups')
        return total / pairs
