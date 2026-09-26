"""The user's own loss, metric, evaluation and sampler: plain functions."""
import torch


def code_norm(code):
    """A loss on the encoder's code (keeps it small)."""
    return code.pow(2).mean()


def g_over_d(g_loss, d_loss):
    """A metric: a function of values every update computes anyway."""
    return g_loss / d_loss


def radius_gap(batches):
    """An evaluation on a model snapshot and holdout data: |mean radius(generated) - mean radius(real)|."""
    generated, reference = [], []
    for batch in batches:
        generated.append(batch['generated'].norm(dim=1))
        reference.append(batch['reference'].norm(dim=1))
    return float((torch.cat(generated).mean() - torch.cat(reference).mean()).abs())


def scatter(samples, *, step=None, bins=11, extent=2.0):
    """A sampler: maps (N, 2) generator output to something a person can look at."""
    points = samples.detach().float().reshape(len(samples), -1)[:, :2]
    grid = torch.zeros(bins, bins, dtype=torch.int64)
    cells = ((points.clamp(-extent, extent - 1e-6) + extent) / (2 * extent) * bins).long()
    for x, y in cells.tolist():
        grid[bins - 1 - y, x] += 1
    shades = ' .:*#'
    rows = [''.join(shades[min(int(v), 4)] for v in row) for row in grid.tolist()]
    return {'kind': 'points2d', 'step': step, 'count': len(points),
            'mean': [round(v, 4) for v in points.mean(0).tolist()],
            'radius': round(float(points.norm(dim=1).mean()), 4), 'ascii': rows}
