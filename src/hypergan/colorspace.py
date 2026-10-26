"""The one luma definition shared by colorization data, models and metrics."""

# Recorded verbatim in colorization manifests; changing it invalidates them.
LUMA_FORMULA = '0.299*R + 0.587*G + 0.114*B'


def luma(rgb):
    """ITU-R BT.601 luma of an NCHW RGB tensor, keeping a one-channel dim."""
    return rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114
