import numpy as np
from scipy.misc import imsave

def plot(config, image, file):
    """ Plot an image."""
    image = np.minimum(image, 1)
    image = np.maximum(image, -1)
    image = np.squeeze(image)
    imsave(file, image)
