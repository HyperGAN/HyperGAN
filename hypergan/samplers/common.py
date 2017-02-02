import numpy as np
from scipy.misc import imsave

def plot(config, image, file):
    """ Plot an image."""
    image = np.squeeze(image)
    imsave(file, image)
