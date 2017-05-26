import numpy as np
from PIL import Image

def plot(config, image, filename):
    """ Plot an image."""
    image = np.squeeze(image)
    # Scale to 0..255.
    imin, imax = image.min(), image.max()
    image = (image - imin) * 255. / (imax - imin) + .5
    Image.fromarray(image.astype(np.uint8)).save(filename)
