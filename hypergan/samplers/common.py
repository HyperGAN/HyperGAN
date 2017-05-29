import numpy as np
from PIL import Image
from hypergan.samplers.viewer import GlobalViewer

def plot(config, image, filename):
    """ Plot an image."""
    image = np.minimum(image, 1)
    image = np.maximum(image, -1)
    image = np.squeeze(image)
    # Scale to 0..255.
    imin, imax = image.min(), image.max()
    image = (image - imin) * 255. / (imax - imin) + .5
    image = image.astype(np.uint8)
    Image.fromarray(image).save(filename)
    GlobalViewer.update(image)
