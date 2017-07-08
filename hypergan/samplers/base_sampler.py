import numpy as np
from PIL import Image
from hypergan.viewer import GlobalViewer

class BaseSampler:
    def __init__(self, gan, samples_per_row=8, session=None):
        self.gan = gan
        self.samples_per_row = samples_per_row

    def _sample(self):
        raise "raw _sample method called.  You must override this"

    def sample(self, path, save_samples):
        gan = self.gan

        with gan.session.as_default():

            sample = self._sample()

            data = sample['generator']

            width = min(gan.batch_size(), self.samples_per_row)
            stacks = [np.hstack(data[i*width:i*width+width]) for i in range(gan.batch_size()//width)]
            sample_data = np.vstack(stacks)
            self.plot(sample_data, path, save_samples)
            sample_name = 'generator'
            samples = [[sample_data, sample_name]]

            return [{'image':path, 'label':'sample'} for sample_data, sample_filename in samples]

    def plot(self, image, filename, save_sample):
        """ Plot an image."""
        image = np.minimum(image, 1)
        image = np.maximum(image, -1)
        image = np.squeeze(image)
        # Scale to 0..255.
        imin, imax = image.min(), image.max()
        image = (image - imin) * 255. / (imax - imin) + .5
        image = image.astype(np.uint8)
        if save_sample:
            try:
                Image.fromarray(image).save(filename)
            except Exception as e:
                print("Warning: could not sample to ", filename, ".  Please check permissions and make sure the path exists")
                print(e)
        GlobalViewer.update(image)
