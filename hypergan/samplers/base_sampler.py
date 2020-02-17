import numpy as np
from PIL import Image
from hypergan.viewer import GlobalViewer

class BaseSampler:
    def __init__(self, gan, samples_per_row=8, session=None):
        self.gan = gan
        self.samples_per_row = samples_per_row

    def _sample(self):
        raise "raw _sample method called.  You must override this"

    def compatible_with(gan):
        return False

    def sample(self, path, save_samples):
        gan = self.gan

        sample = self._sample()

        stacks = []
        for key, data in sample:
            data = data.cpu().permute(0,2,3,1).detach().numpy()
            slots = min(gan.batch_size(), self.samples_per_row)
            slots = min(slots, np.shape(data)[0])
            stacks += [np.hstack(data[i*slots:i*slots+slots]) for i in range(np.shape(data)[0]//slots)]
        sample_data = np.vstack(stacks)
        image = self.plot(sample_data, path, save_samples)
        sample_name = 'generator'
        samples = [[sample_data, sample_name]]

        return [{'image':path, 'label':'sample'} for sample_data, sample_filename in samples]

    def plot(self, image, filename, save_sample, regularize=True):
        """ Plot an image."""
        if regularize:
            image = np.minimum(image, 1)
            image = np.maximum(image, -1)
        image = np.squeeze(image)
        if np.shape(image)[2] == 4:
            fmt = "RGBA"
        else:
            fmt = "RGB"
        # Scale to 0..255.
        imin, imax = -1.0, 1.0
        image = (image - imin) * 255. / (imax - imin) + .5
        image = image.astype(np.uint8)
        if save_sample:
            try:
                Image.fromarray(image, fmt).save(filename)
            except Exception as e:
                print("Warning: could not sample to ", filename, ".  Please check permissions and make sure the path exists")
                print(e)
        GlobalViewer.update(self.gan, image)
        return image
