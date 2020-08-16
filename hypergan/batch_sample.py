import io
import numpy as np

class BatchSample:
    def __init__(self, batch_size, samples):
        stacks = []
        for key, data in samples:
            slots = batch_size
            slots = min(slots, np.shape(data)[0])
            data = data.cpu().permute(0,2,3,1).detach().numpy()
            stacks += [np.hstack(data[i*slots:i*slots+slots]) for i in range(np.shape(data)[0]//slots)]
        sample_data = np.vstack(stacks)
        sample_name = 'generator'
        self.sample = sample_data

    def to_images(format='png'):
        bytes_io = io.BytesIO()
        self.plot(self.sample, bytes_io, False)
        bytes_io.rewind()
        return [bytes_io]

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

        #GlobalViewer.update(self.gan, image)
        return image

    def plot_image(self, image, filename, save_sample, regularize=True):
        """ Plot an image from an external source."""
        if np.shape(image)[2] == 4:
            fmt = "RGBA"
        else:
            fmt = "RGB"
        if save_sample:
            try:
                Image.fromarray(image, fmt).save(filename)
            except Exception as e:
                print("Warning: could not sample to ", filename, ".  Please check permissions and make sure the path exists")
                print(e)

        #GlobalViewer.update(self.gan, image)
        return image
