import numpy as np
import tensorflow as tf
from PIL import Image
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
import time

class BatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = 30
        self.target = None

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        inputs_t = gan.inputs.x

        if self.z is None:
            self.input = gan.session.run(gan.inputs.x)

            self.target = gan.uniform_encoder.sample.eval()[0]
        else:
            self.target = self.z
        self.z = gan.uniform_encoder.sample.eval()[0]


        g=tf.get_default_graph()
        s=np.shape(gan.uniform_encoder.sample.eval())
        bs = s[0]
        mask = np.linspace(0., 1., num=bs)
        self.z = np.tile(np.expand_dims(np.reshape(self.z, [-1]), axis=0), [bs,1])
        targ = np.tile(np.expand_dims(np.reshape(self.target, [-1]), axis=0), [bs,1])
        print("SHAPES", np.shape(mask), np.shape(self.z), np.shape(targ))
        mask = np.tile(np.expand_dims(mask, axis=1), [1, np.shape(targ)[1]])
        z_interp = np.multiply(mask,self.z) + (1-mask)*targ
        z_interp = np.reshape(z_interp, s)
        print("Z_I", np.shape(z_interp), np.shape(self.z))
        self.z = z_interp[-1]
        with g.as_default():
            tf.set_random_seed(1)
            return {
                'generator': gan.session.run(gan.encoder.sample, feed_dict={z_t: z_interp, inputs_t: self.input})
            }

    def sample(self, path, save_samples):
        gan = self.gan

        with gan.session.as_default():

            sample = self._sample()

            data = sample['generator']
            for i in range(np.shape(data)[0]):
                sample_data = data[i:i+1]
                self.plot(sample_data, path, save_samples)
                time.sleep(0.018)

            return []

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
