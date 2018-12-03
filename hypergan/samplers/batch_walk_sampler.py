import numpy as np
import tensorflow as tf
from PIL import Image
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
import time

class BatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z_start = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = []
        self.step_count = 30
        self.target = None
        #self.style_t = gan.styleb.sample
        #self.style_v = gan.session.run(self.style_t)


    def regenerate_steps(self):
        gan = self.gan
        z_t = gan.latent.z
        inputs_t = gan.inputs.x

        s=np.shape(gan.session.run(gan.latent.z))
        bs = 32
        if self.z_start is None:
            self.z_start = gan.session.run(gan.latent.z)[0]
            targ = gan.session.run(gan.latent.z)[0]
        else:
            self.z_start = self.steps[-1]
            targ = gan.session.run(gan.latent.z)[0]
        mask = np.linspace(0., 1., num=bs)
        z = np.tile(np.expand_dims(np.reshape(self.z_start, [-1]), axis=0), [bs,1])
        targ = np.tile(np.expand_dims(np.reshape(targ, [-1]), axis=0), [bs,1])
        mask = np.tile(np.expand_dims(mask, axis=1), [1, np.shape(targ)[1]])
        z_interp = np.multiply(1.0-mask,z) + mask*targ
        z_interp = np.reshape(z_interp, [bs, -1])
        #self.z_start = targ[-1]
        return z_interp

    def sample(self, path, save_samples):
        gan = self.gan
        z_t = gan.latent.z
        inputs_t = gan.inputs.x
        self.step+=1

        if(self.step >= len(self.steps)):
            self.steps = self.regenerate_steps()
            self.step=0
        z = self.steps[self.step]


        with gan.session.as_default():

            z = np.reshape(z, [1, -1])
            sample_data = gan.session.run(gan.generator.sample, feed_dict={z_t: z})
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
