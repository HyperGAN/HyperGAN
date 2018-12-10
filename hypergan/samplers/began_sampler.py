from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class BeganSampler(BaseSampler):
    def __init__(self, gan, samples_per_row):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.x_v = None
        self.z_v = None
        self.created = False


    def _sample(self):
        gan = self.gan
        config = gan.config
        sess = gan.session
        x_t = gan.inputs.x
        z_t = gan.encoder.sample
        if(not self.created):
            self.x_v, self.z_v = sess.run([x_t, z_t])
            self.created=True
        g_t = gan.generator.sample
        rx_t = gan.discriminator.reconstruction
        rx_v, g_v = sess.run([rx_t, g_t], {x_t: self.x_v, z_t: self.z_v})
        stacks = []
        bs = gan.batch_size() // 2
        width = self.samples_per_row
        stacks.append(self.x_v)
        stacks.append(rx_v)
        stacks.append(g_v)

        images = np.vstack(stacks)
        return { 'generator':images}


    def sample(self, path, save_samples=False):

        self.plot(images, path, save_samples)
        return [{'image': path, 'label': 'tiled x sample'}]

