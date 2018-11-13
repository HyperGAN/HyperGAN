from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class SegmentSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.x_v = None
        self.z_v = None
        self.created = False
        self.mask_t = None
    def _sample(self):
        gan = self.gan
        x_t = gan.inputs.x
        g_t = gan.autoencoded_x
        z_t = gan.uniform_distribution.sample

        g1x_t = gan.generator.g1x
        g2x_t = gan.generator.g2x
        g3x_t = gan.generator.g3x


        if self.mask_t is None:
            self.mask_t = (gan.generator.mask-0.5)*2
        sess = gan.session
        config = gan.config
        if(not self.created):
            self.x_v = sess.run(x_t)
            self.created=True

        gens = sess.run(
                [
                    gan.inputs.x,
                    self.mask_t,
                    g_t,
                    g1x_t,
                    g2x_t,
                    g3x_t
                ], {
                    x_t: self.x_v
                })

        stacks = []
        bs = gan.batch_size() // 2
        width = min(gan.batch_size(), 8)
        for gen in gens:
            for i in range(1):
                stacks.append([gen[i*width+j] for j in range(width)])

        images = np.vstack(stacks)
        return {'generator':images}


