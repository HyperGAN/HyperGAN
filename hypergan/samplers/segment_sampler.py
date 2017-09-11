from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class SegmentSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.x_v = None
        self.z_v = None
        self.created = False
    def _sample(self):
        gan = self.gan
        x_t = gan.inputs.x
        g_t = gan.autoencode_x
        z_t = gan.z_hat

        g1x_t = gan.generator.g1x
        g2x_t = gan.generator.g2x
        mask_t = gan.autoencode_mask

        sess = gan.session
        config = gan.config
        if(not self.created):
            self.x_v, self.z_v = sess.run([x_t, z_t])
            self.created=True

        gens = sess.run(
                [
                    x_t,
                    mask_t,
                    gan.x_hat,
                    g1x_t,
                    g2x_t,
                    g_t

                ], {
                    x_t: self.x_v,
                    z_t: self.z_v
                })

        stacks = []
        bs = gan.batch_size() // 2
        width = min(gan.batch_size(), 8)
        for gen in gens:
            for i in range(1):
                stacks.append([gen[i*width+j] for j in range(width)])

        #[print(np.shape(s)) for s in stacks]
        images = np.vstack(stacks)
        return {'generator':images}


