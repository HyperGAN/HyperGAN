from hypergan.samplers.base_sampler import BaseSampler
import numpy as np
import tensorflow as tf

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.g_t = self.replace_none(gan.generator.sample)
        self.rows = 4
        self.columns = 8

    def _sample(self):
        gan = self.gan
        z_t = gan.latent.sample
        inputs_t = gan.inputs.x
        needed = int(self.rows*self.columns / gan.batch_size())

        if self.z is None:
            self.z = [gan.latent.sample.eval() for i in range(needed)]
            self.z = np.reshape(self.z, [self.rows*self.columns, -1])

        z = self.z
        gs = []
        for i in range(int(needed)):
            zi = z[i*gan.batch_size():(i+1)*gan.batch_size()]
            g = gan.session.run(self.g_t, feed_dict={z_t: zi})
            gs.append(g)
        g = np.hstack(gs)
        xshape = gan.ops.shape(gan.inputs.x)
        g = np.reshape(gs, [self.rows, self.columns, xshape[1], xshape[2], xshape[3]])
        g = np.concatenate(g, axis=1)
        g = np.concatenate(g, axis=1)
        g = np.expand_dims(g, axis=0)

        return {
            'generator': g
        }

