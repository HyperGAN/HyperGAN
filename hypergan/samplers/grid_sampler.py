from hypergan.samplers.base_sampler import BaseSampler
import numpy as np


class GridSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        #This isn't doing any gridlike stuff.  Need to feed this into feed dict(also check size)
        y = np.linspace(0,1, 6)

        z = np.mgrid[-0.999:0.999:0.6, -0.999:0.999:0.26].reshape(2,-1).T
        z = np.reshape(z, gan.ops.shape(z_t))
        #z = np.mgrid[-0.499:0.499:0.3, -0.499:0.499:0.13].reshape(2,-1).T
        #z = np.mgrid[-0.299:0.299:0.15, -0.299:0.299:0.075].reshape(2,-1).T
        g = gan.session.run(gan.generator.sample, feed_dict={z_t: z})
        e = gan.session.run(gan.encoder.sample, feed_dict={gan.inputs.xa: g})

        return {
            'generator': np.hstack([g,e])
        }
