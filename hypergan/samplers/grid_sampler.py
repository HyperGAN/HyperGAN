from hypergan.samplers.base_sampler import BaseSampler
import numpy as np
import tensorflow as tf


class GridSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.x = gan.session.run(gan.inputs.x)
        batch = self.x.shape[0]
        self.x = np.reshape(self.x[0], [1, self.x.shape[1], self.x.shape[2], self.x.shape[3]])
        self.x = np.tile(self.x, [batch,1,1,1])

    def _sample(self):
        gan = self.gan
        z_t = gan.latent.z
        #This isn't doing any gridlike stuff.  Need to feed this into feed dict(also check size)
        y = np.linspace(0,1, 6)

        z = np.mgrid[-0.999:0.999:0.6, -0.999:0.999:0.26].reshape(2,-1).T
        z = np.reshape(z, [32,2])
        #z = np.mgrid[-0.499:0.499:0.3, -0.499:0.499:0.13].reshape(2,-1).T
        #z = np.mgrid[-0.299:0.299:0.15, -0.299:0.299:0.075].reshape(2,-1).T
        needed = 32 / gan.batch_size()
        gs = []
        for i in range(int(needed)):
            zi = z[i*gan.batch_size():(i+1)*gan.batch_size()]
            g = gan.session.run(gan.generator.sample, feed_dict={z_t: zi, gan.inputs.x: self.x})
            gs.append(g)
        g = np.hstack(gs)
        xshape = gan.ops.shape(gan.inputs.x)
        g = np.reshape(gs, [4, 8, xshape[1], xshape[2], xshape[3]])
        g = np.concatenate(g, axis=1)
        g = np.concatenate(g, axis=1)
        g = np.expand_dims(g, axis=0)
        x_hat = gan.session.run(gan.autoencoded_x, feed_dict={gan.inputs.x: self.x})
        #e = gan.session.run(gan.encoder.sample, feed_dict={gan.inputs.x: g})

        return {
            'generator':g
        }
