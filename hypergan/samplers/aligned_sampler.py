from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent

import tensorflow as tf
import numpy as np

class AlignedSampler(BaseSampler):
    def __init__(self, gan, samples_per_row):
        BaseSampler.__init__(self, gan, samples_per_row = samples_per_row)
        self.xa_v = None
        self.xb_v = None
        self.created = False
        self.x_cache = None
        self.latent_cache = None

    def sample(self, path, sample_to_file):
        gan = self.gan

        sess = gan.session
        config = gan.config
        if hasattr(gan, 'x_hats'):
            xs, *x_hats = sess.run([gan.inputs.xs[0]]+[_x.sample for _x in gan.x_hats])
        elif hasattr(gan, "generators_cache"):
            feed_dict = {}
            if self.x_cache is None:
                self.x_cache = sess.run(gan.inputs.xs)
                self.latent_cache = sess.run(gan.latent.sample)
            
            for i, x in enumerate(gan.inputs.xs):
                feed_dict[x] = self.x_cache[i]
            xs = sess.run(gan.inputs.xs, feed_dict)
            feed_dict[gan.latent.sample]=self.latent_cache
            x_hats = sess.run([g.sample for g in gan.generators_cache.values()], feed_dict)
        else:
            raise ValidationException("Unknown alignment gan type")

        stacks = []
        bs = gan.batch_size() // 2
        width = min(gan.batch_size(), 8)
        for i in range(len(gan.inputs.xs)):
            print(np.shape(xs))
            stacks.append([xs[i][j] for j in range(width)])
        print(np.shape(x_hats), "XH")
        for i in range(len(x_hats)):
            stacks.append([x_hats[i][j] for j in range(width)])

        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, sample_to_file)
        return [{'image': path, 'label': 'tiled x sample'}]
