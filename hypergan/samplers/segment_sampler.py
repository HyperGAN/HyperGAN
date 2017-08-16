from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class SegmentSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.x_v = None
        self.z_v = None
        self.created = False

    def sample(self, path, save_samples=False):
        gan = self.gan
        x_t = gan.inputs.x
        g_t = gan.generator.sample
        z_t = gan.encoder.sample

        g1x_t = gan.generator.g1x
        g2x_t = gan.generator.g2x
        mask_t = gan.generator.mask

        sess = gan.session
        config = gan.config
        if(not self.created):
            self.x_v = sess.run(x_t)
            self.created=True

        gens = sess.run([gan.inputs.x, g_t, mask_t, g1x_t, g2x_t, gan.generator.g1.sample, gan.generator.g2.sample], {x_t: self.x_v})
        stacks = []
        bs = gan.batch_size() // 2
        width = 8
        for gen in gens:
            for i in range(1):
                stacks.append([gen[i*width+width+j] for j in range(width)])

        #[print(np.shape(s)) for s in stacks]
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, save_samples)
        return [{'image': path, 'label': 'tiled x sample'}]

