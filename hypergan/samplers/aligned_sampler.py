from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class AlignedSampler(BaseSampler):
    def __init__(self, gan, samples_per_row):
        BaseSampler.__init__(self, gan, samples_per_row = samples_per_row)
        self.xa_v = None
        self.xb_v = None
        self.created = False

    def compatible_with(gan):
        if hasattr(gan.inputs, 'xa') and \
            hasattr(gan.inputs, 'xb') and \
            hasattr(gan, 'cyca'):
            return True
        return False

    def sample(self, path, sample_to_file):
        gan = self.gan

        sess = gan.session
        config = gan.config

        xs, x_hats = sess.run([gan.inputs.xs[0]]+[_x.sample for _x in gan.x_hats])
        x_hats = [x_hats]
        print("---")
        print(gan.inputs.xs[0])

        stacks = []
        bs = gan.batch_size() // 2
        width = min(gan.batch_size(), 8)
        for i in range(1):
            stacks.append([xs[i*width+j] for j in range(width)])
        for x_h in x_hats:
            for i in range(1):
                stacks.append([x_h[i*width+j] for j in range(width)])

        print([np.shape(s) for s in stacks])
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, sample_to_file)
        return [{'image': path, 'label': 'tiled x sample'}]
