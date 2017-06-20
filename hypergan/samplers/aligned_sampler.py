from hypergan.samplers.base_sampler import BaseSampler

import tensorflow as tf
import numpy as np

class AlignedSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.xa_v = None
        self.xb_v = None
        self.created = False

    def sample(self, path, sample_to_file):
        gan = self.gan
        cyca = gan.cyca
        cycb = gan.cycb
        xa_t = gan.inputs.xa
        xba_t = gan.xba
        xab_t = gan.xab
        xb_t = gan.inputs.xb

        sess = gan.session
        config = gan.config
        if(not self.created):
            self.xa_v, self.xb_v = sess.run([xa_t, xb_t])
            self.created = True

        xab_v, xba_v, samplea, sampleb = sess.run([xab_t, xba_t, cyca, cycb], {xa_t: self.xa_v, xb_t: self.xb_v})
        stacks = []
        bs = gan.batch_size() // 2
        width = 5
        for i in range(1):
            stacks.append([self.xa_v[i*width+width+j] for j in range(width)])
        for i in range(1):
            stacks.append([xab_v[i*width+width+j] for j in range(width)])
        for i in range(1):
            stacks.append([samplea[i*width+width+j] for j in range(width)])
        for i in range(1):
            stacks.append([self.xb_v[i*width+width+j] for j in range(width)])
        for i in range(1):
            stacks.append([xba_v[i*width+width+j] for j in range(width)])
        for i in range(1):
            stacks.append([sampleb[i*width+width+j] for j in range(width)])

        #[print(np.shape(s)) for s in stacks]
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, sample_to_file)
        return [{'image': path, 'label': 'tiled x sample'}]
