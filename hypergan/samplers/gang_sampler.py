from hypergan.samplers.base_sampler import BaseSampler

from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.segment_sampler import SegmentSampler
import tensorflow as tf
import numpy as np
import hypergan as hg

class GangSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.z = None
        self.samples = 3

    def sample(self, path, sample_to_file):
        gan = self.gan

        sess = gan.session
        config = gan.config
        if self.z is None:
            self.z = [sess.run(gan.uniform_encoder.sample) for i in range(self.samples)]

        current_g = sess.run(gan.trainer.all_g_vars)
        
        stacks = []
        def _samples():
            n = 3
            cs = []
            for i in range(self.samples):
                cs.append(sess.run(gan.generator.sample,{gan.uniform_encoder.sample: self.z[i]}))
            return np.vstack(cs)

        stacks.append(_samples())
        for sg in gan.trainer.sgs:
            gan.trainer.assign_g(sg)
            stacks.append(_samples())
        for i in range((gan.trainer.config.nash_memory_size or 10) - len(stacks)):
            stacks.append(_samples())

        gan.trainer.assign_g(current_g)

        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, sample_to_file)
        return [{'image': path, 'label': 'tiled x sample'}]
