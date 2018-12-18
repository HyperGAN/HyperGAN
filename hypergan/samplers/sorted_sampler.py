from hypergan.samplers.base_sampler import BaseSampler

from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.segment_sampler import SegmentSampler
import tensorflow as tf
import numpy as np
import hypergan as hg

class SortedSampler(BaseSampler):
    def __init__(self, gan):
        BaseSampler.__init__(self, gan)
        self.xs = None
        self.samples = 10
        self.display_count = 5

    def sample(self, path, sample_to_file):
        gan = self.gan

        sess = gan.session
        config = gan.config
        if self.xs is None:
            self.xs = [sess.run(gan.fitness_inputs()) for i in range(self.samples)]

        current_g = sess.run(gan.trainer.all_g_vars)
        
        stacks = []
        def _samples():
            cs = []
            for i in range(self.samples):
                ts = gan.fitness_inputs()
                vs = self.xs[i]
                feed_dict = {}
                for t,v in zip(ts, vs):
                    feed_dict[t]=v
                cs.append(sess.run(gan.generator.sample,feed_dict))
                #cs.append(sess.run(gan.autoencoded_x,feed_dict))
            return cs
        
        gs = _samples()
        priority = []
        for i, sample in zip(range(self.samples), gs):
            priority.append(gan.session.run(gan.loss.d_fake, {gan.generator.sample: sample}))

        sorted_sgs = [[p, v] for p,v in zip(priority, gs)]
        sorted_sgs.sort(key=lambda x: -x[0])
        sorted_sgs = [s[1] for s in sorted_sgs]

        top = sorted_sgs[:self.display_count]
        end = sorted_sgs[(len(gs)-self.display_count):]

        print("TOP", np.shape(top), "end", np.shape(end))
        stacks.append(np.vstack(top))
        stacks.append(np.vstack(end))
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, sample_to_file)
        return [{'image': path, 'label': 'tiled x sample'}]
