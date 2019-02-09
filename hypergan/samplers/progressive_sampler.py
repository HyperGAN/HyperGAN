from hypergan.samplers.base_sampler import BaseSampler

from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.segment_sampler import SegmentSampler
import tensorflow as tf
import numpy as np
import hypergan as hg
from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.generators.segment_generator import SegmentGenerator

z = None
x = None
class IdentitySampler(BaseSampler):
    def __init__(self, gan, node, samples_per_row=8, x=None, z=None):
        self.node = node
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.x = None

    def _sample(self,i,n):
        gan = self.gan
        z_t = gan.latent.sample
        x_t = gan.inputs.x

        global z
        if z is None:
            z = []
            for i in range(n):
                z.append(gan.session.run(z_t))
        if self.x is None:
            self.x = []
            for i in range(n):
                self.x.append(gan.session.run(x_t))

        return {
                'generator': gan.session.run(self.node, {z_t: z[i], x_t: self.x[i]})
        }


class ProgressiveSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        x_t = gan.inputs.x
        global x
        x = gan.session.run(x_t)

        self.samplers = []
        default = tf.zeros_like(gan.generator.sample)
        def add_samples(layer):
            layer = gan.generator.layer(layer)
            if layer is None:
                layer = default

            self.samplers.append(IdentitySampler(gan, tf.image.resize_images(layer, [256,256], method=1), 1))

        add_samples('g8x8')
        add_samples('g16x16')
        add_samples('g32x32')
        add_samples('g64x64')
        add_samples('g128x128')
        add_samples('g256x256')

    def _sample(self):
        ss = []
        n=4
        for i in range(n):
            samples = [sampler._sample(i,n)['generator'] for sampler in self.samplers]
            sample_stack = np.vstack(samples)
            ss += [sample_stack]
        all_samples = np.concatenate(ss, axis=2)

        return {
            'generator':all_samples
        }

