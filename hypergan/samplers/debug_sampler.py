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

class DebugSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.samplers = [
          StaticBatchSampler(gan, samples_per_row),
          BatchSampler(gan, samples_per_row),
          RandomWalkSampler(gan, samples_per_row)
        ]
        if gan.config.loss['class'] == BoundaryEquilibriumLoss:
          self.samplers += [BeganSampler(gan, samples_per_row)]

        print("GANLOSS", gan.loss.__class__.__name__)

        #if hasattr(self.gan.generator, 'g1x'):
        self.samplers += [SegmentSampler(gan)]


    def _sample(self):
        samples = [sampler._sample()['generator'] for sampler in self.samplers]
        all_samples = np.vstack(samples)

        return {
            'generator':all_samples
        }

