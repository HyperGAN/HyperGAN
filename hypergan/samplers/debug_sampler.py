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
    def __init__(self, gan, node, samples_per_row=8):
        self.node = node
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        x_t = gan.inputs.x
        global z,x

        if z is None:
            z = gan.session.run(z_t)
            x = gan.session.run(x_t)

        return {
                'generator': gan.session.run(self.node, {z_t: z, x_t: x})
        }


class DebugSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.samplers = [
          IdentitySampler(gan, gan.inputs.x, samples_per_row),
#          IdentitySampler(gan, gan.autoencoded_x, samples_per_row),
          StaticBatchSampler(gan, samples_per_row),
          BatchSampler(gan, samples_per_row),
          RandomWalkSampler(gan, samples_per_row)
        ]

        self.samplers += [IdentitySampler(gan, gan.inputs.x, samples_per_row)]
        if hasattr(gan.generator, 'pe_layers'):
            self.samplers += [IdentitySampler(gan, gx, samples_per_row) for gx in gan.generator.pe_layers]
            pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
            print("SAMPLERS", pe_layers)
        self.samplers += [IdentitySampler(gan, gan.inputs.x, samples_per_row)]
#          IdentitySampler(gan, gan.autoencoded_x, samples_per_row),
        if gan.config.loss['class'] == BoundaryEquilibriumLoss:
          self.samplers += [BeganSampler(gan, samples_per_row)]


        if isinstance(gan.generator, SegmentGenerator):
            self.samplers += [SegmentSampler(gan)]

        if hasattr(gan, 'seq'):
            self.samplers += [IdentitySampler(gan, gx, samples_per_row) for gx in gan.seq]


    def _sample(self):
        samples = [sampler._sample()['generator'] for sampler in self.samplers]
        all_samples = np.vstack(samples)

        return {
            'generator':all_samples
        }

