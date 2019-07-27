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

        if self.z is None:
            self.z = []
            self.x = []
            for i in range(n):
                self.z.append(gan.session.run(z_t))
                self.x.append(gan.session.run(x_t))

        return {
                'generator': gan.session.run(self.node, {z_t: self.z[i], x_t: self.x[i]})
        }


class DebugSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        x_t = gan.inputs.x
        global x
        x = gan.session.run(x_t)
        self.samplers = [
          #IdentitySampler(gan, gan.inputs.x, samples_per_row),
          #IdentitySampler(gan, gan.inputs.xb, samples_per_row),
          #IdentitySampler(gan, gan.autoencoded_x, samples_per_row),
          #StaticBatchSampler(gan, samples_per_row),
          #BatchSampler(gan, samples_per_row),
          #RandomWalkSampler(gan, samples_per_row)
        ]

        #self.samplers += [IdentitySampler(gan, tf.image.resize_images(gan.inputs.x, [128,128], method=1), samples_per_row)]
        #if hasattr(gan.generator, 'pe_layers'):
        #    self.samplers += [IdentitySampler(gan, gx, samples_per_row) for gx in gan.generator.pe_layers]
        #    pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
        if hasattr(gan, 'noise_generator'):
            self.samplers += [IdentitySampler(gan, tf.concat([gan.noise_generator.sample, gan.generator.sample, gan.noise_generator.sample + gan.generator.sample], axis=0), samples_per_row)]
        #self.samplers += 
        if hasattr(gan, 'autoencoded_x'):
          self.samplers += [IdentitySampler(gan, tf.concat([gan.inputs.x,gan.autoencoded_x], axis=0), samples_per_row)]
        if gan.config.loss['class'] == BoundaryEquilibriumLoss:
          self.samplers += [BeganSampler(gan, samples_per_row)]


        if isinstance(gan.generator, SegmentGenerator):
            self.samplers += [SegmentSampler(gan)]

        if hasattr(gan, 'seq'):
            self.samplers += [IdentitySampler(gan, tf.image.resize_images(gx, [128,128], method=1), samples_per_row) for gx in gan.seq]

        default = gan.generator.sample#tf.zeros_like(gan.generator.layer('gend8x8'))
        def add_samples(layer):
            layer = gan.generator.layer(layer)
            if layer is None:
                layer = default

            self.samplers.append(IdentitySampler(gan, tf.image.resize_images(layer, [128,128], method=1), 1))

        add_samples('gend8x8')
        add_samples('gend16x16')
        #add_samples('gend32x32')
        #add_samples('gend64x64')
        #add_samples('gend128x128')
        if "match_support_mx" in gan.discriminator.named_layers:
            self.samplers.append(IdentitySampler(gan, tf.concat([gan.inputs.x,tf.image.resize_images(gan.discriminator.named_layers['match_support_mx'], [128,128], method=1), tf.image.resize_images(gan.discriminator.named_layers['match_support_m+x'], [128,128], method=1)],axis=0),  1) )
            self.samplers.append(IdentitySampler(gan, tf.concat([gan.generator.sample, tf.image.resize_images(gan.discriminator.named_layers['match_support_mg'], [128,128], method=1), tf.image.resize_images(gan.discriminator.named_layers['match_support_m+g'], [128,128], method=1)],axis=0),  1) ) 




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

