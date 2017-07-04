from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf

class BatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.encoder.z
        inputs_t = gan.inputs.x


        return {
            'generator': gan.session.run(gan.uniform_sample)
        }

