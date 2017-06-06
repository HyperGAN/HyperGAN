from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf

class BatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.encoder.z #TODO
        inputs_t = gan.inputs[0]


        return {
            'generator': gan.session.run(gan.generator.sample)
        }

