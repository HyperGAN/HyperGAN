from hypergan.samplers.common import *
from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None

    def _sample(self):
        gan = self.gan
        z_t = gan.encoders[0].z #TODO
        inputs_t = gan.inputs[0]

        if self.z is None:
            self.z = gan.encoders[0].sample() #TODO
            self.input = gan.sample_input()

        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            {
                'generator': gan.generator.sample(feed_dict={z_t: self.z, inputs_t: self.input})
            }

