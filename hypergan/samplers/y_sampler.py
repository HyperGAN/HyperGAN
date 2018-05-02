import numpy as np
import tensorflow as tf
from PIL import Image
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
import time

class YSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.mask = None
        self.step = 0
        self.steps = 8
        self.target = None
        self.y_t = gan.y.sample
        self.y = gan.session.run(self.y_t)
        self.input = gan.session.run(gan.inputs.x)
        self.styleb_t = gan.styleb.sample
        self.styleb_v = gan.session.run(gan.styleb.sample)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        inputs_t = gan.inputs.x
        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            sample = gan.session.run(gan.gx.sample, feed_dict={inputs_t: self.input, self.styleb_t: self.styleb_v, self.y_t: self.y})
            self.y = sample
            return {
                'generator': sample
            }

