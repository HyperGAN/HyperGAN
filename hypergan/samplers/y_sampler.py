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
        self.styleb_t = gan.styleb.sample
        self.styleb_v = gan.session.run(gan.styleb.sample)
        self.stylea_t = gan.stylea.sample
        self.stylea_v = gan.session.run(gan.stylea.sample)
        self.g=tf.get_default_graph()
        self.frames = gan.session.run(gan.frames)[:-1]
        self.frames_t = gan.frames[:-1]

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            feed_dict = dict(zip(self.frames_t, self.frames))
            feed_dict[self.styleb_t]=self.styleb_v
            feed_dict[self.stylea_t]=self.stylea_v
            sample = gan.session.run(gan.gx.sample, feed_dict=feed_dict)
            self.frames = self.frames[1:] + [sample]
            return {
                'generator': sample
            }

