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
        self.last_frame_1, self.last_frame_2, self.last_frame_3, self.y = gan.session.run([gan.last_frame_1, gan.last_frame_2, gan.last_frame_3, gan.y.sample])
        self.g=tf.get_default_graph()

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        lf1_t = gan.last_frame_1
        lf2_t = gan.last_frame_2
        lf3_t = gan.last_frame_3
        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            sample = gan.session.run(gan.gx.sample, feed_dict={lf1_t: self.last_frame_1, lf2_t: self.last_frame_2, self.styleb_t: self.styleb_v, self.stylea_t: self.stylea_v})
            self.last_frame_1 =self.last_frame_2
            self.last_frame_2 = sample
            #self.last_frame_3 = self.y
            return {
                'generator': sample
            }

