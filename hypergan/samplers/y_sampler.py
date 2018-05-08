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
        self.styleb_v = np.zeros_like(self.styleb_v)
        self.stylea_t = gan.stylea.sample
        self.stylea_v = gan.session.run(gan.stylea.sample)
        self.stylea_v = np.zeros_like(self.stylea_v)
        self.g=tf.get_default_graph()
        self.frames = gan.session.run(gan.frames)[:-1]
        self.frames_t = gan.frames[:-1]
        self.i=0

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

            feed_dict = dict(zip(self.frames_t, self.frames))
            if self.i % 100 == 0:
                if self.i == 0:
                    self.frames3 = gan.session.run(gan.frames)[:-1]
            feed_dict = dict(zip(self.frames_t, self.frames3))
            style_reset = gan.session.run(gan.gx.sample, feed_dict=feed_dict)
            self.frames3 = self.frames3[1:] + [style_reset]

            if self.i % 100 == 0:
                self.frames2 = gan.session.run(gan.frames)[:-1]
            feed_dict = dict(zip(self.frames_t, self.frames2))
            feed_dict[self.styleb_t]=self.styleb_v
            feed_dict[self.stylea_t]=self.stylea_v
            sample_reset = gan.session.run(gan.gx.sample, feed_dict=feed_dict)
            self.frames2 = self.frames2[1:] + [sample_reset]

            if self.i % 100 == 0:
                self.prev_frames = gan.session.run(gan.frames)[1:]
            feed_dict = dict(zip(self.frames_t, [self.prev_frames[0]]+self.prev_frames[:-1]))
            feed_dict[self.styleb_t]=self.styleb_v
            feed_dict[self.stylea_t]=self.stylea_v
            prev_sample = gan.session.run(gan.gy.sample, feed_dict=feed_dict)
            self.prev_frames = [prev_sample] + self.prev_frames[:-1]


            self.i+=1
            return {
                'generator': np.hstack([sample, style_reset, sample_reset, prev_sample])
            }

