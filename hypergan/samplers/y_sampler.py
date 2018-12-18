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
        self.g=tf.get_default_graph()
        self.frames = gan.session.run(gan.frames)
        self.frames_t = gan.frames
        self.zs2, self.cs2 = gan.session.run([gan.zs[-1], gan.cs[-1]])
        self.zs2 = [self.zs2]
        self.cs2 = [self.cs2]
        self.zs_t = [gan.video_generator_last_z]
        self.cs_t = [gan.video_generator_last_c]
        self.zs = gan.session.run([gan.video_generator_last_z])
        self.cs = gan.session.run([gan.video_generator_last_c])
        self.i=0

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_distribution.sample
        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            feed_dict = dict(zip(self.frames_t, self.frames))
            sample, *gstack = gan.session.run(gan.gs_next, feed_dict=feed_dict)
            self.frames = self.frames[1:] + [sample]

            feed_dict = dict(zip(self.frames_t, self.frames))
            if self.i % 100 == 0:
                if self.i == 0:
                    self.frames3 = gan.session.run(gan.frames)
            feed_dict = dict(zip(self.frames_t, self.frames3))
            print('c_drift', gan.session.run(gan.c_drift, feed_dict=feed_dict))

            if self.i % 100 == 0:
                self.frames2 = gan.session.run(gan.frames)
            feed_dict = dict(zip(self.frames_t, self.frames2))
            sample_reset = gan.session.run(gan.gs_next[0], feed_dict=feed_dict)
            self.frames2 = self.frames2[1:] + [sample_reset]

            if self.i % 100 == 0:
                self.prev_frames = gan.session.run(gan.frames)
            feed_dict = dict(zip(gan.frames[1:], [self.prev_frames[0]]+self.prev_frames[:-1]))
            prev_sample = gan.session.run(gan.gy.sample, feed_dict=feed_dict)
            self.prev_frames = [prev_sample] + self.prev_frames[:-1]

            feed_dict = dict(zip(self.zs_t, self.zs))
            feed_dict.update(dict(zip(self.cs_t,self.cs)))
            gx, z_sample, c_sample = gan.session.run([gan.generator.sample, gan.video_generator_last_zn, gan.video_generator_last_cn], feed_dict=feed_dict)
            self.zs = self.zs[1:] + [z_sample]
            self.cs = self.cs[1:] + [c_sample]

            if self.i % 500 == 0:
                self.zs2, self.cs2 = gan.session.run([gan.zs[-1], gan.cs[-1]])
                self.zs2 = [self.zs2]
                self.cs2 = [self.cs2]
            feed_dict = dict(zip(self.zs_t, self.zs2))
            feed_dict.update(dict(zip(self.cs_t,self.cs2)))
            gc, z_sample, c_sample = gan.session.run([gan.generator.sample, gan.video_generator_last_zn, gan.video_generator_last_cn], feed_dict=feed_dict)
            self.zs2 = self.zs2[1:] + [z_sample]
            self.cs2 = self.cs2[1:] + [c_sample]

            feed_dict = dict(zip(self.zs_t, self.zs))
            x = gan.session.run(gan.generator.sample)
            self.i+=1
            return {
                'generator': np.hstack([sample,  sample_reset, gx,gc])
            }

