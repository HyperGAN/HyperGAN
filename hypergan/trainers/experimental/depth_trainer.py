import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class DepthTrainer(BaseTrainer):
    """ Runs an optimizer multiple times and combines the output into a mixture. """
    def _create(self):
        self.hist = [0 for i in range(2)]
        config = self.config
        self.mix_threshold_reached = False
        variables = self.gan.d_vars() + self.gan.g_vars()
        self.ema = [ tf.Variable(_v) for _v in variables ]
        self.store_v = [ _v.assign(_v2) for _v,_v2 in zip(self.ema, variables) ]
        self.combine = [ _v.assign((config.decay or 0.1) *_ema + (1.-(config.decay or 0.1))*_new) for _v, _ema, _new in zip(variables, self.ema, variables)]
        self._delegate = self.gan.create_component(config.trainer, d_vars=self.d_vars, g_vars=self.g_vars)

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config

        self.before_step(self.current_step, feed_dict)
        gan.session.run(self.store_v)
        for i in range(config.depth or 2):
            self._delegate.step(feed_dict)
        gan.session.run(self.combine)
        self.after_step(self.current_step, feed_dict)

