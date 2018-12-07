import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class DepthTrainer(BaseTrainer):
    def create(self):
        self.hist = [0 for i in range(2)]
        config = self.config
        self.global_step = tf.train.get_global_step()
        self.mix_threshold_reached = False
        decay_function = config.decay_function
        self.ema = [ tf.Variable(_v) for _v in self.gan.variables() ]
        self.store_v = [ _v.assign(_v2) for _v,_v2 in zip(self.ema, self.gan.variables()) ]
        self.combine = [ _v.assign(config.decay *_ema + (1.-config.decay)*_new) for _v, _ema, _new in zip(self.gan.variables(), self.ema, self.gan.variables())]

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss 

        self._delegate = self.gan.create_component(config.trainer, d_vars=d_vars, g_vars=g_vars, loss=loss)
        gan.session.run(self.store_v)
        for i in enumerate(config.depth or 2):
            self._delegate._step(feed_dict)
        gan.session.run(self.combine)
