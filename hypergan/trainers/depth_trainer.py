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
        variables = self.gan.d_vars() + self.gan.g_vars()
        self.ema = [ tf.Variable(_v) for _v in variables ]
        self.store_v = [ _v.assign(_v2) for _v,_v2 in zip(self.ema, variables) ]
        self.combine = [ _v.assign(config.decay *_ema + (1.-config.decay)*_new) for _v, _ema, _new in zip(variables, self.ema, variables)]
        self._delegate = self.gan.create_component(config.trainer, d_vars=self.d_vars, g_vars=self.g_vars, loss=self.loss)
        self._delegate.create()
        self.slot_vars_g = self._delegate.slot_vars_g
        self.slot_vars_d = self._delegate.slot_vars_g

        if self.config.candidate:
            self.mixg = tf.Variable(1, dtype=tf.float32)
            self.mixd = tf.Variable(1, dtype=tf.float32)
            self.gan.add_metric('mixg', self.mixg)
            self.gan.add_metric('mixd', self.mixd)
            self.combine_d = [ _v.assign(self.mixd *_ema + (1.-self.mixd)*_new) for _v, _ema, _new in zip(self.gan.d_vars(), self.ema, self.gan.d_vars())]
            self.combine_g = [ _v.assign(self.mixg *_ema + (1.-self.mixg)*_new) for _v, _ema, _new in zip(self.gan.g_vars(), self.ema[len(self.gan.d_vars()):], self.gan.g_vars())]

            self.candidate = [ tf.Variable(_v) for _v in variables ]
            self.store_candidate = [ _v.assign(_v2) for _v,_v2 in zip(self.candidate, variables) ]
            self.reset_discriminator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.d_vars(), self.ema) ]
            self.reset_generator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.g_vars(), self.ema[len(self.gan.d_vars()):]) ]
            self.reset_candidate_discriminator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.d_vars(), self.candidate) ]
            self.reset_candidate_generator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.g_vars(), self.candidate[len(self.gan.d_vars()):]) ]
            self.candidate_loss = self.gan.loss.sample[0] - self.gan.loss.sample[1]
    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss 

        gan.session.run(self.store_v)
        for i in range(config.depth or 2):
            self._delegate.step(feed_dict)
        if self.config.candidate:
            d_fake2b = np.sum([gan.session.run(self.candidate_loss) for i in range(self.config.candidate_tests)])
            gan.session.run(self.store_candidate)
            gan.session.run(self.reset_discriminator)
            d_fake2a = np.sum([gan.session.run(self.candidate_loss) for i in range(self.config.candidate_tests)])
            gan.session.run(self.reset_generator)
            d_fake1a = np.sum([gan.session.run(self.candidate_loss) for i in range(self.config.candidate_tests)])
            gan.session.run(self.reset_candidate_discriminator)
            d_fake1b = np.sum([gan.session.run(self.candidate_loss) for i in range(self.config.candidate_tests)])
            gan.session.run(self.reset_candidate_generator)

            payoff = [[d_fake1a, d_fake1b],[d_fake2a, d_fake2b]]

            d1 = d_fake1a + d_fake1b
            d2 = d_fake2a + d_fake2b
            g1 = d_fake1a + d_fake2a
            g2 = d_fake1b + d_fake2b
            mixd = d2/(d1 + d2)
            mixg = g1/(g1 + g2)
            if self.config.reverse:
                mixd = d1/(d1 + d2)
                mixg = g2/(g1 + g2)
            mixd = np.minimum(1.0, mixd)
            mixd = np.maximum(0.0, mixd)
            mixg = np.minimum(1.0, mixg)
            mixg = np.maximum(0.0, mixg)
            gan.session.run([self.combine_d, self.combine_g], {self.mixd: mixd, self.mixg: mixg})
        else:
            gan.session.run(self.combine)

