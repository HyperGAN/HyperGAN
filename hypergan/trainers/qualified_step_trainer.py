import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class QualifiedStepTrainer(BaseTrainer):
    def create(self):
        self.hist = [0 for i in range(2)]
        config = self.config
        self.global_step = tf.train.get_global_step()
        self.mix_threshold_reached = False
        decay_function = config.decay_function
        variables = self.gan.d_vars() + self.gan.g_vars()
        self.ema = [ tf.Variable(_v) for _v in variables ]
        self.store_v = [ _v.assign(_v2) for _v,_v2 in zip(self.ema, variables) ]
        self._delegate = self.gan.create_component(config.trainer, d_vars=self.d_vars, g_vars=self.g_vars, loss=self.loss)
        self._delegate.create()
        self.slot_vars_g = self._delegate.slot_vars_g
        self.slot_vars_d = self._delegate.slot_vars_g

        self.mixg = tf.Variable(1, dtype=tf.float32)
        self.mixd = tf.Variable(1, dtype=tf.float32)
        self.combine_d = [ _v.assign((1.-self.mixd) *_ema + self.mixd*_new) for _v, _ema, _new in zip(self.gan.d_vars(), self.ema, self.gan.d_vars())]
        self.combine_g = [ _v.assign((1.-self.mixg) *_ema + self.mixg*_new) for _v, _ema, _new in zip(self.gan.g_vars(), self.ema[len(self.gan.d_vars()):], self.gan.g_vars())]

        self.candidate = [ tf.Variable(_v) for _v in variables ]
        self.store_candidate = [ _v.assign(_v2) for _v,_v2 in zip(self.candidate, variables) ]
        self.reset_discriminator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.d_vars(), self.ema) ]
        self.reset_generator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.g_vars(), self.ema[len(self.gan.d_vars()):]) ]
        self.reset_candidate_discriminator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.d_vars(), self.candidate) ]
        self.reset_candidate_generator = [ _v.assign(_v2) for _v,_v2 in zip(self.gan.g_vars(), self.candidate[len(self.gan.d_vars()):]) ]
        self.candidate_loss = self.gan.loss.sample[0] - self.gan.loss.sample[1]
        if self.config.fitness == "d_fake":
            self.candidate_loss = self.gan.loss.sample[0]
        self.g_rate = 0.0
        self.d_rate = 0.0
        self.zs = [self.gan.session.run(self.gan.latent.sample) for i in range(self.config.candidate_tests)]

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss 

        gan.session.run(self.store_v)
        self._delegate.step(feed_dict)
        if(self.config.turn_off_step):
            if(self.config.turn_off_step < self.current_step):
                return
            else:
                print(str(-self.current_step+self.config.turn_off_step) + " qualified steps remain")

        # b = generator at new step
        # a = generator at past step
        # 1 = discriminator at past step
        # 2 = discriminator at new step
        b2 = np.mean([gan.session.run(self.candidate_loss, {gan.latent.sample: self.zs[i]}) for i in range(self.config.candidate_tests)])
        gan.session.run(self.store_candidate)
        gan.session.run(self.reset_discriminator)
        b1 = np.mean([gan.session.run(self.candidate_loss, {gan.latent.sample: self.zs[i]}) for i in range(self.config.candidate_tests)])
        gan.session.run(self.reset_generator)
        a1 = np.mean([gan.session.run(self.candidate_loss, {gan.latent.sample: self.zs[i]}) for i in range(self.config.candidate_tests)])
        gan.session.run(self.reset_candidate_discriminator)
        a2 = np.mean([gan.session.run(self.candidate_loss, {gan.latent.sample: self.zs[i]}) for i in range(self.config.candidate_tests)])
        gan.session.run(self.reset_candidate_generator)

        #    D1 D2
        # G1 a1 b1
        # G2 a2 b2
        payoff = [[a1, b1],[a2, b2]]

        d1 = a1 + a2
        d2 = b1 + b2
        g1 = a1 + b1
        g2 = a2 + b2
        if self.config.negate:
            g1 = -g1
            g2 = -g2
        mixd = d2/(d1 + d2)
        mixg = g1/(g1 + g2)
        if self.config.find_zero:
            slope = d2 - d1
            t = -d1/slope
            print("d", t)
            mixd = t
            slope = g2 - g1
            t = -g1/slope
            print("g", t)
            mixg = t
            print(t * slope + g1)
        if self.config.reverse:
            mixd = d1/(d1 + d2)
            mixg = g2/(g1 + g2)
        if self.config.loud:
            mixd = d1/(d1 + d2)
            mixg = g1/(g1 + g2)
        if self.config.zero:
            if np.abs(d2) <= np.abs(d1):
                mixd = 1.0
                self.d_rate += 1
            else:
                mixd = 0.0
            if np.abs(g2) >= np.abs(g1):
                self.g_rate += 1
                mixg = 1.0
            else:
                mixg = 0.0
        if self.config.unbounded:
            pass
        else:
            bounds = self.config.bounds or [0.0, 1.0]
            mixd = np.minimum(bounds[1], mixd)
            mixd = np.maximum(bounds[0], mixd)
            mixg = np.minimum(bounds[1], mixg)
            mixg = np.maximum(bounds[0], mixg)
        gan.session.run([self.combine_d, self.combine_g], {self.mixd: mixd, self.mixg: mixg})

        if self.current_step % 100 == 0:
            print("rates %d/100 g %d/100 d" % (self.g_rate, self.d_rate))
            self.g_rate = 0.0
            self.d_rate = 0.0
