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
        self.reset_optimizer_t = tf.variables_initializer(self._delegate.variables())
        self.depth_step = 0
        self.fitness = -self.gan.loss.d_fake
        self.latent = None

    def required(self):
        return "".split()

    def _best_latent(self):
        if self.latent is None:
            self.latent = self.gan.session.run(self.gan.latent.sample)
        fitness = self.gan.session.run(self.fitness, {self.gan.latent.sample:self.latent})
        zs = self.latent
        sort_zs = None
        last_fitness = 10000
        count = 0
        while True:
            d = self.gan.session.run([self.fitness,self.gan.latent.sample])
            _f = d[0]
            _z = d[1]
            fitness = np.reshape(fitness, np.shape(_f))
            fitness = np.concatenate([fitness,_f], axis=0)
            zs = np.reshape(zs, np.shape(_z))
            zs = np.concatenate([zs,_z], axis=0)
            sort = np.argsort(fitness.flatten())[:self.gan.batch_size()]
            zs = zs[sort]
            fitness = fitness.flatten()[sort]
            if fitness.flatten()[-1] < last_fitness:
                last_fitness = fitness[-1]
                count = 0
            else:
                count += 1
                if count > self.config.heuristic:
                    #print("z fit ", i)
                    sort_zs = np.reshape(zs, np.shape(_z))
                    break
        return sort_zs

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        depth = self.config.depth
        if depth:
            if self.current_step % depth == 0:
                if self.config.freeze_latent:
                    if self.config.freeze_latent == "best":
                        self.latent = self._best_latent()
                    else:
                        self.latent = self.gan.session.run(self.gan.latent.sample)
                    feed_dict[gan.latent.sample] = self.latent
                self.before_step(self.current_step, feed_dict)
                gan.session.run(self.store_v)
                if self.config.reset_optimizer:
                    self.gan.session.run([self.reset_optimizer_t])

            if self.config.freeze_latent:
                feed_dict[gan.latent.sample] = self.latent
            self._delegate.step(feed_dict)
            if self.current_step % depth == depth - 1:
                gan.session.run(self.combine)
                self.after_step(self.current_step, feed_dict)
        else:
            if self.depth_step == 0:
                if self.config.freeze_latent:
                    if self.config.freeze_latent == "best":
                        self.latent = self._best_latent()
                    else:
                        self.latent = self.gan.session.run(self.gan.latent.sample)
                    feed_dict[gan.latent.sample] = self.latent
                self.before_step(self.current_step, feed_dict)
                gan.session.run(self.store_v)
                self.max_gradient_mean = 0.0
            if self.config.freeze_latent:
                feed_dict[gan.latent.sample] = self.latent
            self._delegate.step(feed_dict)
            gradient_mean = gan.session.run(gan.gradient_mean, feed_dict)
            self.depth_step += 1
            if gradient_mean > self.max_gradient_mean:
                self.max_gradient_mean = gradient_mean
            if gradient_mean/self.max_gradient_mean < (0.2 or self.config.gradient_threshold):
                gan.session.run(self.combine)
                self.after_step(self.current_step, feed_dict)
                self.depth_step = 0

    def variables(self):
        return self._delegate.variables()
