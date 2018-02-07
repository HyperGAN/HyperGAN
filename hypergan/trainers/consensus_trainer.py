import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class ConsensusTrainer(BaseTrainer):
    def create(self):
        config = self.config
        lr = config.learn_rate
        self.global_step = tf.train.get_global_step()
        decay_function = config.decay_function
        if decay_function:
            print("!!using decay function", decay_function)
            decay_steps = config.decay_steps or 50000
            decay_rate = config.decay_rate or 0.9
            decay_staircase = config.decay_staircase or False
            self.lr = decay_function(lr, self.global_step, decay_steps, decay_rate, decay_staircase)
        else:
            self.lr = lr

        return self._create()


    def _create(self):
        gan = self.gan
        config = self.config

        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())

        loss = self.loss or gan.loss
        d_loss, g_loss = loss.sample
        allloss = d_loss + g_loss
        allvars = d_vars + g_vars

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        grads = d_grads + g_grads

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # Jacobian times gradiant
        Jgrads = tf.gradients(reg, allvars)


        def applyvec(g, alpha, jg):
            print("G", g, "jg", jg)
            return g + alpha * jg
        # Gradient updates
        jg_alpha = config.jg_alpha or 0.1
        apply_vec = [
             (applyvec(g, jg_alpha, Jg), v)
             for (g, Jg, v) in zip(grads, Jgrads, allvars) if Jg is not None
        ]


        #optimizer = tf.train.RMSPropOptimizer(self.d_lr)
        defn = {k: v for k, v in config.items() if k in inspect.getargspec(config.trainer).args}
        optimizer = config.trainer(self.lr, **defn)
        optimizer = optimizer.apply_gradients(apply_vec)
        #optimizer = self.build_optimizer(config, '', config.d_trainer, self.d_lr, allvars, allloss)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.optimizer = optimizer

        return optimizer, optimizer

    def required(self):
        return "trainer learn_rate".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics

        metric_values = sess.run([self.optimizer] + self.output_variables(metrics), feed_dict)[1:]

        if self.current_step % 100 == 0:
            print("____", self.output_string(metrics), metric_values)
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values))

