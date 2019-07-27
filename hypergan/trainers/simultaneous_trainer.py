import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        gan = self.gan
        config = self.config

        if hasattr(self, 'loss'):
            loss = self.loss 
        else:
            loss = self.gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))
        config.optimizer["loss"] = loss.sample

        self.optimizer = self.gan.create_optimizer(config.optimizer)
        d_vars = self.d_vars or self.gan.d_vars()
        g_vars = self.g_vars or self.gan.g_vars()

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)
        apply_vec = list(zip((d_grads + g_grads), (d_vars + g_vars))).copy()
        self.gan.gradient_mean = sum([tf.reduce_mean(tf.abs(grad)) for grad in d_grads+g_grads])/len(d_grads+g_grads)
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        self.optimize_t = self.optimizer.apply_gradients(apply_vec, global_step=self.global_step)

        return self.optimize_t, self.optimize_t

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        d_loss, g_loss = loss.sample

        self.before_step(self.current_step, feed_dict)
        metric_values = sess.run([self.optimize_t] + self.output_variables(metrics), feed_dict)[1:]
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

