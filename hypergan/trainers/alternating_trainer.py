import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class AlternatingTrainer(BaseTrainer):
    def _create(self):
        gan = self.gan
        config = self.config

        loss = gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        g_optimizer = config.g_optimizer or config.optimizer
        d_optimizer = config.d_optimizer or config.optimizer
        d_optimizer["loss"] = d_loss
        g_optimizer["loss"] = g_loss
        g_optimizer = self.gan.create_optimizer(g_optimizer)
        d_optimizer = self.gan.create_optimizer(d_optimizer)

        d_grads = tf.gradients(d_loss, gan.trainable_d_vars())
        g_grads = tf.gradients(g_loss, gan.trainable_g_vars())
        #d_grads = [gan.distribution_strategy.reduce( tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in d_grads]
        #g_grads = [gan.distribution_strategy.reduce( tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in g_grads]
        apply_vec_g = list(zip((g_grads), (gan.trainable_g_vars()))).copy()
        apply_vec_d = list(zip((d_grads), (gan.trainable_d_vars()))).copy()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self
        g_optimizer_t = g_optimizer.apply_gradients(apply_vec_g)
        d_optimizer_t = d_optimizer.apply_gradients(apply_vec_d)

        self.d_optimizer = d_optimizer
        self.d_optimizer_t = d_optimizer_t
        self.g_optimizer = g_optimizer
        self.g_optimizer_t = g_optimizer_t

        return g_optimizer, d_optimizer

    def variables(self):
        return self.ops.variables() + self.d_optimizer.variables() + self.g_optimizer.variables()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        d_loss, g_loss = loss.sample

        self.before_step(self.current_step, feed_dict)
        for i in range(config.d_update_steps or 1):
            sess.run([self.d_optimizer_t], feed_dict)

        metric_values = sess.run([self.g_optimizer_t] + self.output_variables(metrics), feed_dict)[1:]
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 1 == 0:
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

    def distributed_step(self):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        ops = []
        for i in range(config.d_update_steps or 1):
          ops += [self.d_optimizer_t]
        ops += [self.g_optimizer_t]
        with tf.control_dependencies(ops):
          return tf.identity(1.0)
