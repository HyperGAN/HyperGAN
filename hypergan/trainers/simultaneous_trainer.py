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
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.step_ops = None
        config.optimizer["loss"] = loss.sample

        self.optimizer = self.gan.create_optimizer(config.optimizer)
        d_vars = self.d_vars or self.gan.d_vars()
        g_vars = self.g_vars or self.gan.g_vars()

        if self.gan.distribution_strategy is not None:
            return

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)
        apply_vec = list(zip((d_grads + g_grads), (d_vars + g_vars))).copy()
        for grad, v in apply_vec:
            if grad is None:
                print("Gradient is None:", v)
        for t in self.train_hooks:
            d_grads, g_grads = t.gradients(d_grads, g_grads)
        apply_vec = list(zip((d_grads + g_grads), (d_vars + g_vars))).copy()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        self.optimize_t = self.optimizer.apply_gradients(apply_vec)

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
        if self.step_ops is None:
            ops = [self.optimize_t]
            update_train_hooks = [t.update_op() for t in self.train_hooks]
            update_train_hooks = [op for op in update_train_hooks if op is not None]
            self.step_ops = ops + update_train_hooks
        sess.run(self.step_ops, feed_dict)
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            metric_values = self.gan.session.run(self.output_variables(metrics))
            self.print_metrics(self.current_step)

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.gan.session.run(self.output_variables(metrics))
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

