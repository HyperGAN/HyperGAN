import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    def _create(self):
        gan = self.gan
        config = self.config

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        optimizer = hc.lookup_functions(config.optimizer)
        optimizer['gan']=self.gan
        optimizer['config']=optimizer
        defn = {k: v for k, v in optimizer.items() if k in inspect.getargspec(optimizer['class']).args}
        lr = optimizer.learn_rate or optimizer.learning_rate
        if 'learning_rate' in optimizer:
            del defn['learning_rate']
        tr = optimizer['class'](lr, **defn)
        self.optimizer = tr

        d_grads = tf.gradients(d_loss, gan.d_vars())
        g_grads = tf.gradients(g_loss, gan.g_vars())
        apply_vec = list(zip((d_grads + g_grads), (gan.d_vars() + gan.g_vars()))).copy()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        self.optimize_t = tr.apply_gradients(apply_vec, global_step=self.global_step)

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

