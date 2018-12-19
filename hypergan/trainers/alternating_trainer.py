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
        def create_optimizer(klass, options):
            options['gan']=self.gan
            options['config']=options
            defn = {k: v for k, v in options.items() if k in inspect.getargspec(klass).args}
            learn_rate = options.learn_rate or options.learning_rate
            if 'learning_rate' in options:
                del defn['learning_rate']
            return klass(learn_rate, **defn)

        g_optimizer = hc.lookup_functions(config.g_optimizer or config.optimizer)
        g_optimizer = create_optimizer(g_optimizer['class'], g_optimizer)
        d_optimizer = hc.lookup_functions(config.d_optimizer or config.optimizer)
        d_optimizer = create_optimizer(d_optimizer['class'], d_optimizer)
        
        d_grads = tf.gradients(d_loss, gan.d_vars())
        g_grads = tf.gradients(g_loss, gan.g_vars())
        apply_vec_g = list(zip((g_grads), (gan.g_vars()))).copy()
        apply_vec_d = list(zip((d_grads), (gan.d_vars()))).copy()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self
        g_optimizer_t = g_optimizer.apply_gradients(apply_vec_g, global_step=self.global_step)
        d_optimizer_t = d_optimizer.apply_gradients(apply_vec_d, global_step=self.global_step)

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_optimizer_t = d_optimizer_t
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

        for i in range(config.d_update_steps or 1):
            sess.run([self.d_optimizer_t], feed_dict)

        self.before_step(self.current_step, feed_dict)
        metric_values = sess.run([self.g_optimizer_t] + self.output_variables(metrics), feed_dict)[1:]
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

