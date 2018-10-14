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
        for g, d_v in zip(grads,d_vars):
            if g is None:
                print("!!missing gradient")
                print(d_v)
                return
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads if g is not None
        )
        # Jacobian times gradiant
        if config.update_rule == "ttur" or config.update_rule == 'single-step':
            Jgrads = [0 for i in allvars]
        else:
            Jgrads = tf.gradients(reg, allvars)

        print("JG", Jgrads)

        self.g_gradient = tf.ones([1])
        def amp_for(v):
            if v in g_vars:
                return config.g_w_lambda or 3
            if v in d_vars:
                return config.d_w_lambda or 1

        def applyvec(g, jg, v, decay):
            prev = v
            nextw = v+g + jg * (config.jg_alpha or 0.1)
            if decay is not None:
                return ((decay) * prev + (1.0-decay)*nextw)-v
            else:
                return nextw-v

        def gradient_for(g, jg, v, decay):
            if config.update_rule == "ttur":
                if decay is not None:
                    amp = v+amp_for(v)*g
                    ng = ((decay) * v + (1.0-decay)*amp)-v
                else:
                    ng = amp_for(v)*g
            else:
                if decay is not None:
                    if v in g_vars:
                        ng = applyvec(g, jg, v, decay)
                    else:
                        ng = applyvec(g, jg, v, None)
                else:
                    ng = applyvec(g, jg, v, decay)
            return ng
        decay = config.g_exponential_moving_average_decay
        apply_vec = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(grads, Jgrads, allvars) if Jg is not None ]
        apply_vec_d = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(d_grads, Jgrads[:len(d_vars)], d_vars) if Jg is not None ]
        apply_vec_g = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(g_grads, Jgrads[len(d_vars):], g_vars) if Jg is not None ]

        defn = {k: v for k, v in config.items() if k in inspect.getargspec(config.trainer).args}
        tr = config.trainer(self.lr, **defn)


        optimizer = tr.apply_gradients(apply_vec, global_step=self.global_step)
        d_optimizer = tr.apply_gradients(apply_vec_d, global_step=self.global_step)
        g_optimizer = tr.apply_gradients(apply_vec_g, global_step=self.global_step)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.optimizer = optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        return optimizer, optimizer

    def required(self):
        return "trainer learn_rate".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = gan.metrics()

        metric_values = sess.run([self.optimizer] + self.output_variables(metrics), feed_dict)[1:]

        if self.current_step % 100 == 0:
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values))

