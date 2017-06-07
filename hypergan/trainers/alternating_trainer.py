import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class AlternatingTrainer(BaseTrainer):

    def capped_optimizer(optimizer, cap, loss, vars):
        gvs = optimizer.compute_gradients(loss, var_list=vars)
        def create_cap(grad,var):
            if(grad == None) :
                print("Warning: No gradient for variable ",var.name)
                return None
            return (tf.clip_by_value(grad, -cap, cap), var)

        capped_gvs = [create_cap(grad,var) for grad, var in gvs]
        capped_gvs = [x for x in capped_gvs if x != None]
        return optimizer.apply_gradients(capped_gvs)


    def build_optimizer(self, config, prefix, trainer_config, learning_rate, vars, loss):
        with tf.variable_scope(prefix):
            defn = {k[2:]: v for k, v in config.items() if k[2:] in inspect.getargspec(trainer_config).args and k.startswith(prefix)}
            optimizer = trainer_config(learning_rate, **defn)
            if(config.clipped_gradients):
                apply_gradients = self.capped_optimizer(optimizer, config.clipped_gradients, loss, vars)
            else:
                apply_gradients = optimizer.minimize(loss, var_list=vars)

        return apply_gradients

    def _create(self):
        gan = self.gan
        config = self.config
        g_lr = config.g_learn_rate
        d_lr = config.d_learn_rate

        d_vars = gan.discriminator.variables()
        g_vars = gan.encoder.variables() + gan.generator.variables()

        d_loss, g_loss = gan.loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        g_optimizer = self.build_optimizer(config, 'g_', config.g_trainer, g_lr, g_vars, g_loss)
        d_optimizer = self.build_optimizer(config, 'd_', config.d_trainer, d_lr, d_vars, d_loss)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        if config.d_clipped_weights:
            gan.graph.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]

        return g_optimizer, d_optimizer

    def output_string(self, metrics):
        output = "\%2d: " 
        for name in sorted(metrics.keys()):
            output += " " + name
            output += " %.2f"
        return output


    def output_variables(self, metrics):
        gan = self.gan
        sess = gan.session
        return [metrics[k] for k in sorted(metrics.keys())]

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = gan.config
        metrics = gan.loss.metrics

        d_loss, g_loss = self.gan.loss.sample

        _ = sess.run(self.d_optimizer, feed_dict)
        metric_values = sess.run([self.g_optimizer] + self.output_variables(metrics), feed_dict)[1:]

        # TODO in the original WGAN paper, values are clipped.  This is slow.
        if(config.d_clipped_weights):
            sess.run(gan.graph.clip)

        if self.current_step % 100 == 0:
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values))
