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
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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

        d_vars = gan.discriminator_variables()
        g_vars = gan.encoder_variables() + gan.generator_variables()

        d_loss, g_loss = gan.loss_sample_tensor(cache=True)

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        g_optimizer = self.build_optimizer(config, 'g_', config.g_trainer, g_lr, g_vars, g_loss)
        d_optimizer = self.build_optimizer(config, 'd_', config.d_trainer, d_lr, d_vars, d_loss)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        if config.d_clipped_weights is not None:
            gan.graph.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]

        return g_optimizer, d_optimizer

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = gan.config
        d_fake_loss = gan.graph.d_fake_loss
        d_real_loss = gan.graph.d_real_loss
        d_class_loss = gan.graph.d_class_loss

        _, d_cost, d_log = sess.run([self.d_optimizer, self.d_loss, self.d_log], feed_dict)

        # in WGAN paper, values are clipped.  This might not work, and is slow.
        if(config.d_clipped_weights):
            sess.run(gan.graph.clip)

        if(d_class_loss is not None):
            _, g_cost,d_fake,d_real,d_class = sess.run([self.g_optimizer, self.g_loss, d_fake_loss, d_real_loss, d_class_loss], feed_dict)
            if self.current_step % 100 == 0:
                print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_class %.2f d_log %.2f" % (self.current_step, g_cost, d_cost, d_real, d_class, d_log ))
        else:
            _, g_cost,d_fake,d_real = sess.run([self.g_optimizer, self.g_loss, d_fake_loss, d_real_loss], feed_dict)
            if self.current_step % 100 == 0:
                print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_log %.2f" % (self.current_step, g_cost, d_cost, d_real, d_log ))

        return d_cost, g_cost
