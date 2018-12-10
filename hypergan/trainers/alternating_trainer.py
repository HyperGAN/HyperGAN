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

        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())

        loss = self.loss or gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))

        g_optimizer = self.build_optimizer(config, 'g_', config.g_trainer, self.g_lr, g_vars, g_loss)
        d_optimizer = self.build_optimizer(config, 'd_', config.d_trainer, self.d_lr, d_vars, d_loss)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        if config.d_clipped_weights:
            self.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]
        else:
            self.clip = []

        return g_optimizer, d_optimizer

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics

        d_loss, g_loss = loss.sample

        for i in range(config.d_update_steps or 1):
            sess.run([self.d_optimizer] + self.clip, feed_dict)

        metric_values = sess.run([self.g_optimizer] + self.output_variables(metrics), feed_dict)[1:]

        if self.current_step % 100 == 0:
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values))

