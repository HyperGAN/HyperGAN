import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class AlphaTrainer(BaseTrainer):
    def __init__(self, gan, config, losses=[], var_lists=[]):
        BaseTrainer.__init__(self, gan, config)
        self.losses = losses
        self.var_lists = var_lists

    def _create(self):
        gan = self.gan
        config = self.config
        losses = self.losses
        lr = config.g_learn_rate

        optimizers = []
        self.lr = tf.Variable(lr, dtype=tf.float32)
        for i, _ in enumerate(losses):
            loss = losses[i]
            var_list = self.var_lists[i]

            optimizer = self.build_optimizer(config, 'g_', config.g_trainer, self.lr, var_list, loss)
            optimizers.append(optimizer) #TODO prefx

        self.optimizers = optimizers


        if config.d_clipped_weights:
            self.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]
        else:
            self.clip = []

        return None

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        losses = self.losses

        for i, _ in enumerate(losses):
            loss = losses[i]
            optimizer = self.optimizers[i]
            #metrics = loss.metrics
            _ = sess.run(optimizer)
            #metric_values = sess.run([optimizer] + self.output_variables(metrics), feed_dict)[1:]

            #if self.current_step % 100 == 0:
            #    print("loss " + str(i) + "  "+ self.output_string(metrics) % tuple([self.current_step] + metric_values))
