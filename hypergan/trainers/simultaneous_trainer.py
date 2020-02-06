import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        pass

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        d_loss, g_loss = loss.sample

        self.before_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

