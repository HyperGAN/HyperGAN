import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class AlternatingTrainer(BaseTrainer):
    """ Steps G and D alternating """
    def _create(self):
        self.d_optimizer = torch.optim.Adam(self.gan.d_parameters(), lr=1e-3, betas=(0,.999))
        self.g_optimizer = torch.optim.Adam(self.gan.g_parameters(), lr=1e-3, betas=(0,.999))

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        self.d_optimizer.zero_grad()

        self.before_step(self.current_step, feed_dict)

        d_loss, g_loss = self.gan.forward_loss()
        d_loss.mean().backward()
        self.d_optimizer.step()

        self.g_optimizer.zero_grad()
        d_loss, g_loss = self.gan.forward_loss()
        g_loss.mean().backward()
        self.g_optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)


    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

