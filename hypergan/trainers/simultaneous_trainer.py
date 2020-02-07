import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = torch.optim.Adam(self.gan.parameters(), lr=1e-3, betas=(0,.999))

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        self.optimizer.zero_grad()

        self.before_step(self.current_step, feed_dict)


        d_loss, g_loss = self.gan.forward_loss()

        for p in self.gan.g_parameters():
            p.requires_grad = True
        for p in self.gan.d_parameters():
            p.requires_grad = False
        g_loss.mean().backward(retain_graph=True)
        for p in self.gan.d_parameters():
            p.requires_grad = True
        for p in self.gan.g_parameters():
            p.requires_grad = False
        d_loss.mean().backward()
        for p in self.gan.g_parameters():
            p.requires_grad = True
        self.optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)


    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

