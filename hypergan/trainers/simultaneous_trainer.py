import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = torch.optim.Adam(self.gan.generator.parameters(), lr=1e-3, betas=(.9,.999))
        #remaining
        #* set optimizer from config
        #* use discriminator
        #* use loss
        #* make sure metrics work
        #* generator architecture
        #* gan.parameters() sholud return all component params


    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        self.optimizer.zero_grad()

        g_loss = torch.abs(self.gan.generator(self.gan.latent.sample()))

        #d_loss, g_loss = loss.sample

        (-g_loss.mean()).backward()

        self.before_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)

        self.optimizer.step()

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

