import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer
from hypergan.optimizers.adamirror import Adamirror

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = torch.optim.Adam(self.gan.parameters(), lr=self.config.optimizer["learn_rate"], betas=(.9,.999))
        #self.optimizer = Adamirror(self.gan.parameters(), lr=self.config.optimizer["learn_rate"], betas=(0.0,.999))
        #self.adamirror = Adamirror(self.gan.parameters(), lr=self.config.optimizer["learn_rate"], betas=(0.9074537537537538,.997))
        #self.adamirror2 = Adamirror(self.gan.parameters(), lr=self.config.optimizer["learn_rate"]*3, betas=(0.9074537537537538,.997))
        #self.optimizer = self.adamirror
        self.gan.add_component("optimizer", self.optimizer)
        #self.gan.add_component("optimizer", self.adamirror2)
        #self.gan.add_component("optimizer", self.adamirror)

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
        self.d_loss = d_loss
        self.g_loss = g_loss
        for hook in self.train_hooks:
            loss = hook.forward()
            if loss[0] is not None:
                d_loss += loss[0]
            if loss[1] is not None:
                g_loss += loss[1]

        for p in self.gan.g_parameters():
            p.requires_grad = True
        for p in self.gan.d_parameters():
            p.requires_grad = False
        g_loss = g_loss.mean()
        g_loss.backward(retain_graph=True)
        for p in self.gan.d_parameters():
            p.requires_grad = True
        for p in self.gan.g_parameters():
            p.requires_grad = False
        d_loss = d_loss.mean()
        d_loss.backward(retain_graph=True)
        for p in self.gan.g_parameters():
            p.requires_grad = True

        d_grads = [p.grad for p in self.gan.d_parameters()]
        g_grads = [p.grad for p in self.gan.g_parameters()]

        for hook in self.train_hooks:
            d_grads, g_grads = hook.gradients(d_grads, g_grads)
        for p, np in zip(self.gan.d_parameters(), d_grads):
            p.grad = np
        for p, np in zip(self.gan.g_parameters(), g_grads):
            p.grad = np

        self.optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)


    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

