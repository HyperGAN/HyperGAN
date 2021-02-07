import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.gan_component import ValidationException, GANComponent
from hypergan.trainers.base_trainer import BaseTrainer
from hypergan.optimizers.adamirror import Adamirror
from hypergan.optimizers.sam import SAM

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = self.create_optimizer()
        self.ttur = self.config.ttur or 1.0

    def required(self):
        return "optimizer".split()

    def _step(self, feed_dict):
        metrics = self.gan.metrics()

        self.before_step(self.current_step, feed_dict)
        self.gan.next_inputs()
        d_grads, g_grads = self.calculate_gradients()

        for hook in self.train_hooks:
            d_grads, g_grads = hook.gradients(d_grads, g_grads)
        for p, np in zip(self.trainable_gan.d_parameters(), d_grads):
            p.grad = np * self.ttur
        for p, np in zip(self.trainable_gan.g_parameters(), g_grads):
            p.grad = np
        if self.config.gradient_max_norm:
            torch.nn.utils.clip_grad_norm_(self.trainable_gan.parameters(), self.config.gradient_max_norm)

        if(isinstance(self.optimizer, SAM)):
            self.optimizer.first_step(zero_grad=True)
            d_grads, g_grads = self.calculate_gradients()

            for hook in self.train_hooks:
                d_grads, g_grads = hook.gradients(d_grads, g_grads)
            for p, np in zip(self.trainable_gan.d_parameters(), d_grads):
                p.grad = np * self.ttur
            for p, np in zip(self.trainable_gan.g_parameters(), g_grads):
                p.grad = np

            if self.config.gradient_max_norm:
                torch.nn.utils.clip_grad_norm_(self.trainable_gan.parameters(), self.config.gradient_max_norm)

            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)

    def calculate_gradients(self):
        self.optimizer.zero_grad()
        d_loss, g_loss = self.trainable_gan.forward_loss()
        self.gan.add_metric('d_loss', d_loss.mean())
        self.gan.add_metric('g_loss', g_loss.mean())
        for hook in self.train_hooks:
            loss = hook.forward(d_loss, g_loss)
            if loss[0] is not None:
                d_loss += loss[0]
            if loss[1] is not None:
                g_loss += loss[1]

        self.trainable_gan.set_generator_trainable(True)
        self.trainable_gan.set_discriminator_trainable(False)

        g_loss.mean().backward(retain_graph=True)

        self.trainable_gan.set_generator_trainable(False)
        self.trainable_gan.set_discriminator_trainable(True)

        d_loss.mean().backward(retain_graph=True)
        self.trainable_gan.set_generator_trainable(True)

        d_grads = [p.grad * self.ttur for p in self.trainable_gan.d_parameters()]
        g_grads = [p.grad for p in self.trainable_gan.g_parameters()]
        return d_grads, g_grads

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

