import numpy as np
import torch
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class AlternatingTrainer(BaseTrainer):
    """ Steps G and D alternating """
    def _create(self):
        self.d_optimizer = self.create_optimizer("d_optimizer")
        self.g_optimizer = self.create_optimizer("g_optimizer")

    def required(self):
        return "".split()

    def calculate_gradients(self, targets=['d','g']):
        if 'g' in targets and \
            ( self.config.train_g_every == None or
              self.config.train_g_every and (self.gan.steps % self.config.train_g_every == 0) ):

            self.g_optimizer.zero_grad()
            for p in self.gan.g_parameters():
                p.requires_grad = True
            for p in self.gan.d_parameters():
                p.requires_grad = False
            _, g_loss = self.gan.forward_loss()#TODO targets=['d']
            self.gan.add_metric('g_loss', g_loss.mean())
            for hook in self.train_hooks:
                loss = hook.forward()#TODO targets=['d']
                if loss[1] is not None:
                    g_loss += loss[1]

            g_loss = g_loss.mean()
            g_loss.backward(retain_graph=True)
            g_grads = [p.grad for p in self.gan.g_parameters()]
        else:
            g_grads = []

        if 'd' in targets \
            and ( self.config.train_d_every == None or
              (self.gan.steps % self.config.train_d_every == 0)) \
            and ( self.config.freeze_d_every == None or 
              (self.gan.steps % self.config.freeze_d_every) < \
                (self.config.freeze_d_every - self.config.freeze_d_duration)):
            self.d_optimizer.zero_grad()
            for p in self.gan.d_parameters():
                p.requires_grad = True
            for p in self.gan.g_parameters():
                p.requires_grad = False

            d_loss, _ = self.gan.forward_loss()#TODO targets=['d']
            self.gan.add_metric('d_loss', d_loss.mean())
            for hook in self.train_hooks:
                loss = hook.forward()#TODO targets=['d']
                if loss[0] is not None:
                    d_loss += loss[0]

            d_loss = d_loss.mean()
            d_loss.backward(retain_graph=True)
            d_grads = [p.grad for p in self.gan.d_parameters()]
        else:
            d_grads = []

        return d_grads, g_grads


    def _step(self, feed_dict):
        gan = self.gan
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        self.before_step(self.current_step, feed_dict)

        d_grads, _ = self.calculate_gradients(['d'])

        for hook in self.train_hooks:
            d_grads, _ = hook.gradients(d_grads, _)
        for p, np in zip(self.gan.d_parameters(), d_grads):
            p.grad = np

        if(len(d_grads) > 0):
                self.d_optimizer.step()

        _, g_grads = self.calculate_gradients(['g'])

        for hook in self.train_hooks:
            _, g_grads = hook.gradients(_, g_grads)
        for p, np in zip(self.gan.g_parameters(), g_grads):
            p.grad = np

        if(len(g_grads) > 0):
            self.g_optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)


    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

