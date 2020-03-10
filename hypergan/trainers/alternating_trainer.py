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
        return "d_optimizer g_optimizer".split()

    def g_grads(self, d_real=None, d_fake=None):
        self.g_optimizer.zero_grad()
        self.setup_gradient_flow(self.gan.g_parameters(), self.gan.d_parameters())

        if d_fake is None:
            _, g_loss = self.gan.forward_loss()#TODO targets=['d']
        else:
            _, g_loss = self.gan.loss.forward(d_real, d_fake)#TODO targets=['d']
        _, g_loss = self.gan.forward_loss()#TODO targets=['d']

        self.gan.add_metric('g_loss', g_loss.mean())
        g_loss += sum([l[0] for l in self.train_hook_losses() if l[0] is not None])
        g_loss = g_loss.mean()

        return self.grads_for(g_loss, self.gan.g_parameters())

    def d_grads(self, d_real=None, d_fake=None):
        self.d_optimizer.zero_grad()
        self.setup_gradient_flow(self.gan.d_parameters(), self.gan.g_parameters())

        if d_fake is None:
            d_loss, _ = self.gan.forward_loss()#TODO targets=['d']
        else:
            d_loss, _ = self.gan.loss.forward(d_real, d_fake)#TODO targets=['d']
        self.gan.add_metric('d_loss', d_loss.mean())
        d_loss += sum([l[0] for l in self.train_hook_losses() if l[0] is not None])
        d_loss = d_loss.mean()

        return self.grads_for(d_loss, self.gan.d_parameters())

    def setup_gradient_flow(self, train_params, ignore_params):
        for p in train_params:
            p.requires_grad = True
        for p in ignore_params:
            p.requires_grad = False

    def grads_for(self, loss, train_params):
        if loss == 0:
            return []
        loss.backward(retain_graph=True)
        #return torch_grad(outputs=loss, inputs=train_params, retain_graph=True)
        return [p.grad for p in train_params]

    def train_hook_losses(self):
        losses = []
        for hook in self.train_hooks:
            losses.append(hook.forward())#TODO targets=['d']
        return losses

    def calculate_gradients(self, targets=['d','g']):
        g_grads = []
        if 'g' in targets:
            if( self.config.train_g_every is None or
                (self.gan.steps % self.config.train_g_every == 0)):
                g_grads = self.g_grads()
            if (self.config.pretrain_d is not None and
                self.config.pretrain_d > self.gan.steps[0]):
                g_grads = []

        d_grads = []
        if 'd' in targets:
            if ( self.config.train_d_every is None or
               ( self.gan.steps % self.config.train_d_every == 0)):
                d_grads = self.d_grads()

            if (self.config.pretrain_d is not None and
                self.config.pretrain_d > self.gan.steps):
                d_grads = self.d_grads()

        return d_grads, g_grads

    def train_d(self, grads):
        if(len(grads) == 0):
            return

        for hook in self.train_hooks:
            d_grads, _ = hook.gradients(grads, [])
        for p, np in zip(self.gan.d_parameters(), grads):
            p.grad = np

        self.d_optimizer.step()

    def train_g(self, grads):
        if(len(grads) == 0):
            return

        for hook in self.train_hooks:
            _, grads = hook.gradients([], grads)
        for p, np in zip(self.gan.g_parameters(), grads):
            p.grad = np

        self.g_optimizer.step()

    def _step(self, feed_dict):
        metrics = self.gan.metrics()

        self.before_step(self.current_step, feed_dict)

        d_grads, _ = self.calculate_gradients(['d'])
        self.train_d(d_grads)
        _, g_grads = self.calculate_gradients(['g'])
        self.train_g(g_grads)

        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)


    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

