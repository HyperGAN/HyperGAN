import numpy as np
import torch
import hyperchamber as hc
import inspect
from torch.autograd import Variable

from hypergan.trainers.alternating_trainer import AlternatingTrainer

TINY = 1e-12

class BalancedTrainer(AlternatingTrainer):
    """ G or D depending on which scores better """
    def _create(self):
        self.d_optimizer = self.create_optimizer("d_optimizer")
        self.g_optimizer = self.create_optimizer("g_optimizer")
        self.dcount = 0
        self.gcount = 0
        self.last_d_fake = None

    def required(self):
        return "".split()

    def calculate_gradients(self):
        d_real, d_fake = self.gan.forward_discriminator()
        self.add_metric("R", d_real.mean())
        self.add_metric("F", d_fake.mean())

        if (self.config.d_until or 0) > self.gan.steps:
            step_d = True
        elif self.config.d_fake_balance:
            if self.last_d_fake is None or d_fake.mean() > self.last_d_fake:
                step_d = True
            else:
                step_d = False
            self.last_d_fake = d_fake.mean()
        elif d_real.mean() < (d_fake.mean()+(self.config.imbalance or 0.1)):
            step_d = True
        else:
            step_d = False

        if step_d:
            d_grads = self.d_grads(d_real=d_real, d_fake=d_fake)
            g_grads = []

            self.dcount+=1
        else:
            d_grads = []
            g_grads = self.g_grads(d_real=d_real, d_fake=d_fake)
            self.gcount+=1
        self.gan.add_metric("dcount", self.dcount)
        self.gan.add_metric("gcount", self.gcount)

        return d_grads, g_grads


    def _step(self, feed_dict):
        metrics = self.gan.metrics()

        self.before_step(self.current_step, feed_dict)

        d_grads, g_grads = self.calculate_gradients()
        self.train_d(d_grads)
        self.train_g(g_grads)

        self.after_step(self.current_step, feed_dict)

        if self.current_step % 20 == 0:
            self.print_metrics(self.current_step)


