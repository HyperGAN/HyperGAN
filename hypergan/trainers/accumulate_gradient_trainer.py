import numpy as np
import torch
import hyperchamber as hc
import inspect
from torch.autograd import Variable

from hypergan.trainers.alternating_trainer import AlternatingTrainer

class AccumulateGradientTrainer(AlternatingTrainer):
    """ G gradients accumulate over many D steps """
    def _create(self):
        self.d_optimizer = self.create_optimizer("d_optimizer")
        self.g_optimizer = self.create_optimizer("g_optimizer")
        self.accumulated_g_grads = None
        self.accumulation_steps = 0
        self.relu = torch.nn.ReLU()

    def calculate_gradients(self):
        accumulate = (self.config.accumulate or 3)
        if self.accumulation_steps == accumulate:
            g_grads = self.accumulated_g_grads
            d_grads = []
            #print("G_G", sum([g.abs().sum() for g in self.accumulated_g_grads[0]]), len(self.accumulated_g_grads))
            self.accumulated_g_grads = None
            self.accumulation_steps = 0
        else:
            gs = self.g_grads()
            if self.accumulated_g_grads is None:
                self.accumulated_g_grads = [g.clone()/accumulate for g in gs]
            else:
                for i, g in enumerate(self.accumulated_g_grads):
                    if self.config.type == 'agree':
                        self.accumulated_g_grads[i] = (self.accumulated_g_grads[i] + gs[i].clone()/accumulate) * self.relu(torch.sign(self.accumulated_g_grads[i]*gs[i].clone()))
                    else:
                        self.accumulated_g_grads[i] += gs[i].clone() / accumulate

            #print("D_G", sum([g.abs().sum() for g in gs]), len(self.accumulated_g_grads))

            d_grads = self.d_grads()
            g_grads = []
            self.accumulation_steps += 1

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


