from ..gan_component import ValidationException
from .base_distribution import BaseDistribution
from hypergan.gan_component import ValidationException, GANComponent
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.distributions import uniform
import hyperchamber as hc
import numpy as np
import torch

class OptimizeDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)
        klass = GANComponent.lookup_function(None, self.config['source'])
        self.source = klass(gan, config)
        self.current_channels = config["z"]
        self.current_width = 1
        self.current_height = 1
        self.current_input_size = config["z"]
        self.z = self.source.z
        self.hardtanh = torch.nn.Hardtanh()
        self.relu = torch.nn.ReLU()
        self.z_var = None

    def create(self):
        pass

    def next(self):
        sample = self.source.next()
        if self.z_var is None:
            self.z_var = Variable(sample, requires_grad=True).cuda()
            self.optimizer = torch.optim.SGD([self.z_var], lr=1.0)
        z = self.z_var
        optimizer = self.optimizer
        z.data = sample

        for i in range(self.config.steps or 1):
            optimizer.zero_grad()
            #z_move = torch_grad(outputs=loss, inputs=z, retain_graph=True, create_graph=True)
            #z.data -= z_move[0]*100
            #loss = -self.gan.discriminator(self.gan.generator(self.hardtanh(z))).mean()
            fake = self.gan.discriminator(self.gan.generator(self.hardtanh(z))).mean()
            real = self.gan.discriminator(self.gan.inputs.sample).mean()
            if self.config.formulation == "relu(d(x)-d(g))":
                loss = self.relu(real - fake)**2
            elif self.config.formulation == 'flex':
                loss = self.relu((real - fake).abs() - self.config.flex) ** 2 * (self.config.gamma or 1.0)
            elif self.config.formulation == 'flexforce':
                loss = ((real - fake).abs() - self.config.flex) ** 2 * (self.config.gamma or 1.0)
            elif self.config.formulation == 'flexforce2':
                loss = ((real - fake) - self.config.flex) ** 2 * (self.config.gamma or 1.0)
            else:
                uaeotnhu()
                loss = (real - fake)**2*1e8
            #if(loss.item() < 1e-4):
            #    break
            #print(loss)
            #loss = -self.gan.discriminator(self.gan.generator(self.hardtanh(z))).mean()
            loss.backward()
            optimizer.step()
        z.requires_grad=False

        return z
