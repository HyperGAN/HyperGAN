#https://openreview.net/pdf?id=H1Yp-j1Cb
import hyperchamber as hc
import torch
import copy
import math
import random

from hypergan.losses.base_loss import BaseLoss
from torch.functional import F
import torch.nn as nn

def M(d_real, d_fake):
    log_real = torch.log(F.sigmoid(d_real)+1e-13)
    log_fake = torch.log(1 - F.sigmoid(d_fake)+1e-13)
    result = 0.5 * log_real + 0.5 * log_fake
    return result

class ChekhovLoss(BaseLoss):

    def __init__(self, gan, config):
        super(ChekhovLoss, self).__init__(gan, config)
        self.generators = nn.ModuleList([]).cuda()
        self.discriminators = nn.ModuleList([]).cuda()
        self.step = 0
        self.k = 3
        self.index=0

    def weights(self, component):
        parameters = list(component.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def reg(self, weights):
        return torch.norm(weights)

    def copy_data(self, m1, m2):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            p2.data = p1.data.clone()

    def mix_weights(self,sources, target):
        for i, p2 in enumerate(target.parameters()):
            full_data = p2.data.clone()
            for source in sources:
                p1 = list(source.parameters())[i]
                full_data = full_data + p1.data.clone()
            p2.data = full_data/(len(sources)+1)


    def mixed_generator(self, latent):
        self.mix_weights(self.generators + [self.gan.generator], self.sample_g)
        return self.sample_g(latent)


    def _forward(self, d_real, d_fake):
        d_regularize = self.reg(self.weights(self.gan.discriminator)) * 0.1/math.sqrt(self.step+1)
        self.gan.add_metric('dl', d_regularize)
        g_regularize = self.reg(self.weights(self.gan.generator)) * 0.1/math.sqrt(self.step+1)
        self.gan.add_metric('gl', g_regularize)
        if(len(self.generators) == 0):
            g = self.gan.initialize_component('generator').cuda()
            self.sample_g = g

        while len(self.generators) < self.k:
            print("Creating g for ", self.index, len(self.generators))
            g = self.gan.initialize_component('generator').cuda()
            self.generators.insert(0, g)
            d = self.gan.initialize_component('discriminator').cuda()
            self.discriminators.insert(0, d)


        vs = []
        for discriminator in self.discriminators:
            vs.append(M(discriminator(self.gan.x), discriminator(self.gan.g.clone())))

        vs.append(M(d_real, d_fake))
        us = []
        for generator in self.generators:
            us.append(-M(d_real, self.gan.discriminator(generator(self.gan.augmented_latent).clone().detach())))
        us.append(-M(d_real, d_fake))
        d_loss = sum(vs)/(len(vs))-d_regularize
        g_loss = sum(us)/(len(us))+g_regularize
        mod = 100
        if self.step > 1000:
            mod = 1000
        if (self.step % mod == 0 and self.step > 0):
            self.index = (self.index + 1) % self.k
            print("CLONING GENERATOR", self.index)
        g = self.generators[self.index]
        d = self.discriminators[self.index]
        self.copy_data(self.gan.generator, g)
        self.copy_data(self.gan.discriminator, d)


                #g.load_state_dict(copy.deepcopy(self.gan.generator.state_dict()))
                #d.load_state_dict(copy.deepcopy(self.gan.discriminator.state_dict()))

        self.step += 1


        return [-d_loss, -g_loss]

