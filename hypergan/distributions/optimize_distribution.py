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
        klass = GANComponent.lookup_function(None, self.config['source']['class'])
        self.source = klass(gan, config['source'])
        self.current_channels = config['source']["z"]
        self.current_width = 1
        self.current_height = 1
        self.current_input_size = config['source']["z"]
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
        defn = self.config.optimizer.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]
        self.optimizer = klass([self.z_var], **defn)
        z = self.z_var
        z.data.copy_(sample)

        z.grad = torch.zeros_like(z)
        for i in range(self.config.steps or 1):
            self.optimizer.zero_grad()
            fake = self.gan.discriminator(self.gan.generator(self.hardtanh(z))).mean()
            real = self.gan.discriminator(self.gan.inputs.sample).mean()
            loss = self.gan.loss.forward_adversarial_norm(real, fake)
            if loss == 0.0:
                if self.config.verbose:
                    print("[optimize distribution] No loss")
                break
            z_move = torch_grad(outputs=loss, inputs=z, retain_graph=True, create_graph=True)
            z_change = z_move[0].abs().mean()
            if self.config.z_change_threshold or self.config.verbose:
                if i == 0:
                    first_z_change = z_change
            z._grad.copy_(z_move[0])
            self.optimizer.step()
            if self.config.verbose:
                print("[optimize distribution]", i, "loss", loss.item(), "mean movement", (z-sample).abs().mean().item(), (z_change/first_z_change).item())
            if self.config.z_change_threshold and z_change/first_z_change < self.config.z_change_threshold:
                if self.config.info:
                    print("[optimize distribution] z_change_threshold steps", i, "loss", loss.item(), "mean movement", (z-sample).abs().mean().item(), (z_change/first_z_change).item())
                break
            if self.config.loss_threshold and loss < self.config.loss_threshold:
                if self.config.info:
                    print("[optimize distribution] loss_threshold steps", i, "loss", loss.item(), "mean movement", (z-sample).abs().mean().item(), (z_change/first_z_change).item())
                break

        if self.config.info and i == self.config.steps-1:
            print("[optimize distribution] steps_threshold steps", i, "loss", loss.item(), "mean movement", (z-sample).abs().mean().item())
        self.instance = z
        return z
