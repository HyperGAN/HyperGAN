import torch
import collections
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from hypergan.gan_component import ValidationException, GANComponent
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class StableGANLoss2TrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.gamma1 = torch.tensor(10.0, device=gan.device)
        self.g_gamma1 = torch.tensor(10.0, device=gan.device)
        self.gamma2 = torch.tensor(10.0, device=gan.device)
        self.g_gamma2 = torch.tensor(0.1, device=gan.device)
        self.inverse_gamma = torch.tensor(1e3, device=gan.device)
        self.target1_x = None
        self.target1_g = None
        self.target2_x = None
        self.target2_g = None
        if self.config.visualize:
            self.gan.adversaries = [torch.zeros_like(self.gan.inputs.next()) for i in range(8)]

    def loss_fn(self, d_real, d_fake):
        #cr = torch.mean(d_real,0)
        #cf = torch.mean(d_fake,0)
        criterion = torch.nn.BCEWithLogitsLoss()
        #g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
        #d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        #g_loss = torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        #d_loss = torch.log(torch.sigmoid(d_real) + 1e-13) +torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        return d_loss, g_loss

 
    def forward(self, d_loss, g_loss):
        discriminator = self.gan.forward_discriminator
        xs = [self.gan.x]
        gs = [self.gan.g]
        d_fake = self.gan.d_fake
        d_real = self.gan.d_real

        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_losses = []
        g_losses = []
        d_loss, g_loss = self.loss_fn(d_real, d_fake)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        if self.target1_x == None:
            self.target1_g = [Parameter(g, requires_grad=True) for g in gs]
            self.target1_x = [Parameter(x, requires_grad=True) for x in xs]
            self.target2_g = [Parameter(g, requires_grad=True) for g in gs]
            self.target2_x = [Parameter(x, requires_grad=True) for x in xs]
        for target, g in zip(self.target1_g + self.target2_g, gs + gs):
            target.data = g.clone()
        for target, x in zip(self.target1_x + self.target2_x, xs + xs):
            target.data = x.clone()

        neg_inverse_fake = self.target2_g
        neg_inverse_real = self.target2_x
        target2_g_l = None
        target2_g_l_init = None
        target2_g_l_diff = None
        target2_x_l = None
        target2_g_l_init = None
        target2_x_l_diff = None
        for i in range(self.config.iterations or 1):
            l, neg_inverse_fake = self.inverse(discriminator(*neg_inverse_fake), d_real, neg_inverse_fake, negate_loss=True)
            if target2_g_l_init is None:
                target2_g_l_init = l.abs()
            if target2_g_l is None:
                target2_g_l_diff = 100000
            else:
                target2_g_l_diff = (l-target2_g_l).abs()
            target2_g_l = l
            self.gan.add_metric('nifl', l)
            if self.config.visualize:
                self.gan.adversaries[2] = (neg_inverse_fake[0] - self.gan.g)
            neg_inverse_fake = [Parameter(g, requires_grad=True) for g in neg_inverse_fake]
            l, neg_inverse_real = self.inverse(d_fake, discriminator(*neg_inverse_real), neg_inverse_real, negate_loss=True)
            self.gan.add_metric('nirl', l)
            if self.config.visualize:
                self.gan.adversaries[4] = (neg_inverse_real[0] - self.gan.x)
            neg_inverse_real = [Parameter(g, requires_grad=True) for g in neg_inverse_real]
            if target2_g_l_diff/target2_g_l_init < (self.config.threshold or 0.01):
                print(i, target2_g_l_diff/target2_g_l_init)
                break

        reg_fake, _ = self.loss_fn(d_real, discriminator(*neg_inverse_fake))
        reg_real, g_ = self.loss_fn(discriminator(*neg_inverse_real), d_fake)

        d_losses.append(self.gamma2*(reg_fake+reg_real))
        g_losses.append(self.g_gamma2 * g_)

        inverse_fake = self.target1_g
        inverse_real = self.target1_x
        if self.config.visualize:
            self.gan.adversaries[0] = self.gan.g
            self.gan.adversaries[3] = self.gan.x
        target1_g_l = None
        target1_g_l_init = None
        target1_g_l_diff = None
        target1_x_l = None
        target1_x_l_init = None
        target1_x_l_diff = None
        for i in range(self.config.iterations or 1):

            l, inverse_fake = self.inverse(d_real, discriminator(*inverse_fake), inverse_fake, negate_loss=True)
            if target1_g_l_init is None:
                target1_g_l_init = l.abs()
            if target1_g_l is None:
                target1_g_l_diff = 100000
            else:
                target1_g_l_diff = (l-target1_g_l).abs()
            target1_g_l = l
            self.gan.add_metric('ifl', l)
            if self.config.visualize:
                self.gan.adversaries[1] = (inverse_fake[0] - self.gan.g)
            inverse_fake = [Parameter(g, requires_grad=True) for g in inverse_fake]
            l, inverse_real = self.inverse(discriminator(*inverse_real), d_fake, inverse_real, negate_loss=True)
            self.gan.add_metric('irl', l)
            if self.config.visualize:
                self.gan.adversaries[5] = (inverse_real[0] - self.gan.x)
            inverse_real = [Parameter(g, requires_grad=True) for g in inverse_real]
            if target1_g_l_diff/target1_g_l_init < (self.config.threshold or 0.01):
                break

        if self.config.visualize:
            self.gan.adversaries[6] = (self.gan.adversaries[5] - self.gan.adversaries[4])
            self.gan.adversaries[7] = (self.gan.adversaries[5] + self.gan.adversaries[4])*5
        reg_fake, g_ = self.loss_fn(discriminator(*inverse_fake), d_fake)
        reg_real = self.loss_fn(d_real, discriminator(*inverse_real))[0]

        d_losses.append(self.gamma1*(reg_fake+reg_real))
        g_losses.append(self.g_gamma1 * g_)

        return sum(d_losses), sum(g_losses)

    def inverse(self, d_real, d_fake, target, negate_loss=False):
        if self.config.inverse_type == "l2":
            loss = torch.abs(d_fake - d_real) * self.inverse_gamma
        else:
            loss = self.loss_fn(d_fake, d_real)[0] * self.inverse_gamma
        if negate_loss:
            loss = -loss.mean()
        else:
            loss = loss.mean()
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        #return [_t + _d1/_d1.norm() for _d1, _t in zip(d1_grads, target)]
        return loss, [_t + _d1 for _d1, _t in zip(d1_grads, target)]
