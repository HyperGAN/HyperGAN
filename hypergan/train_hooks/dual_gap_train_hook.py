from hypergan.viewer import GlobalViewer
import copy
import hyperchamber as hc
import numpy as np
import inspect
import torch
from hypergan.gan_component import ValidationException, GANComponent
from torch.autograd import grad as torch_grad
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

from hypergan.viewer import GlobalViewer

class DualGapTrainHook(BaseTrainHook):
    """ 
    https://arxiv.org/pdf/2103.12685v1.pdf
    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.g_copy = None
        self.d_copy = None
        self.loss = self.gan.create_component("loss")

    def parameters(self):
        return []

    def forward(self, d_loss, g_loss):
        if self.g_copy is None:
            defn = self.config.optimizer.copy()
            klass = GANComponent.lookup_function(None, defn['class'])
            del defn["class"]
            self.g_copy = self.gan.initialize_component("generator", self.gan.latent)
            self.d_copy = self.gan.initialize_component("discriminator")
            self.doptim = klass(self.d_copy.parameters(), **defn)
            self.goptim = klass(self.g_copy.parameters(), **defn)
        self.g_copy.set_trainable(True)
        self.d_copy.set_trainable(True)

        for p, copyp in zip(self.gan.discriminator.parameters(), self.d_copy.parameters()):
            copyp.data.copy_(p.data.clone().detach())
        for p, copyp in zip(self.gan.generator.parameters(), self.g_copy.parameters()):
            copyp.data.copy_(p.data.clone().detach())


        dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
        dreal = self.d_copy(self.gan.x).mean()
        dloss = self.loss.forward(dreal, dfake)
        gloss = None
        #gnorm = None
        #dnorm = None
        i = 0
        dnorm = 0
        gnorm = 0
        dsteps = 0
        gsteps = 0

        for j in range(self.config.steps or 10):
            self.goptim.zero_grad()
            self.doptim.zero_grad()
            if dloss is None or \
                dloss[0] >= self.config.dlosscutoff:
                dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
                dreal = self.d_copy(self.gan.x).mean()
                dloss = self.loss.forward(dreal, dfake)
                d_grads = torch_grad(dloss[0], self.d_copy.parameters(), create_graph=True, retain_graph=True)
                for p, g in zip(self.d_copy.parameters(), d_grads):
                    p.grad = g
                self.doptim.step()
                dsteps += 1
                #dnorm = sum([p.grad.norm() for p in self.d_copy.parameters()])
                #dnorm = torch.max(torch.cat([torch.flatten(p.grad) for p in self.d_copy.parameters()], 0).abs())
            if gloss is None or \
                gloss[0] <= self.config.glosscutoff:
                gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
                greal = self.gan.discriminator(self.gan.x).mean()
                gloss = self.loss.forward(greal, gfake)
                g_grads = torch_grad(gloss[0], self.g_copy.parameters(), create_graph=True, retain_graph=True)
                for p, g in zip(self.g_copy.parameters(), g_grads):
                    p.grad = -g
                self.goptim.step()
                gsteps+=1
                #gnorm = sum([p.grad.norm() for p in self.g_copy.parameters()])
                #gnorm = torch.max(torch.cat([torch.flatten(p.grad) for p in self.g_copy.parameters()], 0).abs())
            i+=1
            if self.config.cutoff and dnorm < self.config.cutoff:
                break
            if self.config.glosscutoff and gloss[0] > self.config.glosscutoff \
                and self.config.dlosscutoff and dloss[0] < self.config.dlosscutoff:
                break
            #print(" %d -> dl %.2e maxgrad %.2e gl %.2e maxgrad %.2e " % (i, dloss[0], dnorm, gloss[0], gnorm))
        #print("/ %d -> dl %.2e maxgrad %.2e gl %.2e maxgrad %.2e " % (i, dloss[0], dnorm, gloss[0], gnorm))
        self.gan.add_metric('dworst', dloss[0])
        self.gan.add_metric('gworst', gloss[0])
        self.gan.add_metric('Dsteps', dsteps)
        self.gan.add_metric('Gsteps', gsteps)
        dfake = self.d_copy(self.gan.generator(self.gan.latent.instance))
        dreal = self.d_copy(self.gan.x)
        dloss = self.loss._forward(dreal, dfake)
        gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance))
        greal = self.gan.discriminator(self.gan.x)
        gloss = self.loss._forward(greal, gfake)

        #self.losses = [dloss[0] - gloss[0], dloss[0] - gloss[0]]
        #self.losses = [gloss[0] - dloss[0], gloss[0] - dloss[0]]
        loss = (gloss[0]-dloss[0]).mean()
        #m = (dloss[0] + gloss[0])/2
        #loss = gloss[0] * torch.log(gloss[0]/m) + m * torch.log(m/(dloss[0]))
        #loss = gloss[0] * torch.log(gloss[0]/dloss[0])
        self.losses = [loss, loss]
        self.gan.add_metric('DG', self.losses[0])

        self.g_copy.set_trainable(False)
        self.d_copy.set_trainable(False)
        return self.losses

