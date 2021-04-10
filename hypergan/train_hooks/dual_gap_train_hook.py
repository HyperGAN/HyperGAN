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
            copyp.grad = torch.zeros_like(copyp.data)
        for p, copyp in zip(self.gan.generator.parameters(), self.g_copy.parameters()):
            copyp.data.copy_(p.data.clone().detach())
            copyp.grad = torch.zeros_like(copyp.data)


        dloss = None
        gloss = None

        for i in range(self.config.steps or 10):
            dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
            dreal = self.d_copy(self.gan.x).mean()
            dloss = self.loss.forward(dreal, dfake)
            d_grads = torch_grad(dloss[1], self.d_copy.parameters(), create_graph=True, retain_graph=True)
            for p, g in zip(self.d_copy.parameters(), d_grads):
                p.grad = g
            self.doptim.step()
            gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
            greal = self.gan.discriminator(self.gan.x).mean()
            gloss = self.loss.forward(greal, gfake)
            g_grads = torch_grad(gloss[0], self.g_copy.parameters(), create_graph=True, retain_graph=True)
            for p, g in zip(self.d_copy.parameters(), d_grads):
                p.grad = g
            #self.goptim.step()
            print(" %d -> dl %.2e gl %.2e " % (i, dloss[1], gloss[0]))
        dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
        dreal = self.d_copy(self.gan.x).mean()
        dloss = self.loss.forward(dreal, dfake)
        gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
        greal = self.gan.discriminator(self.gan.x).mean()
        gloss = self.loss.forward(greal, gfake)

        #self.losses = [dloss[0] - gloss[0], dloss[0] - gloss[0]]
        self.losses = [gloss[0] - dloss[0], gloss[0] - dloss[0]]

        self.g_copy.set_trainable(False)
        self.d_copy.set_trainable(False)
        return self.losses

