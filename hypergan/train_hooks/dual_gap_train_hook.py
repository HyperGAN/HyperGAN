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
            self.goptim2 = klass(self.gan.generator.parameters(), **defn)

        for p, copyp in zip(self.gan.discriminator.parameters(), self.d_copy.parameters()):
            copyp.data.copy_(p.data.clone().detach())
        for p, copyp in zip(self.gan.generator.parameters(), self.g_copy.parameters()):
            copyp.data.copy_(p.data.clone().detach())


        dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
        dreal = self.d_copy(self.gan.x).mean()
        dloss = self.loss.forward(dreal, dfake)
        gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
        greal = self.gan.discriminator(self.gan.x).mean()
        gloss = self.loss.forward(greal, gfake)
        gsteps = 0
        dsteps = 0

        for i in range(self.config.steps or 10):
            self.goptim.zero_grad()
            self.goptim2.zero_grad()
            self.doptim.zero_grad()
            cont = False

            if gloss is None or self.config.glosscutoff < gloss[0]:
                #self.gan.latent.instance = self.gan.latent.next()
                gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
                greal = self.gan.discriminator(self.gan.x).mean()
                gloss = self.loss.forward(greal, gfake)
                #dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
                #dreal = self.d_copy(self.gan.x).mean()
                #dloss = self.loss.forward(dreal, dfake)
                g_grads = torch_grad(gloss[0], self.g_copy.parameters(), create_graph=True, retain_graph=True)
                for p, g in zip(self.g_copy.parameters(), g_grads):
                    p.grad = g.detach()
                self.goptim.step()
                if self.config.train_generator:
                    for p, g in zip(self.gan.generator.parameters(), g_grads):
                        p.grad = g.detach()
                    self.goptim2.step()
                gsteps += 1
                cont = True

            if dloss is None or self.config.dlosscutoff > dloss[0]:
                #print("DLOSS !! gloss %.2f dloss %.2f gstep %d dstep %d" % (gloss[0], dloss[0], gsteps, dsteps))
                dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
                dreal = self.d_copy(self.gan.x).mean()
                dloss = self.loss.forward(dreal, dfake)
                d_grads = torch_grad(dloss[0], self.d_copy.parameters(), create_graph=True, retain_graph=True)
                for p, g in zip(self.d_copy.parameters(), d_grads):
                    p.grad = -g.detach()
                self.doptim.step()
                dsteps += 1
                cont = True
            if cont == False:
                break
            if i > 100:
                print("gloss %.2f dloss %.2f gstep %d dstep %d" % (gloss[0], dloss[0], gsteps, dsteps))
        print("gloss %.2f dloss %.2f gstep %d dstep %d %.2f %.2e" % (gloss[0], dloss[0], gsteps, dsteps, self.gan.x.mean(), self.gan.latent.instance.mean()))
        self.gan.add_metric('dworst', dloss[0])
        self.gan.add_metric('gworst', gloss[0])
        self.gan.add_metric('dstep', dsteps)
        self.gan.add_metric('gstep', gsteps)
        dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
        dreal = self.d_copy(self.gan.x).mean()
        dloss = self.loss._forward(dreal, dfake)
        gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance))
        greal = self.gan.discriminator(self.gan.x)
        gloss = self.loss._forward(greal, gfake)

        #loss = (gloss[0]-dloss[0]).mean()
        #loss = (dloss[0]-gloss[0]).mean()
        if self.config.kl:
            loss = (dloss[0] * torch.log(dloss[0]/gloss[0])).mean()
        else:
            loss = (dloss[0]-gloss[0]).mean()
        self.losses = [loss, loss]
        self.gan.add_metric('DG', self.losses[0])
        if self.gan.config.cache_inputs:
            if dsteps != 0:
                self.gan.x = self.gan.inputs.next()
                print("Next X")
                self.gan.latent.instance = self.gan.latent.next()
            if dsteps > 0:
                return [loss, loss * 0]
            else:
                return [loss * 0, loss]
        return [loss, loss]

    def state_dict(self):
        return {}

