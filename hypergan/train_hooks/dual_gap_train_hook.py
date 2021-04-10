from hypergan.viewer import GlobalViewer
import copy
import hyperchamber as hc
import numpy as np
import inspect
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

    def before_step(self, step, feed_dict, depth=0):
        defn = self.config.optimizer.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]
        if self.g_copy is None:
            self.g_copy = self.gan.initialize_component("generator", self.gan.latent)
            self.d_copy = self.gan.initialize_component("discriminator")

        for p, copyp in zip(self.gan.discriminator.parameters(), self.d_copy.parameters()):
            copyp.data = p.data

        for p, copyp in zip(self.gan.generator.parameters(), self.g_copy.parameters()):
            copyp.data = p.data

        self.doptim = klass(self.d_copy.parameters(), **defn)
        self.goptim = klass(self.g_copy.parameters(), **defn)

        dloss = None
        gloss = None

        for i in range(self.config.steps or 5):
            self.goptim.zero_grad()
            self.doptim.zero_grad()
            dfake = self.d_copy(self.gan.generator(self.gan.latent.instance)).mean()
            dreal = self.d_copy(self.gan.x).mean()
            dloss = self.loss.forward(dreal, dfake)
            dloss[0].backward(retain_graph=True)
            self.doptim.step()
            gfake = self.gan.discriminator(self.g_copy(self.gan.latent.instance)).mean()
            greal = self.gan.discriminator(self.gan.x).mean()
            gloss = self.loss.forward(greal, gfake)
            gloss[1].backward(retain_graph=True)
            self.goptim.step()

        self.losses = [dloss[0] - gloss[0], dloss[0] - gloss[0]]


    def forward(self, d_loss, g_loss):
        return self.losses

