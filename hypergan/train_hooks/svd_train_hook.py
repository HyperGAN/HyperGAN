import torch
import random
from hypergan.losses.stable_gan_loss import StableGANLoss
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class SVDTrainHook(BaseTrainHook):
    """ Trains a classifier for svd interpolations of latent parameters. """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.svd_classifier = gan.create_component("svd_classifier", defn=self.config.svd_classifier, input=torch.zeros_like(gan.latent.next()))
        self.z_stable_loss = StableGANLoss(gan=self.gan, gammas=self.config.gammas, offsets=self.config.offsets, metric_name="z")

    def gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u

        nk = vv.size(0)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[:, 0] = vv[:, 0].clone()
        for k in range(1, nk):
            vk = vv[k].clone()
            uk = 0
            for j in range(0, k):
                uj = uu[:, j].clone()
                uk = uk + projection(uj, vk)
            uu[:, k] = vk - uk
        for k in range(nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
        return uu


    def forward(self, d_loss, g_loss):
        z = self.gan.latent.z
        g_params = self.gan.latent_parameters()
        eigvecs = [torch.linalg.svd(g_p) for g_p in g_params]
        e = eigvecs[0][2]
        if self.config.type == "orthogonal":
            xsch = self.gram_schmidt(g)
            x = xsch / xsch.abs().sum()

        d_l = []
        g_l = []
        if self.config.svd_classifier:
            i = []
            selections = torch.randint(0, e.shape[0], [self.gan.batch_size()]).cuda()
            z2 = self.gan.augmented_latent - (0.5 * e[selections])
            z3 = self.gan.augmented_latent + (0.5 * e[selections])
            p = self.gan.generator(z2)
            f = self.gan.generator(z3)
            #g = self.gan.g
            x = torch.cat([p,f], dim=1)
            #g2 = torch.cat([p,g], dim=1)
            logits = self.svd_classifier(x)
            l = self.ce_loss(logits, selections).sum()
            g_l += [l]
            d_l += [l]
            self.gan.add_metric('vd', g_l[0])
            self.gan.add_metric('vg', d_l[0])
        return sum(d_l), sum(g_l)

    def generator_components(self):
        return []

    def discriminator_components(self):
        if self.config.svd_classifier:
            return [self.svd_classifier]
        return []
