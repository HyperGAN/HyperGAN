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
from info_nce import InfoNCE

class SVDTrainHook(BaseTrainHook):
    """ Trains a classifier for svd interpolations of latent parameters. """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.config.svd_classifier:
            self.svd_classifier = gan.create_component("svd_classifier", defn=self.config.svd_classifier, input=torch.zeros_like(gan.latent.next()))
        if self.config.info_nce:
            self.info_nce = InfoNCE()

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

        d_l = []
        g_l = []
        if self.config.info_nce:
            selections = torch.randint(0, e.shape[0], [self.gan.batch_size()]).cuda()
            z2 = self.gan.augmented_latent - (0.5 * e[selections])
            z3 = self.gan.augmented_latent + (0.5 * e[selections])
            g2 = self.gan.generator(z2)
            g = self.gan.g
            g3 = self.gan.generator(z3)
            p1 = torch.cat([g2, g, g3], dim=1)
            positive  = self.svd_classifier(p1)
            query = torch.randn_like(positive)
            negatives = []
            for i in range(2):
                selections2 = torch.randint(0, e.shape[0], [self.gan.batch_size()]).cuda()
                z3f = self.gan.augmented_latent - (0.5 * e[selections2])
                z4f = self.gan.augmented_latent + (0.5 * e[selections2])
                g3f = self.gan.generator(z3f)
                g4f = self.gan.generator(z4f)
                n1 = torch.cat([g3f, g, g4f], dim=1)
                negative = self.svd_classifier(n1)
                negatives.append(negative)
            negatives = torch.cat(negatives, dim=0)
            loss = self.info_nce(query, positive, negatives)
            d_l.append(loss)
            g_l.append(loss)
            self.gan.add_metric("infoNCE", loss)
        elif self.config.svd_classifier:
            i = []
            selections = torch.randint(0, e.shape[0], [self.gan.batch_size()]).cuda()
            z2 = self.gan.augmented_latent - (0.5 * e[selections])
            z3 = self.gan.augmented_latent + (0.5 * e[selections])
            p = self.gan.generator(z2)
            f = self.gan.generator(z3)
            g = self.gan.g
            x = torch.cat([p,g,f], dim=1)
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
