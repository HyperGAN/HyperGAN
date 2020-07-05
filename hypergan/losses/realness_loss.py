import hyperchamber as hc
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from hypergan.losses.base_loss import BaseLoss

class CategoricalLoss(nn.Module):
    def __init__(self, atoms=51, v_max=1.0, v_min=-1.0):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms).cuda() # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).cuda().fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).cuda().view(-1, 1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).cuda()
        skewed_anchor = torch.zeros(batch_size, self.atoms).cuda()
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  

        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()

        return loss

class RealnessLoss(BaseLoss):
    """https://arxiv.org/pdf/2002.05512v1.pdf"""
    def __init__(self, gan, config):
        super(RealnessLoss, self).__init__(gan, config)

    def required(self):
        return "skew".split()

    def _forward(self, d_real, d_fake):
        num_outcomes = d_real.shape[1]
        if not hasattr(self, 'anchor0'):
            gauss = np.random.normal(0, 0.3, 1000)
            count, bins = np.histogram(gauss, num_outcomes)
            self.anchor0 = count / num_outcomes

            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            self.anchor1 = count / num_outcomes
            self.anchor_real = torch.zeros((self.gan.batch_size(), num_outcomes), dtype=torch.float).cuda() + torch.tensor(self.anchor1, dtype=torch.float).cuda()
            self.anchor_fake = torch.zeros((self.gan.batch_size(), num_outcomes), dtype=torch.float).cuda() + torch.tensor(self.anchor0, dtype=torch.float).cuda()
            self.Triplet_Loss = CategoricalLoss(num_outcomes)
        feat_real = d_real.log_softmax(1).exp()
        feat_fake = d_fake.log_softmax(1).exp()
        d_loss = self.Triplet_Loss(self.anchor_real, feat_real, skewness=self.config.skew[1]) + \
                 self.Triplet_Loss(self.anchor_fake, feat_fake, skewness=self.config.skew[0])

        g_loss = -self.Triplet_Loss(self.anchor_fake, feat_fake, skewness=self.config.skew[0])
        g_loss += self.Triplet_Loss(self.anchor_real, feat_fake, skewness=self.config.skew[0])

        return [d_loss, g_loss]

