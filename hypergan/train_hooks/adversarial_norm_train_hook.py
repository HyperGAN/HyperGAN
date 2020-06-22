import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter

class AdversarialNormTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.gamma = torch.Tensor([self.config.gamma]).float()[0].cuda()#self.gan.configurable_param(self.config.gamma or 1.0)
      self.relu = torch.nn.ReLU()
      if self.config.target == 'x' or self.config.target is None:
          self.target = Parameter(self.gan.inputs.next(), requires_grad=True)
      elif self.config.target == 'g':
          self.target = Parameter(self.gan.inputs.next(), requires_grad=True)

  def forward(self):
      if self.config.target == 'x' or self.config.target is None:
          self.target.data = self.gan.x.data.clone()
          loss, norm = self.gan.regularize_adversarial_norm(self.gan.discriminator(self.target), self.gan.d_fake, self.target)
      elif self.config.target == 'g':
          self.target.data = self.gan.g.data.clone()
          loss, norm = self.gan.regularize_adversarial_norm(self.gan.d_real, self.gan.discriminator(self.target), self.target)

      if loss is None:
          return [None, None]

      if self.config.loss:
        if "g" in self.config.loss:
            self.g_loss = self.gamma * norm.mean()
            self.gan.add_metric('gn_g', self.g_loss)
        if "d" in self.config.loss:
            self.d_loss = self.gamma * norm.mean()
            self.gan.add_metric('gn_d', self.d_loss)
      else:
        self.d_loss = self.gamma * norm.mean()
        self.gan.add_metric('gn_d', self.d_loss)

      return [self.d_loss, self.g_loss]
