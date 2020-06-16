from copy import deepcopy

import hyperchamber as hc
import numpy as np
import inspect
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class OnlineEWCTrainHook(BaseTrainHook):
  """ https://faculty.washington.edu/ratliffl/research/2019conjectures.pdf """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.gan = gan


  def forward(self):
      if not hasattr(self, 'd_ewc_params'):
          self.d_ewc_params = []
          self.d_ewc_g = []

          for p in self.gan.d_parameters():
              self.d_ewc_params += [Variable(p)]
              self.d_ewc_g += [Variable(torch.zeros_like(p))]

          self.g_ewc_params = []
          self.g_ewc_g = []

          for p in self.gan.g_parameters():
              self.g_ewc_params += [Variable(p)]
              self.g_ewc_g += [Variable(torch.zeros_like(p))]
      
      g_loss = self.gan.trainer.g_loss.mean()
      d_loss = self.gan.trainer.d_loss.mean()
      d_params = list(self.gan.d_parameters())
      g_params = list(self.gan.g_parameters())
      d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
      self.d_loss = 0
      with torch.no_grad():
          for i, (dp, dp_g) in enumerate(zip(d_params, d_grads)):
              self.d_ewc_g[i] += dp_g
              self.d_loss += (self.config.gamma or 1.0) * ((dp - self.d_ewc_params[i]) ** 2 * self.d_ewc_g[i] ** 2).sum()
              self.d_ewc_g[i] += dp_g
              self.d_ewc_params[i] = 0.5*dp + 0.5*self.d_ewc_params[i]
      self.g_loss = 0
      g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
      with torch.no_grad():
          for i, (gp, gp_g) in enumerate(zip(g_params, g_grads)):
              self.g_ewc_g[i] += gp_g.detach()
              self.g_loss += (self.config.gamma or 1.0) * ((gp - self.g_ewc_params[i]) ** 2 * self.g_ewc_g[i] ** 2).sum()
              self.g_ewc_g[i] += gp_g.detach()
              self.g_ewc_params[i] = 0.5*gp + 0.5*self.g_ewc_params[i]
 
      self.gan.add_metric('ewc_d', self.d_loss)
      self.gan.add_metric('ewc_g', self.g_loss)

      return [self.d_loss, self.g_loss]

