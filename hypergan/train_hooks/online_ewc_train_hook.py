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
              self.d_ewc_params += [Variable(p, requires_grad=False)]
              self.d_ewc_g += [Variable(torch.zeros_like(p), requires_grad=False)]

          self.g_ewc_params = []
          self.g_ewc_g = []

          for p in self.gan.g_parameters():
              self.g_ewc_params += [Variable(p, requires_grad=False)]
              self.g_ewc_g += [Variable(torch.zeros_like(p), requires_grad=False)]
      
      d_loss = self.gan.trainer.d_loss.mean()
      d_params = list(self.gan.d_parameters())
      g_params = list(self.gan.g_parameters())
      d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
      self.d_loss = 0
      decay = (self.config.d_decay or self.config.decay or 0.1)
      grad_decay = (self.config.d_grad_decay or self.config.grad_decay or 1.0)
      for i, (dp, dp_g) in enumerate(zip(d_params, d_grads)):
          self.d_loss += (self.config.gamma or 1.0) * ((dp - self.d_ewc_params[i]) ** 2 * self.d_ewc_g[i]).sum()
          with torch.no_grad():
              self.d_ewc_g[i] = (1.0-grad_decay) * (self.d_ewc_g[i]**2) + dp_g**2
              self.d_ewc_params[i] = decay*dp + (1.0-decay)*self.d_ewc_params[i]
      self.gan.add_metric('ewc_d', self.d_loss)
      self.d_loss = torch.min(torch.tensor([self.d_loss, 100.0]))
      if self.config.skip_g:
          return [self.d_loss, None]
      self.g_loss = 0
      g_loss = self.gan.trainer.g_loss.mean()
      g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
      decay = (self.config.g_decay or self.config.decay or 0.1)
      grad_decay = (self.config.g_grad_decay or self.config.grad_decay or 1.0)
      for i, (gp, gp_g) in enumerate(zip(g_params, g_grads)):
          self.g_loss += (self.config.gamma or 1.0) * ((gp - self.g_ewc_params[i]) ** 2 * self.g_ewc_g[i]).sum()
          with torch.no_grad():
              self.g_ewc_g[i] = (1.0-grad_decay) * (self.g_ewc_g[i]**2) + gp_g**2
              self.g_ewc_params[i] = decay*gp + (1.0-decay)*self.g_ewc_params[i]
      self.g_loss = torch.min(torch.tensor([self.g_loss, 100.0]))
 
      self.gan.add_metric('ewc_g', self.g_loss)

      return [self.d_loss, self.g_loss]

