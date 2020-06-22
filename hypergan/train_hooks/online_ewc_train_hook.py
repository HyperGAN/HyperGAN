from copy import deepcopy

import hyperchamber as hc
import numpy as np
import inspect
import torch
from torch.nn.parameter import Parameter
from torch.autograd import grad as torch_grad
import torch.nn as nn
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class OnlineEWCTrainHook(BaseTrainHook):
  """ https://faculty.washington.edu/ratliffl/research/2019conjectures.pdf """
  def __init__(self, gan=None, config=None):
      super().__init__(config=config, gan=gan)
      self.d_loss = None
      self.g_loss = None
      self.gan = gan

      self.d_ewc_params = []
      self.d_ewc_fisher = []

      for p in self.gan.d_parameters():
          self.d_ewc_params += [Parameter(p, requires_grad=False)]
          self.d_ewc_fisher += [Parameter(torch.rand(p.shape).cuda(), requires_grad=False)]

      self.g_ewc_params = []
      self.g_ewc_fisher = []

      for p in self.gan.g_parameters():
          self.g_ewc_params += [Parameter(p, requires_grad=False)]
          self.g_ewc_fisher += [Parameter(torch.rand(p.shape).cuda(), requires_grad=False)]

      for i, (param, fisher) in enumerate(zip(self.d_ewc_params, self.d_ewc_fisher)):
          self.register_parameter('d_ewc'+str(i), param)
          self.register_parameter('d_fisher'+str(i), fisher)

      for i, (param, fisher) in enumerate(zip(self.g_ewc_params, self.g_ewc_fisher)):
          self.register_parameter('g_ewc'+str(i), param)
          self.register_parameter('g_fisher'+str(i), fisher)



  def forward(self):

      if self.config.skip_after_steps and self.config.skip_after_steps < self.gan.steps:
          return [None, None]

      d_loss = self.gan.trainer.d_loss
      self.d_loss = 0
      if d_loss is not None:
          d_loss = d_loss.mean()
          d_params = list(self.gan.d_parameters())
          d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
          mean_decay = self.config.d_mean_decay or self.config.mean_decay
          for i, (dp, dp_g) in enumerate(zip(d_params, d_grads)):
              self.d_loss += (self.config.beta or 1.0) * ((dp - self.d_ewc_params[i]) ** 2 * self.d_ewc_fisher[i]).sum()
              with torch.no_grad():
                  self.d_ewc_fisher[i] = self.config.gamma * self.d_ewc_fisher[i] + dp_g**2
                  self.d_ewc_params[i] = (1.0-mean_decay) * dp.clone() + mean_decay * self.d_ewc_params[i]
          self.gan.add_metric('ewc_d', self.d_loss)

      skip_g_after_steps = False
      if self.config.skip_g_after_steps:
          skip_g_after_steps = self.config.skip_g_after_steps < self.gan.steps
      skip_g = self.config.skip_g or skip_g_after_steps
      if skip_g:
          return [self.d_loss, None]

      self.g_loss = 0
      g_loss = self.gan.trainer.g_loss
      if g_loss is not None:
          g_loss = g_loss.mean()
          g_params = list(self.gan.g_parameters())
          g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
          mean_decay = self.config.g_mean_decay or self.config.mean_decay
          for i, (gp, gp_g) in enumerate(zip(g_params, g_grads)):
              self.g_loss += (self.config.beta or 1.0) * ((gp - self.g_ewc_params[i]) ** 2 * self.g_ewc_fisher[i]).sum()
              with torch.no_grad():
                  self.g_ewc_fisher[i] = self.config.gamma * self.g_ewc_fisher[i] + gp_g**2
                  self.g_ewc_params[i] = (1.0-mean_decay) * gp.clone() + mean_decay * self.g_ewc_params[i]
          self.gan.add_metric('ewc_g', self.g_loss)

      return [self.d_loss, self.g_loss]

