#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn

import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class GradientPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      flex = 1
      if self.config.flex is not None:
          flex = self.config.flex
      self.flex = self.gan.configurable_param(flex)
      self.relu = nn.ReLU()

  def forward(self, d_loss, g_loss):
    x = self.gan.inputs.sample
    g = self.gan.generator_sample

    if self.config.input == 'x':
        inp = x
    elif self.config.input == 'g':
        inp = g
    else:
        alpha = torch.rand(self.gan.batch_size(), 1, 1, 1).cuda()
        inp = alpha * x + (1 - alpha) * g
    #inp = inp(interpolated, requires_grad=True).cuda()

    gamma= self.gan.configurable_param(self.config['lambda'] or 1)

    d = self.gan.discriminator(inp, context=self.gan.gp_context)
    parameters = list(self.gan.d_parameters())#+ [interpolated]

    grad = torch_grad(outputs=d.mean(), inputs=parameters, create_graph=True, retain_graph=True)

    grad = [_g.view(-1) for _g in grad]
    if self.config.type == "relu":
        grad_loss = [(self.relu(torch.sqrt(torch.sum(_g**2, dim=0) +1e-12) - self.flex) ** 2).mean() for _g in grad]
    else:
        grad_loss = [((torch.sqrt(torch.sum(_g**2, dim=0) +1e-12) - self.flex) ** 2).mean() for _g in grad]
    grad_loss = sum(grad_loss)

    loss = gamma*grad_loss
    return [loss, loss]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
