#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class GradientPenaltyTrainHook(BaseTrainHook):
  def forward(self):
    x = self.gan.inputs.sample
    g = self.gan.generator_sample

    alpha = torch.rand(self.gan.batch_size(), 1, 1, 1).cuda()
    interpolated = alpha * x + (1 - alpha) * g
    interpolated = Variable(interpolated, requires_grad=True).cuda()

    gamma= self.gan.configurable_param(self.config['lambda'] or 1)

    d = self.gan.discriminator(interpolated)

    grad = torch_grad(outputs=d, inputs=interpolated,
                               grad_outputs=torch.ones(d.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

    grad = grad.view(self.gan.batch_size(), -1) 
    grad_loss = ((torch.sqrt(torch.sum(grad**2, dim=1) +1e-12) - 1) ** 2).mean()

    loss = gamma*grad_loss
    return [loss, loss]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
