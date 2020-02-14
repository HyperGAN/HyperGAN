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

class CatchupGTrainHook(BaseTrainHook):
  """ https://github.com/rothk/Stabilizing_GANs """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.sig = torch.nn.Sigmoid().cuda()
      self.gamma = self.gan.configurable_param(self.config.gamma or 1.0)
      self.relu = torch.nn.ReLU()

  def forward(self):
      x = self.gan.inputs.sample
      g = self.gan.generator_sample
      d1_params = Variable(x, requires_grad=True).cuda()#self.gan.d_parameters()
      d2_params = Variable(g, requires_grad=True).cuda()#self.gan.g_parameters()
      d1_z_params = Variable(self.gan.gp_context_x["z"], requires_grad=True).cuda()
      d2_z_params = Variable(self.gan.gp_context_g["z"], requires_grad=True).cuda()
      d1_logits = self.gan.discriminator(d1_params, context={"z":d1_z_params})
      d2_logits = self.gan.discriminator(d2_params, context={"z":d2_z_params})
      d1 = self.sig(d1_logits)
      d2 = self.sig(d2_logits)
      params = list(self.gan.discriminator.parameters())# + [d1_logits]
      if self.config.components:
          params = []
          for component in self.config.components:
              params += list(getattr(self.gan, component).parameters())
      loss = self.relu(d1_logits.mean() - d2_logits.mean()) ** 2
      if loss == 0:
          return [None, None]
      d1_grads = torch_grad(outputs=loss, inputs=d1_params, retain_graph=True, create_graph=True)
      d1_z_grads = torch_grad(outputs=loss, inputs=d1_z_params, retain_graph=True, create_graph=True)
      d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
      d1_z_norm = [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_z_grads]
      reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
      reg_d1 += [((_d1_norm**2).cuda()) for _d1_norm in d1_z_norm]
      #reg_d1 = [((d1**2).cuda() * (_d1_norm**2).cuda()) for _d1_norm in d1_norm]
      #reg_d2 = [(((1.0-d2)**2).cuda() * (_d2_norm**2).cuda()) for _d2_norm in d2_norm]
      reg_d1 = sum(reg_d1)
      #reg_d2 = sum(reg_d1)
      #self.d_loss = self.gamma * reg_d1.mean()
      if self.config.loss:
        if "g" in self.config.loss:
            self.g_loss = self.gamma * reg_d1.mean()
        if "d" in self.config.loss:
            self.d_loss = self.gamma * reg_d1.mean()
      else:
        self.d_loss = self.gamma * reg_d1.mean()
      #self.gan.add_metric('stable_js', self.ops.squash(self.d_loss))

      return [self.d_loss, self.g_loss]
