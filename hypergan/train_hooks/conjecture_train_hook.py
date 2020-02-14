#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hyperchamber as hc
import numpy as np
import inspect
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class ConjectureTrainHook(BaseTrainHook):
  """ https://faculty.washington.edu/ratliffl/research/2019conjectures.pdf """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None


  def gradients(self, d_grads, g_grads):
      nsteps = self.config.nsteps
      g_loss = self.gan.trainer.g_loss
      d_loss = self.gan.trainer.d_loss
      d_params = list(self.gan.d_parameters())
      g_params = list(self.gan.g_parameters())
      lr = self.config.learn_rate or 1e-2

      if self.config.fast_conjectures:
          g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
          d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
          d_grads2 = self.hvp(g_loss, g_params, d_params, [lr * _g for _g in g_grads], g_grads)
          g_grads2 = self.hvp(d_loss, d_params, g_params, [lr * _d for _d in d_grads], d_grads)
          d_grads = [_p + _g * (self.config.fast_conjectures_gamma) for _p, _g in zip(d_grads, d_grads2)]
          g_grads = [_p + _g * (self.config.fast_conjectures_gamma) for _p, _g in zip(g_grads, g_grads2)]

      if self.config.fast_strategic_conjectures:
          f1 = d_loss
          f2 = g_loss
          p1 = d_params
          p2 = g_params
          d2f2 = torch_grad(outputs=f2.mean(), inputs=p2, create_graph=True, retain_graph=True)
          d2f1 = torch_grad(outputs=f1.mean(), inputs=p2, create_graph=True, retain_graph=True)
          d1f2 = torch_grad(outputs=f2.mean(), inputs=p1, create_graph=True, retain_graph=True)
          d1f1 = torch_grad(outputs=f1.mean(), inputs=p1, create_graph=True, retain_graph=True)

          d12f2d2f1 = self.hvp(f2, p2, p1, [lr * _g for _g in d2f1])
          d12f1d2f2 = self.hvp(f1, p2, p1, [lr * _g for _g in d2f2])

          d21f2d1f1 = self.hvp(f2, p1, p2, [lr * _g for _g in d1f1])
          d21f1d1f2 = self.hvp(f1, p1, p2, [lr * _g for _g in d1f2])
          d_grads = [_p - (_g1 + _g2) * (self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(d_grads, d12f1d2f2, d12f2d2f1)]
          g_grads = [_p - (_g1 + _g2) * (self.config.fast_strategic_conjectures_gamma) for _p, _g1, _g2 in zip(g_grads, d21f1d1f2, d21f2d1f1)]

      if self.config.jare:
          # https://github.com/weilinie/JARE/blob/master/src/ops.py#L226
          g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
          d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
          d_grad_norm = [torch.norm(_d_grads) for _d_grads in d_grads]
          g_grad_norm = [torch.norm(_g_grads) for _g_grads in g_grads]
          d_reg = 0.5 * sum([_x**2 for _x in g_grad_norm])
          g_reg = 0.5 * sum([_x**2 for _x in d_grad_norm])
          if self.config.d_jare_gamma != 0:
              d_grads2 = torch_grad(d_reg, d_params, retain_graph=True)
              d_grads = [_p + _g * (self.config.d_jare_gamma or self.config.jare_gamma) for _p, _g in zip(d_grads, d_grads2)]
          if self.config.g_jare_gamma != 0:
              g_grads2 = torch_grad(g_reg, g_params, retain_graph=True)
              g_grads = [_p + _g * (self.config.g_jare_gamma or self.config.jare_gamma) for _p, _g in zip(g_grads, g_grads2)]

 
      if self.config.consensus:
          # https://github.com/weilinie/JARE/blob/master/src/ops.py#L127
          g_grads = torch_grad(g_loss, g_params, create_graph=True, retain_graph=True)
          d_grads = torch_grad(d_loss, d_params, create_graph=True, retain_graph=True)
          d_grad_norm = [torch.norm(_d_grads) for _d_grads in d_grads]
          g_grad_norm = [torch.norm(_g_grads) for _g_grads in g_grads]
          d_reg = 0.5 * sum([_x**2 for _x in d_grad_norm])
          g_reg = 0.5 * sum([_x**2 for _x in g_grad_norm])
          reg = d_reg + g_reg
          d_grads2 = torch_grad(reg, d_params, retain_graph=True)
          g_grads2 = torch_grad(reg, g_params, retain_graph=True)
          if self.config.d_consensus_gamma != 0:
              d_grads = [_p + _g * (self.config.d_consensus_gamma or self.config.consensus_gamma) for _p, _g in zip(d_grads, d_grads2)]
          if self.config.g_consensus_gamma != 0:
              g_grads = [_p + _g * (self.config.g_consensus_gamma or self.config.consensus_gamma) for _p, _g in zip(g_grads, g_grads2)]

      if self.config.sga:
          # SGA as defined in competitive gradient descent https://github.com/devzhk/Implicit-Competitive-Regularization/blob/master/optimizers.py 
          g_grad_rev = torch_grad(d_loss, g_params, create_graph=True, retain_graph=True)
          d_grad_rev = torch_grad(g_loss, d_params, create_graph=True, retain_graph=True)
          d_sga = self.hvp(d_loss, g_params, d_params, [lr * _g for _g in g_grad_rev])
          g_sga = self.hvp(g_loss, d_params, g_params, [lr * _g for _g in d_grad_rev])
          d_grads = [_p - (self.config.sga_gamma)*_s for _p, _s in zip(d_grads, d_sga)]
          g_grads = [_p - (self.config.sga_gamma)*_s for _p, _s in zip(g_grads, g_sga)]

      return [d_grads, g_grads]

  def forward(self):
      #d_loss = self.gan.loss.sample[0]
      #g_loss = self.gan.loss.sample[1]
      #d_params = self.gan.d_vars()
      #g_params = self.gan.g_vars()
      if self.config.locally_stable:
          d_gradient_norm_sq = tf.square(tf.global_norm(tf.gradients(g_loss, d_params)))
          self.d_loss = self.config.locally_stable_gamma * d_gradient_norm_sq
          self.gan.add_metric('locally_stable', self.d_loss)

      return [self.d_loss, self.g_loss]

  def hvp(self, ys, xs, xs2, vs, grads=None):
      if grads is None:
        grads = torch_grad(outputs=ys, inputs=xs, create_graph=True, retain_graph=True)
     
      #grads = [_g.contiguous().view(-1) * (self.config.hvp_lambda or self.config.learn_rate or 1e-4) for _g in grads]
      return torch_grad(outputs=grads, inputs=xs2, grad_outputs=vs, retain_graph=True)
