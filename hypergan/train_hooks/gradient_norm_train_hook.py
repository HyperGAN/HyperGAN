#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class GradientNormTrainHook(BaseTrainHook):
  """ https://github.com/rothk/Stabilizing_GANs """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.gamma = self.gan.configurable_param(self.config.gamma or 1.0)
      self.relu = torch.nn.ReLU()

  def forward(self):
      loss, norm = self.gan.regularize_gradient_norm(lambda real, fake: self.relu(real.mean() - fake.mean()) ** 2)

      if loss is None:
          return [None, None]

      if self.config.loss:
        if "g" in self.config.loss:
            self.g_loss = self.gamma * norm.mean()
        if "d" in self.config.loss:
            self.d_loss = self.gamma * norm.mean()
      else:
        self.d_loss = self.gamma * norm.mean()
      #self.gan.add_metric('stable_js', self.ops.squash(self.d_loss))

      return [self.d_loss, self.g_loss]
