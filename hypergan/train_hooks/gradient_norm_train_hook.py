import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class GradientNormTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.gamma = self.gan.configurable_param(self.config.gamma or 1.0)
      self.relu = torch.nn.ReLU()

  def forward(self):
      loss, norm = self.gan.regularize_gradient_norm()

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
