import torch
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class NegativeMomentumTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None):
      super().__init__(config=config, gan=gan)
      self.d_grads = None
      self.g_grads = None

  def gradients(self, d_grads, g_grads):
      if self.d_grads is None:
          self.d_grads = [torch.zeros_like(_g) for _g in d_grads]
          self.g_grads = [torch.zeros_like(_g) for _g in g_grads]
      
      new_d_grads = [g.clone() for g in d_grads]
      new_g_grads = [g.clone() for g in g_grads]
      d_grads = [_g - self.config.gamma * _g2 for _g, _g2 in zip(d_grads, self.d_grads)]
      g_grads = [_g - self.config.gamma * _g2 for _g, _g2 in zip(g_grads, self.g_grads)]
      self.d_grads = new_d_grads
      self.g_grads = new_g_grads

      return [d_grads, g_grads]
