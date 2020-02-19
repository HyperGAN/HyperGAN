import math
import torch
from torch.optim import Optimizer

class GigaWolfOptimizer(Optimizer):
  def __init__(self, params, optimizer, optimizer2):
    defaults = dict()

    self.optimizer = optimizer
    self.optimizer2 = optimizer2
    super(GigaWolfOptimizer, self).__init__(params, defaults)

   def step(self, closure=None):
    """Performs a single optimization step across the two optimizers and uses GigaWolf.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
