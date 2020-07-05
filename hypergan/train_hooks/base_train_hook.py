#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypergan.gan_component import GANComponent
import hyperchamber as hc
import inspect

class BaseTrainHook(GANComponent):
  def __init__(self, gan=None, config=None, trainer=None):
    super().__init__(gan, config)

  def create(self):
    pass

  def losses(self):
    return [None, None]

  def before_step(self, step, feed_dict):
    pass

  def after_step(self, step, feed_dict):
    pass

  def after_create(self):
    pass

  def update_op(self):
    return None

  def gradients(self, d_grads, g_grads):
    return [d_grads, g_grads]

  def forward(self, d_loss, g_loss):
    return [None, None]
