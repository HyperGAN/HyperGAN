#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypergan.gan_component import GANComponent
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import hyperchamber as hc
import inspect

class BaseTrainHook(GANComponent):
  def __init__(self, gan=None, config=None, trainer=None, name="BaseTrainHook"):
    super().__init__(gan, config, name=name)
    self.trainer = trainer
    self.name=name

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
