#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class ZeroPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="GradientPenaltyTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    gan_inputs = self.gan.inputs.x

    self._lambda = self.gan.configurable_param(config['lambda'] or 1)

    if self.config.component:
        v = getattr(gan, self.config.component)
    else:
        v = gan.discriminator
    if self.config.layer is not None:
        layer = v.layer(self.config.layer)
    else:
        layer = v.sample
    self.loss = self._lambda * tf.reduce_sum(tf.square(layer))
    gan.add_metric('zero', self.loss)

  def losses(self):
    return [self.loss, self.loss]
