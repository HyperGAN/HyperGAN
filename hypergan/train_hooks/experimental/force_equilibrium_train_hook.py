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

class ForceEquilibriumTrainHook(BaseTrainHook):
  "Forces d_fake close to d_real iff too far apart"
  def __init__(self, gan=None, config=None, trainer=None, name="GpSnMemoryTrainHook", distance=0.001, lam=1.0):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)

    klip = self.gan.configurable_param(distance)
    k_lip = (lam * tf.nn.relu(tf.abs(tf.reduce_mean(self.gan.loss.d_real-self.gan.loss.d_fake))-klip))
    self.gan.add_metric("force_eq", k_lip)
    self.loss = [k_lip, None]

  def losses(self):
      return self.loss

