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

class GradientLocallyStableTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="GradientLocallyStableTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    d_vars = gan.d_vars()
    g_vars = gan.g_vars()
    d_loss = gan.loss.sample[0]
    gls = tf.gradients(d_loss, d_vars+g_vars)
    gls = tf.square(tf.global_norm(gls))
    self.g_loss = self.config["lambda"] * gls
    self.add_metric('gradient_locally_stable', ops.squash(gls, tf.reduce_mean))

  def losses(self):
    return [None, self.g_loss]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
