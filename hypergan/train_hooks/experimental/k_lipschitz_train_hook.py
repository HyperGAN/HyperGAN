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

class KLipschitzTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="KLipschitzTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    if config.ragan:
        lipschitz_penalty = tf.maximum(tf.square(d_real-d_fake) - 1, 0) + tf.maximum(tf.square(d_fake-d_real) - 1, 0)
        self.add_metric('k_lipschitz_ragan', lipschitz_penalty)
    else:
        lipschitz_penalty = tf.maximum(tf.square(d_real) - 1, 0) + tf.maximum(tf.square(d_fake) - 1, 0)
        self.add_metric('k_lipschitz', ops.squash(lipschitz_penalty))
    self.d_loss = lipschitz_penalty

  def losses(self):
    return [self.d_loss, None]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
