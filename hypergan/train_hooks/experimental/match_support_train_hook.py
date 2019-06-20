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

class MatchSupportTrainHook(BaseTrainHook):
  """ Makes d_fake and d_real match by training on a zero-based addition to the input images. """
  def __init__(self, gan=None, config=None, trainer=None, name="GradientPenaltyTrainHook", layer="match_support"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    m_x = self.gan.discriminator.layer(layer+"_mx")
    m_g = self.gan.discriminator.layer(layer+"_mg")
    self.zero_x = tf.assign(m_x, tf.zeros_like(m_x))
    self.zero_g = tf.assign(m_g, tf.zeros_like(m_g))
    d_fake = tf.reduce_mean(self.gan.loss.d_fake)
    d_real = tf.reduce_mean(self.gan.loss.d_real)
    self.loss = tf.square(d_fake - d_real)*100
    if self.config.loss == "abs":
        self.loss = tf.square(d_fake)*100 + tf.square(d_real)*100

    learn_rate = 100.0

    self.optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    self.train_t = self.optimizer.minimize(self.loss, var_list=[m_x, m_g])

  def before_step(self, step, feed_dict):
    self.gan.session.run(self.zero_x)
    self.gan.session.run(self.zero_g)
    begin = self.gan.session.run(self.loss, feed_dict)
    for i in range(self.config.max_steps or 100):
        loss, _ = self.gan.session.run([self.loss, self.train_t], feed_dict)
        if loss < (self.config.loss_threshold or 1e-8):
            break
    if np.any(np.isnan(loss)):
        print("NAN")
        return self.before_step(step, feed_dict)

    print("> steps", i, "Loss begin " + str(begin) + " Loss end "+str(i)+":", loss)

