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
  def __init__(self, gan=None, config=None, trainer=None, name="GradientPenaltyTrainHook", layer="match_support", variables=["x","g"]):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    component = getattr(self.gan,self.config.component or "discriminator")
    m_x = component.layer(layer+"_mx")
    m_g = component.layer(layer+"_mg")
    self.zero_x = tf.assign(m_x, tf.zeros_like(m_x))
    self.zero_g = tf.assign(m_g, tf.zeros_like(m_g))
    target = getattr(self.gan, self.config.target or "loss")
    d_fake = tf.reduce_mean(target.d_fake)
    d_real = tf.reduce_mean(target.d_real)
    self.loss = tf.square(d_fake - d_real)*(self.config.loss_lambda or 10000.0)
    #self.loss = tf.abs(d_fake - d_real)*(self.config.loss_lambda or 1.0)
    if self.config.loss == "abs":
        self.loss = tf.square(d_fake)*100 + tf.square(d_real)*(self.config.ali_lambda or 1.0)
    if self.config.loss == "xzero":
        self.loss += (self.config.l2_lambda or 1e-4) * tf.reduce_sum(tf.abs(m_x))
    if self.config.loss == "ali-manifold_guided":
        d_fake2 = tf.reduce_mean(self.gan.z_loss.d_fake)
        d_real2 = tf.reduce_mean(self.gan.z_loss.d_real)
        self.loss += tf.square(d_fake2 - d_real2)*(self.config.ali_loss_lambda or 1.0)

    var_list = []
    if "x" in variables:
        var_list.append(m_x)
    if "g" in variables:
        var_list.append(m_g)
    self.optimizer = self.gan.create_optimizer(self.config.optimizer)
    self.train_t = self.optimizer.minimize(self.loss, var_list=var_list)
    self.reset_optimizer_t = tf.variables_initializer(self.optimizer.variables())

  def before_step(self, step, feed_dict):
    self.gan.session.run([self.zero_x, self.zero_g, self.reset_optimizer_t])
    begin = self.gan.session.run(self.loss, feed_dict)
    last_loss = begin
    for i in range(self.config.max_steps or 100):
        _ = self.gan.session.run(self.train_t, feed_dict)
        loss = self.gan.session.run(self.loss, feed_dict)
        convergence = 1.0-loss/last_loss
        last_loss = loss
        print("Convergence:", convergence, loss)
        if convergence < self.config.convergence_threshold and loss < self.config.loss_threshold:
            break
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        print("NAN")
        return self.before_step(step, feed_dict)

    print("> steps", i, "Loss begin " + str(begin) + " Loss end "+str(i)+":", loss)

  def after_step(self, step, feed_dict):
    self.gan.session.run([self.zero_x, self.zero_g])
