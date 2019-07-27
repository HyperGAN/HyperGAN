#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from hypergan.viewer import GlobalViewer
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
        
    if not isinstance(layer, list):
        layer = [layer]
    m_x = [component.layer(l+"_mx") for l in layer]
    m_g = [component.layer(l+"_mg") for l in layer]
    if 'x' in variables:
        self.zero_x = [tf.assign(m, tf.zeros_like(m)) for m in m_x]
    else:
        self.zero_x = [tf.no_op()]
    if 'g' in variables:
        self.zero_g = [tf.assign(m, tf.zeros_like(m)) for m in m_g]
    else:
        self.zero_g = [tf.no_op()]
    target = getattr(self.gan, self.config.target or "loss")
    self.target = target
    d_fake = target.d_fake
    d_real = target.d_real
    if self.config.loss == "base":
        self.loss = tf.reduce_mean(tf.square(d_fake - d_real))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == "zero":
        self.loss += tf.reduce_mean(tf.square(d_fake))*10000.0
    if self.config.loss == "wgan":
        self.loss = tf.reduce_mean(tf.square(d_fake))*10000.0
        self.loss += tf.reduce_mean(tf.square(d_real))*10000.0
    if self.config.loss == "close":
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(-d_fake-0.1)))*1000.0
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(d_fake-0.1)))*1000.0
    if self.config.loss == "close2":
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_fake)-(self.config.distance or 0.1))))*1000.0
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_real)-(self.config.distance or 0.1))))*1000.0
    if self.config.loss == 'fixed':
        self.loss = tf.reduce_mean(tf.square(tf.reduce_mean(d_fake - d_real)))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == 'fixed2':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == 'ali2':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.reduce_mean(self.gan.standard_loss.d_real) - tf.reduce_mean(self.gan.standard_loss.d_fake))))*(self.config.loss_lambda or 10000.0)
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.reduce_mean(self.gan.z_loss.d_real) - tf.reduce_mean(self.gan.z_loss.d_fake))))*(self.config.loss_lambda or 10000.0)

    if self.config.loss == 'fixed5':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))))*(self.config.loss_lambda or 10000.0)
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_fake)-(self.config.distance or 0.01))))*10000.0
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_real)-(self.config.distance or 0.01))))*10000.0
        self.gan.add_metric('d_fake_ms', tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_fake)-(self.config.distance or 0.01))))*10000.0)
        self.gan.add_metric('ms_loss', self.loss)
    if self.config.loss == 'fixed7':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_fake)-(self.config.distance or 0.01))))*10000.0
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_real)-(self.config.distance or 0.01))))*10000.0
    if self.config.loss == 'd1':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_fake)-(self.config.distance or 0.01))))*10000.0
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(d_real)-(self.config.distance or 0.01))))*10000.0
 
    if self.config.loss == "fixed3":
        self.loss = tf.reduce_mean(tf.square(d_real+0.05))*1000.0
        self.loss += tf.reduce_mean(tf.square(d_fake-0.05))*1000.0
    if self.config.loss == 'fixed4':
        self.loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))))*(self.config.loss_lambda or 10000.0)
        self.loss += tf.reduce_mean(tf.square(tf.nn.relu(-d_fake)))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == 'targets':
        self.loss = tf.reduce_mean(tf.square(-d_fake-1e3))*(self.config.loss_lambda or 10000.0)
        self.loss += tf.reduce_mean(tf.square(d_real-1e3))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == 'absdiff':
        self.loss = tf.reduce_mean(tf.square(tf.abs(d_fake)-tf.abs(d_real)))*(self.config.loss_lambda or 10000.0)
    if self.config.loss == 'xor':
        self.loss = tf.reduce_mean(tf.square(d_fake+d_real))*(self.config.loss_lambda or 10000.0)
        self.loss += tf.reduce_mean(tf.square(tf.abs(d_fake) - 0.01))*1000.0




    var_list = []
    for v in variables:
        if "x" == v:
            [var_list.append(m) for m in m_x]
            continue
        if "g" == v:
            [var_list.append(m) for m in m_g]
            continue
        var_list.append(getattr(self.gan, v).variables())
    self.initial_learn_rate = self.config.optimizer.learn_rate
    self.learn_rate = tf.Variable(self.config.optimizer.learn_rate)
    self.config.optimizer['learn_rate']=self.learn_rate
    self.optimizer = self.gan.create_optimizer(self.config.optimizer)
    self.train_t = self.optimizer.minimize(self.loss, var_list=var_list)
    self.reset_optimizer_t = tf.variables_initializer(self.optimizer.variables())

  def before_step(self, step, feed_dict, depth=0):
    max_depth = self.config.max_depth
    if max_depth is None:
        max_depth = 2
    begin = self.gan.session.run(self.loss, feed_dict)
    last_loss = begin
    if last_loss < self.config.loss_threshold:
        if self.config.verbose:
            print(self.config.component, "> Loss begin " + str(begin) + " skipping training")
        return
    learn_rate = self.initial_learn_rate / 2*(depth+1)
    feed_dict[self.learn_rate] = learn_rate
    if self.config.verbose:
        print("Learn rate: ", learn_rate)
    self.gan.session.run(self.zero_x+ self.zero_g+ [self.reset_optimizer_t])
    for i in range((self.config.max_steps or 100)*(1+depth)):
        _ = self.gan.session.run(self.train_t, feed_dict)
        loss = self.gan.session.run(self.loss, feed_dict)
        if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
            if max_depth == depth+1:
                print("NAN during X and G training.  Resetting.")
                return self.before_step(step, feed_dict, depth+1)
        convergence = 1.0-loss/last_loss
        last_loss = loss
        if self.config.verbose:
            print("Convergence:", convergence, loss)
        if (self.config.convergence_threshold is not None and convergence < self.config.convergence_threshold) and convergence > 0.0:
            if self.config.verbose:
                print("Convergence threshold reached", self.config.convergence_threshold)
            break

        if self.config.loss_threshold is not None and loss < self.config.loss_threshold:
            if self.config.verbose:
                print("Loss threshold reached", self.config.loss_threshold)
            break

    if i+1 == ((self.config.max_steps or 100)*(1+depth)) and loss > self.config.loss_threshold:
        if max_depth != depth+1:
            print("No convergence, decreasing learn rate", feed_dict[self.learn_rate], depth, loss)
            return self.before_step(step, feed_dict, depth+1)

    print(self.config.component, "> steps", i, "Loss begin " + str(begin) + " Loss end "+str(i)+":", loss, "lr", feed_dict[self.learn_rate])

  def after_step(self, step, feed_dict):
    self.gan.session.run(self.zero_x+ self.zero_g)
