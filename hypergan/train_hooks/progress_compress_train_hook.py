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

EPS=1e-8

class ProgressCompressTrainHook(BaseTrainHook):
  """https://arxiv.org/pdf/1805.06370v2.pdf"""
  def __init__(self, gan=None, config=None, trainer=None, name="ProgressCompressTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    d_loss = []

    self.x = tf.Variable(tf.zeros_like(gan.inputs.x))
    self.g = tf.Variable(tf.zeros_like(gan.generator.sample))

    stacked = tf.concat([self.x, self.g], axis=0)
    self.kb = gan.create_component(config["knowledge_base"], name="knowledge_base", input=stacked)
    self.assign_x = tf.assign(self.x, gan.inputs.x)
    self.assign_g = tf.assign(self.g, gan.generator.sample)
    self.re_init_d = [d.initializer for d in gan.discriminator.variables()]
    gan.hack = self.g

    self.assign_knowledge_base = []
    for c in gan.components:
        if hasattr(c, 'knowledge_base'):
            for name, net in c.knowledge_base:
                assign = self.kb.named_layers[name]
                if self.ops.shape(assign)[0] > self.ops.shape(net)[0]:
                    assign = tf.slice(assign,[0 for i in self.ops.shape(net)] , [self.ops.shape(net)[0]]+self.ops.shape(assign)[1:])
                self.assign_knowledge_base.append(tf.assign(net, assign))

    def kl_divergence(_p, _q):
        return tf.reduce_sum(_p * tf.log(_p/(_q+EPS)+EPS))

    if self.config.method == 'gan':
        bs = gan.batch_size()
        real = tf.reshape(gan.loss.sample[:2], [bs, 2])
        fake = tf.reshape(self.kb.sample, [bs, 2])
        gl = tf.concat([real, fake], axis=0)
        self.kb_d = gan.create_component(self.config.knowledge_discriminator, name="kb_d", input=gl)
        self.kb_loss = self.gan.create_loss(self.kb_d)
        self.loss = self.kb_loss.sample[0] + self.kb_loss.sample[1]
        variables = self.kb_d.variables()
        variables += self.kb.variables()
    else:
        self.kb_loss = self.gan.create_component(gan.config.loss, discriminator=self.kb, split=2)

        self.loss = kl_divergence(gan.loss.sample[0], self.kb_loss.sample[0])
        if self.config.kl_on_d_fake:
            self.loss = kl_divergence(gan.loss.d_real, self.kb_loss.d_real)
        if self.config.kl_on_g:
            self.loss += kl_divergence(gan.loss.sample[1], self.kb_loss.sample[1])
        variables = self.kb.variables()
    config["optimizer"]["loss"] = self.loss
    self.optimizer = self.gan.create_optimizer(config["optimizer"])
    grads = tf.gradients(self.loss, variables)
    apply_vec = list(zip(grads, variables)).copy()
    self.optimize_t = self.optimizer.apply_gradients(apply_vec, global_step=gan.global_step)

    self.gan.add_metric('kbl', self.kb_loss.sample[0])
    self.gan.add_metric('kblg', self.kb_loss.sample[1])
    self.gan.add_metric('compress', self.loss)

  def losses(self):
      return [None, None]

  def after_step(self, step, feed_dict):
    if step % (self.config.step_count or 1) != 0:
      return
    # compress
    for i in range(self.config.night_steps or 1):
        self.gan.session.run(self.optimize_t)
    if self.config.reinitialize_every:
        if step % (self.config.reinitialize_every)==0 and step > 0:
            print("Reinitializing active D")
            self.gan.session.run(self.re_init_d)

  def before_step(self, step, feed_dict):
    self.gan.session.run([self.assign_x, self.assign_g]+ self.assign_knowledge_base)

