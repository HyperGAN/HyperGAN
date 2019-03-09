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

class IMLETrainHook(BaseTrainHook):
  """https://arxiv.org/pdf/1809.09087.pdf"""
  def __init__(self, gan=None, config=None, trainer=None, name="ClosestExampleTrainHook", search_size=4, memory_size=1, new_entries=-1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    latent = gan.latent
    if self.config.use_encoder:
        latent = gan.u_to_z
    self.x_matched = [ tf.Variable(tf.zeros_like(gan.generator.sample)) for j in range(memory_size)]
    self.latent_max = tf.Variable( tf.zeros_like(latent.sample))
    self.latent = [ tf.Variable(tf.zeros_like(latent.sample)) for j in range(memory_size)]
    self.assign_x = [tf.assign(self.x_matched[j], gan.inputs.x) for j in range(memory_size)]
    self.d_lambda = config['lambda'] or 1

    self.search_size = search_size
    self.memory_size = memory_size
    self.new_entries = new_entries
    if self.new_entries == -1:
        self.new_entries = memory_size

    if self.config.use_encoder:
        encoded = self.gan.encoder.sample
        self.assign_encoded_latent = [self.latent[j].assign(encoded) for j in range(memory_size)]
    else:
        self.assign_latent_to_min = [self.latent[j].assign(self.latent_max) for j in range(memory_size)]
    self.assign_latent = self.latent_max.assign(latent.sample)

    l2_losses = tf.zeros([1])
    self.gi = []
    for j in range(memory_size):
        self.gi.append(self.gan.create_generator(self.latent[j], reuse=True))
        diff = tf.abs(self.gi[-1].sample-self.x_matched[j])
        n = tf.norm(diff, ord=2)
        l2_losses += tf.reduce_sum(diff/n)
        
    self.l2_loss_on_saved = tf.reduce_sum(self.d_lambda * l2_losses)

    l2_losses = []
    for j in range(memory_size):
        diff = tf.abs(gan.generator.sample-self.x_matched[j])
        n = tf.norm(diff, ord=2)
        l2_losses.append(tf.reduce_sum(diff/n))

    self.l2_loss_on_g = l2_losses
    self.offset = 0

    self.gan.add_metric('perceptual', self.l2_loss_on_saved)

  def losses(self):
      return [None, self.l2_loss_on_saved]

  def after_step(self, step, feed_dict):
      pass

  def before_step(self, step, feed_dict):
    if step % (self.config.step_count or 1000) != 0:
      return

    new_entries = self.new_entries
    if step == 0:
        new_entries = self.memory_size
    print("[IMLE] recalculating likelihood")
    if self.config.use_encoder:
        for j in range(new_entries):
            _j = (j+self.offset) % self.memory_size
            self.gan.session.run([self.assign_x[_j], self.assign_encoded_latent[_j]])
    else:
        for j in range(new_entries):
            _j = (j+self.offset) % self.memory_size
            self.gan.session.run(self.assign_x[_j])
            min_score = None
            for i in range(self.search_size):
                s,_ = self.gan.session.run([self.l2_loss_on_g[_j], self.assign_latent])
                if( min_score == None or min_score > s):
                    self.gan.session.run(self.assign_latent_to_min[_j])
                    min_score = s
                else:
                    pass
                    #print("Ignoring score", s)
            print("  Min ", min_score)
    self.offset += new_entries

