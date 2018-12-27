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
    self.x_matched = [ tf.Variable(tf.zeros_like(gan.generator.sample)) for j in range(memory_size)]
    self.latent_max = [ tf.Variable( tf.zeros_like(gan.latent.sample)) for i in range(search_size)]
    self.latent = [ tf.Variable(tf.zeros_like(gan.latent.sample)) for j in range(memory_size)]
    self.assign_x = [tf.assign(self.x_matched[j], gan.inputs.x) for j in range(memory_size)]
    self.d_lambda = config['lambda'] or 1

    self.search_size = search_size
    self.memory_size = memory_size
    self.new_entries = new_entries
    if self.new_entries == -1:
        self.new_entries = memory_size

    self.assign_latent_to_max = [[self.latent[j].assign(self.latent_max[i]) for i in range(search_size)] for j in range(memory_size)]
    self.assign_latent = [self.latent_max[i].assign(gan.latent.sample) for i in range(search_size)]

    l2_losses = tf.zeros([1])
    self.gi = []
    for j in range(memory_size):
        self.gi.append(self.gan.create_component(self.gan.config.generator, name='generator', input=self.latent[j], reuse=True))
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
    #self.l2_loss_on_g = np.roll(self.l2_loss_on_g, new_entries)
    #self.assign_latent_to_max = np.roll(self.assign_latent_to_max, new_entries)
    #self.assign_x = np.roll(self.assign_x, new_entries)
    #self.l2_loss_on_g = np.roll(self.l2_loss_on_g, new_entries)
    self.x_matched = np.roll(self.x_matched, new_entries)
    self.latent = np.roll(self.latent, new_entries)
    self.gi = np.roll(self.gi, new_entries)
    print("[IMLE] recalculating likelihood")
    for j in range(new_entries):
        scores = []
        self.gan.session.run(self.assign_x[j])
        for i in range(self.search_size):
            s,_ = self.gan.session.run([self.l2_loss_on_g[j], self.assign_latent[i]])
            scores.append(s)
        sort = zip(scores, self.assign_latent_to_max[j])
        sort2 = sorted(sort, key=itemgetter(0))
        print("  Max ", sort2[-1][0])
        print("  Min ", sort2[0][0])
        self.gan.session.run(sort2[0][-1])

