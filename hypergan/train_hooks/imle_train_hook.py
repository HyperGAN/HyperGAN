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
  def __init__(self, gan=None, config=None, trainer=None, name="ClosestExampleTrainHook", search_size=4):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    gan_inputs = self.gan.inputs.x
    self.s_max = [ tf.Variable( tf.zeros_like(gan_inputs)) for i in range(search_size)]
    self.x_matched = tf.Variable(tf.zeros_like(gan.generator.sample))
    self.latent_max = [ tf.Variable( tf.zeros_like(gan.latent.sample)) for i in range(search_size)]
    self.latent = tf.Variable(tf.zeros_like(gan.latent.sample))
    self.assign_x = tf.assign(self.x_matched, gan.inputs.x)
    self.d_lambda = config['lambda'] or 1

    self.search_size = search_size

    self.current = tf.Variable(tf.zeros_like(gan_inputs))
    self.assign_latent_to_max = [self.latent.assign(self.latent_max[i]) for i in range(search_size)]
    self.assign_s_max = [self.s_max[i].assign(gan.generator.sample) for i in range(search_size)]
    self.assign_s_max_to_current = [self.current.assign(self.s_max[i]) for i in range(search_size)]
    self.assign_latent = [self.latent_max[i].assign(gan.latent.sample) for i in range(search_size)]
    self.gi = self.gan.create_component(self.gan.config.generator, name='generator', input=self.latent, reuse=True)
    diff = tf.abs(self.gi.sample-self.x_matched)
    n = tf.norm(diff, ord=2)
    self.l2_loss_on_saved = self.d_lambda * tf.reduce_sum(diff/n)

    diff_on_g = tf.abs(gan.generator.sample-self.x_matched)
    n_on_g = tf.norm(diff, ord=2)
    self.l2_loss_on_g = self.d_lambda * tf.reduce_sum(diff_on_g/n_on_g)
    self.gan.add_metric('perceptual', self.l2_loss_on_saved)
    self.gan.add_metric('latent', tf.reduce_sum(self.latent))
    self.gan.add_metric('xm', tf.reduce_sum(self.x_matched))

  def losses(self):
      return [None, self.l2_loss_on_saved]

  def after_step(self, step, feed_dict):
      pass

  def before_step(self, step, feed_dict):
    if step % (self.config.step_count or 1000) != 0:
      return
    print("[IMLE] recalculating likelihood")
    scores = []
    self.gan.session.run(self.assign_x)
    for i in range(self.search_size):
        s,_,_ = self.gan.session.run([self.l2_loss_on_g, self.assign_s_max[i], self.assign_latent[i]])
        scores.append(s)
    sort = zip(scores, self.assign_s_max_to_current, self.assign_latent_to_max)
    sort2 = sorted(sort, key=itemgetter(0))
    print("  Max ", scores[0])
    print("  Min ", scores[-1])
    self.gan.session.run(sort2[-1][-1])
    self.gan.session.run(sort2[-1][-2])

