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
from hypergan.gans.base_gan import BaseGAN
from hypergan.train_hooks.base_train_hook import BaseTrainHook

EPS=1e-8


class KBGAN(BaseGAN):
    def __init__(self, latent=None, x=None, *args, **kwargs):
        self.discriminator = None
        self.latent = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.session = None
        self.latent = latent
        self.x = x
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        with tf.device(self.device):
            self.session = self.ops.new_session(self.ops_config)
            self.generator = self.create_component(config.generator, name="generator", input=self.latent)
            d_input = tf.concat([self.x, self.generator.sample],axis=0)
            self.discriminator = self.create_component(config.discriminator, name="discriminator", input=d_input)
            self.loss = self.create_component(config.loss, discriminator=self.discriminator)
            self.trainer = self.create_component(config.trainer)
            self.session.run(tf.global_variables_initializer())

    def create_generator(self, latent, reuse=False):
        return self.create_component(self.config.generator, name="generator", input=latent, reuse=reuse)

    def width(self):
        return 1

    def height(self):
        return 1

    def g_vars(self):
        return self.generator.variables()

    def d_vars(self):
        return self.discriminator.variables()

    def input_nodes(self):
        "used in hypergan build"
        return [
                self.latent
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
                self.latent,
                self.random_z
        ]
class ProgressCompressKBGanTrainHook(BaseTrainHook):
  """https://arxiv.org/pdf/1805.06370v2.pdf"""
  def __init__(self, gan=None, config=None, trainer=None, name="ProgressCompressTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    d_loss = []

    self.x = tf.Variable(tf.zeros_like(gan.inputs.x))
    self.g = tf.Variable(tf.zeros_like(gan.generator.sample))

    stacked = tf.concat([self.gan.inputs.x, self.gan.generator.sample], axis=0)
    self.assign_x = tf.assign(self.x, gan.inputs.x)
    self.assign_g = tf.assign(self.g, gan.generator.sample)
    self.re_init_d = [d.initializer for d in gan.discriminator.variables()]
    gan.hack = self.g

    self.assign_knowledge_base = []

    bs = gan.batch_size()
    real = gan.discriminator.named_layers['knowledge_base_target']#tf.reshape(gan.loss.sample[:2], [2,-1])
    _inputs = hc.Config({'x':real})
    inner_gan = KBGAN(config=self.config.knowledge_base, inputs=_inputs, x=real, latent=stacked)
    self.kb_loss = inner_gan.loss
    self.kb = inner_gan.generator
    self.trainer = inner_gan.trainer
    variables = inner_gan.variables()
    #variables += self.kb.variables()

    for c in gan.components:
        if hasattr(c, 'knowledge_base'):
            for name, net in c.knowledge_base:
                assign = self.kb.named_layers[name]
                if self.ops.shape(assign)[0] > self.ops.shape(net)[0]:
                    assign = tf.slice(assign,[0 for i in self.ops.shape(net)] , [self.ops.shape(net)[0]]+self.ops.shape(assign)[1:])
                self.assign_knowledge_base.append(tf.assign(net, assign))

    self.gan.add_metric('d_kb', self.kb_loss.sample[0])
    self.gan.add_metric('g_kb', self.kb_loss.sample[1])

  def losses(self):
      return [None, None]

  def after_step(self, step, feed_dict):
    if step % (self.config.step_count or 1) != 0:
      return
    # compress
    for i in range(self.config.night_steps or 1):
        self.trainer.step(feed_dict)
    if self.config.reinitialize_every:
        if step % (self.config.reinitialize_every)==0 and step > 0:
            print("Reinitializing active D")
            self.gan.session.run(self.re_init_d)

  def before_step(self, step, feed_dict):
    self.gan.session.run([self.assign_x, self.assign_g]+ self.assign_knowledge_base)

