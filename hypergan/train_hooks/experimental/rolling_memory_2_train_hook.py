#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypergan.gan_component import ValidationException
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class RollingMemoryTrainHook(BaseTrainHook):
  "Keeps a rolling memory of the best scoring discriminator samples."
  def __init__(self, gan=None, config=None, trainer=None, name="RollingMemoryTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    self.train_hook_index = len(trainer.train_hooks)
    config = hc.Config(config)
    s = self.gan.ops.shape(self.gan.generator.sample)
    self.memories = {}
    for pairs in self.config.types:
        for _type in pairs.split("/"):
            if _type in self.memories:
                pass
            elif _type == "g":
                self.memories["g"]={"d_input": self.gan.generator.sample}
            elif _type == "x":
                self.memories["x"]={"d_input": self.gan.inputs.x}
            elif _type[0:4] == "g(mz":
                src = self.source(_type)
                with tf.variable_scope((self.config.name or self.name), reuse=self.gan.reuse) as scope:
                    mem=tf.get_variable(self.gan.ops.generate_name()+"_z_dontsave", src.shape, dtype=tf.float32,
                              initializer=tf.compat.v1.constant_initializer(0), 
                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA, 
                              trainable=False, 
                              synchronization=tf.VariableSynchronization.NONE)
                mem_gen = self.gan.create_component(self.gan.config.generator, name='generator', input=mem, reuse=True).sample
                gen = self.gan.generator.sample
                mem_sw = tf.reshape(self.sw(mem_gen, _type), [self.gan.batch_size(), 1])
                self.memories[_type]={"var": mem, "source": src, "sw": mem_sw,
                                      "d_input": mem_gen,
                                      "assign": tf.assign(mem, self.select_top(src, gen, _type) * mem_sw + (1.0 - mem_sw) * mem)
                                      }
            else:
                src = self.source(_type)
                with tf.variable_scope((self.config.name or self.name), reuse=self.gan.reuse) as scope:
                    mem=tf.get_variable(self.gan.ops.generate_name()+"_dontsave", src.shape, dtype=tf.float32,
                              initializer=tf.compat.v1.constant_initializer(0), 
                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA, 
                              trainable=False, 
                              synchronization=tf.VariableSynchronization.NONE)
                self.memories[_type]={"var": mem, "source": src, "sw": self.sw(src, _type),
                                      "d_input": mem,
                                      "assign": tf.assign(mem, self.select_top(src, src, _type) * self.sw(mem, _type) + (1.0 - self.sw(mem, _type)) * mem)
                                      }
    self.loss = [tf.zeros(1), tf.zeros(1)]
    for i, pair in enumerate(self.config.types):
        left, right = pair.split("/")
        vleft = self.memories[left]["d_input"]
        vright = self.memories[right]["d_input"]
        d = gan.create_component(gan.config.discriminator, name="discriminator", input=tf.concat([vleft, vright],axis=0), features=[gan.features], reuse=True)
        l = gan.create_component(gan.config.loss, discriminator=d)
        if self.config.add_to_losses:
            self.gan.losses += [l]
        if self.config.lambdas and i < len(self.config.lambdas):
            lam = self.config.lambdas[i]
        else:
            lam = (self.config.lam or 1.0)
        self.loss[0] += lam  * l.sample[0]
        self.loss[1] += lam * l.sample[1]
        self.gan.add_metric("rolling_d_loss_"+pair, self.loss[0])
    self.update_memory = []
    for _type, memory in self.memories.items():
      if "assign" in memory:
        self.update_memory += [memory["assign"]]
    self.update_memory = tf.group(*self.update_memory)

  def source(self, name):
    name = name.replace("+","").replace("-","")
    if name == "mg":
      return self.gan.generator.sample
    if name == "mx":
      return self.gan.inputs.x
    if name[0:4] == "g(mz":
      return self.gan.latent.sample
    raise ValidationException("Unknown rolling type: " + name)

  def select_top(self, src, var, name):
    if (self.config.top_k or 1) != 1:
      return var#unsupported
    print("VAR ", var, "SRNC", src, self.sw(var, name))
    if self.config.reverse_top:
        if "+" in name:
            name = name.replace('+','-')
        else:
            name = name.replace('-','+')

    sw = self.sw(var, name)
    sw = tf.reshape(sw, [self.ops.shape(src)[0]] + [1 for i in range(len(self.ops.shape(src))-1)])
    top = src * sw
    top = tf.reduce_sum(top, axis=[0], keep_dims=True)
    top = tf.tile(top, [self.ops.shape(src)[0]] + [1 for i in range(len(self.ops.shape(src))-1)])
    return top

  def sw(self, var, name):
    def calculate(dscore):
      swx = dscore
      swx = tf.reshape(swx, [-1])
      _, swx = tf.nn.top_k(swx, k=(self.config.top_k or 1), sorted=True, name=None)
      swx = tf.one_hot(swx, self.gan.batch_size(), dtype=tf.float32)
      swx = tf.reduce_sum(swx, reduction_indices=0)
      swx = tf.reshape(swx, [self.gan.batch_size(), 1, 1, 1])
      return swx
    def create_disc(var):
      return self.gan.create_component(self.gan.config.discriminator, name="discriminator", input=tf.concat([var, var],axis=0), features=[self.gan.features], reuse=True)

    d = create_disc(var)
    l = self.gan.create_component(self.gan.config.loss, discriminator=d)

    if name == "mg-":
        return calculate(-l.d_fake)
    if name == "mg+":
        return calculate(l.d_fake)
    if name == "mx-":
        return calculate(-l.d_real)
    if name == "mx+":
        return calculate(l.d_real)

    if name == "g(mz-)":
        return calculate(-l.d_fake)
    if name == "g(mz+)":
        return calculate(l.d_fake)


    raise ValidationException("Unknown rolling type: " + name)

  def distributed_step(self, input_iterator_next):
    def assign_m(name):
        op=self.gan.replica.trainer.train_hooks[self.train_hook_index].memories[name]["assign"]
        with tf.control_dependencies([op]):
            return tf.no_op()
        #return mg.assign(gen * swg + (1.0 - swg) * mg)
    #mxop = self.gan.distribution_strategy.extended.update(self.mx, assign_mx, args=(self.swx,))
    #mgop = self.gan.distribution_strategy.extended.update(self.mg, assign_mg, args=(self.gan.generator.sample, self.swg,))
    ops = []
    for name, memory in self.memories.items():
        if name[0] == "m":
            ops.append(self.gan.distribution_strategy.extended.call_for_each_replica(assign_m, args=(name,)))
    return ops

  def distributed_debug(self):
    ops = []
    for name, memory in self.memories.items():
        ops.append(self.gan.distribution_strategy.extended.read_var(memory["var"]))
    return ops


  def distributed_initial_step(self, input_iterator_next):
    def assign_mx(mx, inp):
        return mx.assign(inp)

    def assign_mg(mg, gen):
        return mg.assign(gen)

    ops = []
    for name, memory in self.memories.items():
        op = None
        if "mx" in name:
            op = self.gan.distribution_strategy.extended.call_for_each_replica(assign_mx, args=(memory["var"],input_iterator_next,))
        elif "mg" in name:
            op = self.gan.distribution_strategy.extended.call_for_each_replica(assign_mg, args=(memory["var"],self.gan.generator.sample,))
        if op is not None:
            ops += [op]

    #mxop = self.gan.distribution_strategy.extended.update(self.mx, assign_mx)
    #mgop = self.gan.distribution_strategy.extended.update(self.mg, assign_mg, args=(self.gan.generator.sample,))
    return ops


  #def update_op(self):
  #    return tf.group(self.assign_mg, self.assign_mx)

  def before_step(self, step, feed_dict):
      if step == 0:
          for _type, mem in self.memories.items():
              if "var" in mem and "source" in mem:
                  self.gan.session.run(tf.assign(mem["var"], mem["source"]))
          #self.gan.session.run(tf.assign(self.mx, self.gan.inputs.x))
          #self.gan.session.run(tf.assign(self.mg, self.gan.generator.sample))

  def after_step(self, step, feed_dict):
      if (step % (self.config.steps_between_memory_update or 1)) == (self.config.offset_memory_update or 0):
          self.gan.session.run(self.update_memory)

  def variables(self):
    var = []
    for _type, memory in self.memories.items():
      if "var" in memory and "assign" in memory:
        var += [memory["var"]]
    return var

  def d_inputs(self):
    var = []
    for pairs in self.config.types:
      for _type in pairs.split("/"):
        if _type in self.memories:
          var += [self.memories[_type]["d_input"]]
    return var


  def losses(self):
    return self.loss
