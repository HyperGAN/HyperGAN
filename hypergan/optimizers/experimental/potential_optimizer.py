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
import inspect

class PotentialOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None, rho=1, beta=1, gamma=1):
    super().__init__(use_locking, name)
    self.gan = gan
    self.config = config
    self.name = name
    self.optimizer = self.gan.create_optimizer(optimizer)
 
  def _prepare(self):
    super()._prepare()
    self.optimizer._prepare()

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    var_list = [ v for _,v in grads_and_vars]
    grad_list = [ g for g,_ in grads_and_vars]
    d_vars = []
    g_vars = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")

    with ops.init_scope():
        [self._get_or_make_slot(v, v, "potential", self._name) for v in var_list]
        [self._get_or_make_slot(v, v, "start", self._name) for v in var_list]
        self.optimizer._create_slots([v for g,v in grads_and_vars])
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                self._zeros_slot(var, "potential", self.name)
                self._zeros_slot(var, "start", self.name)

    self._prepare()
    potential_slots = [self.get_slot(v, "potential") for v in var_list]
    start_slots = [self.get_slot(v, "start") for v in var_list]
    for name in self.optimizer.get_slot_names():
        for var in self.optimizer.variables():
            var_list += [var]
            potential_slots += [self._zeros_slot(var, "potential", self.name)]
            start_slots += [self._zeros_slot(var, "start", self.name)]

    consensus_reg = [tf.square(g) for g in grad_list[:len(d_vars)] if g is not None]
    Jgrads = tf.gradients(consensus_reg, d_vars) + [tf.zeros_like(g) for g in g_vars]

    shapes = [self.gan.ops.shape(l) for l in grad_list]
    u = [tf.reshape(l, [-1]) for l in grad_list[:len(d_vars)]]
    v = [tf.reshape(l, [-1]) if l is not None else tf.reshape(tf.zeros_like(v), [-1]) for l,v in zip(Jgrads[:len(d_vars)], d_vars)]
    
    def proj(u, v,shape):
        bounds = [-1.0,1.0]
        if self.config.ortho_bounds:
            bounds = self.config.ortho_bounds
        dot = tf.tensordot(v, u, 1) / (tf.square(u)+1e-8)
        dot = tf.maximum(bounds[0], dot)
        dot = tf.minimum(bounds[1], dot)
        dot = dot * u
        dot = tf.reshape(dot, shape)
        return dot
    projs = [proj(_u, _v, _s) for _u, _v, _s in zip(u, v, shapes)]
    if self.config.formulation == 'g-p,-p':
        h_grads = [grad-proj for grad, proj in zip(grad_list, projs)]
    elif self.config.formulation == 'g-p,g-p':
        h_grads = [grad-proj for grad, proj in zip(grad_list, projs)]
    elif self.config.formulation == 'g-p,g+p':
        h_grads = [grad-proj for grad, proj in zip(grad_list, projs)]
    elif self.config.formulation == 'p,-p':
        h_grads = [proj for grad, proj in zip(grad_list, projs)]
    elif self.config.formulation == 'g-p,p':
        h_grads = [grad-proj for grad, proj in zip(grad_list, projs)]
    else:
        h_grads = [grad+proj for grad, proj in zip(grad_list, projs)]
    step1_grads = h_grads + grad_list[len(d_vars):]

    step1 = list(zip(step1_grads, var_list))

    mlam = self.config.merge_lambda or 0.5

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(start_slots, var_list)]) # store variables
    with tf.get_default_graph().control_dependencies([op1]):
        op2 = self.optimizer.apply_gradients(step1.copy(), global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op2]):
            op3 = tf.group(*[tf.assign(w, v) for w,v in zip(potential_slots, var_list)]) # store variables
            with tf.get_default_graph().control_dependencies([op3]):
                op4 = tf.group(*[tf.assign(w, v) for w,v in zip(var_list, start_slots)])
                with tf.get_default_graph().control_dependencies([op4]):
                    if self.config.formulation == 'p,-p':
                        p_grads = [-proj for _g, proj in zip(grad_list, projs)]
                    elif self.config.formulation == 'g-p,g-p':
                        p_grads = [_g-proj for _g, proj in zip(grad_list, projs)]
                    elif self.config.formulation == 'g-p,g+p':
                        p_grads = [_g+proj for _g, proj in zip(grad_list, projs)]
                    elif self.config.formulation == 'g-p,-p':
                        p_grads = [-proj for _g, proj in zip(grad_list, projs)]
                    elif self.config.formulation == 'g-p,p':
                        p_grads = [proj for _g, proj in zip(grad_list, projs)]
                    else:
                        p_grads = [proj for _g, proj in zip(grad_list, projs)]
                    step2_grads = p_grads + grad_list[len(d_vars):]
                    step2 = list(zip(step2_grads, var_list))
                    op5 = self.optimizer.apply_gradients(step2.copy(), global_step=global_step, name=name)
                    with tf.get_default_graph().control_dependencies([op5]):
                        if self.config.ema:
                            op6 = tf.group(*[tf.assign(w, self.config.ema*start + (1.0-self.config.ema)*(mlam*h+(1-mlam)*p)) for start,w,h,p in zip(start_slots, var_list, var_list, potential_slots)])
                        else:
                            op6 = tf.group(*[tf.assign(w, mlam*h+(1-mlam)*p) for w,h,p in zip(var_list, var_list, potential_slots)])
                        with tf.get_default_graph().control_dependencies([op6]):
                            return tf.no_op()


  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer.variables()
