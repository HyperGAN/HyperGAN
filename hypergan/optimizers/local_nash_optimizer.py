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

class LocalNashOptimizer(optimizer.Optimizer):
  """https://arxiv.org/pdf/1901.00838v1.pdf"""
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="SOSOptimizer", optimizer=None, alpha=1, loss=None):
    super().__init__(use_locking, name)
    self._alpha = alpha
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
    optimizer['loss']=loss
    self.loss = loss

    self.optimizer = self.gan.create_optimizer(optimizer)
 
  def _prepare(self):
    super()._prepare()
    self.optimizer._prepare()

  def _create_slots(self, var_list):
    super()._create_slots(var_list)
    self.optimizer._create_slots(var_list)

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)

  def finite_differences(self, grads_and_vars, global_step, name, d_vars, g_vars, d_grads, g_grads):
    """ Attempt to directly compute hessian and apply equation (6) """
    d_grads = []
    g_grads = []
    d_vars = []
    g_vars = []
    alpha = 0.5
    if self.config.alpha is not None:
        alpha = self.gan.configurable_param(self.config.alpha)
    beta = 0.5
    if self.config.beta is not None:
        beta = self.gan.configurable_param(self.config.beta)

    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
            d_grads += [grad]
        elif var in self.gan.g_vars():
            g_vars += [var]
            g_grads += [grad]
        else:
            raise("Couldn't find var in g_vars or d_vars")
    orig_grads = d_grads+g_grads
    all_vars = d_vars + g_vars

    def curl():
        grads = tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
        op3 = tf.group(*[tf.assign_sub(v, self._lr_t*grad) for grad,v in zip(grads, all_vars)])
        with tf.get_default_graph().control_dependencies([op3]):
            def curlcombine(g1,g2):
                stepsize = self._lr_t
                return g1-(g2-g1)/stepsize
            new_grads = tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
            g3s = [curlcombine(g1,g2) for g1,g2 in zip(grads,new_grads)]
            return g3s
 
    #gamma12
    if self.config.method == 'curl':
        all_grads = curl()
        d_grads = all_grads[:len(d_vars)]
        g_grads = all_grads[len(d_vars):]

    all_grads = d_grads + g_grads

    with ops.init_scope():
        [self._zeros_slot(v, "orig", self._name) for _,v in grads_and_vars]

    v1 = [self.get_slot(v, "orig") for v in all_vars]

    restored_vars = all_vars
    tmp_vars = v1

    e1 = 0.0001
    e2 = 0.0001

    #gamma12
    save = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars.copy(), restored_vars.copy())]) # store variables

    with tf.get_default_graph().control_dependencies([save]):
        #opboth = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step, name=name)
        #opdp = self.optimizer.apply_gradients(grads_and_vars[:len(d_vars)], global_step=global_step, name=name)
        #opgp = self.optimizer.apply_gradients(grads_and_vars[len(d_vars):], global_step=global_step, name=name)
        restore = tf.group(*[tf.assign(w, v) for w,v in zip(restored_vars.copy(), tmp_vars.copy())]) # store variables
        opboth = [tf.assign_sub(w, self._lr_t * v) for w,v in zip(all_vars.copy(), all_grads.copy())] # store variables
        with tf.get_default_graph().control_dependencies([tf.group(*opboth)]):
            if self.config.method == "curl":
                gboth = curl()
            else:
                gboth = tf.gradients(self.loss[0], d_vars) + tf.gradients(self.loss[1], g_vars)
            with tf.get_default_graph().control_dependencies([restore]):
                opd = opboth[:len(d_vars)]
                with tf.get_default_graph().control_dependencies([tf.group(*opd)]):
                    if self.config.method == "curl":
                        new_d_grads = curl()
                    else:
                        new_d_grads = tf.gradients(self.loss[0], d_vars) + tf.gradients(self.loss[1], g_vars)
                    with tf.get_default_graph().control_dependencies([restore]):
                        opg = opboth[len(d_vars):]
                        with tf.get_default_graph().control_dependencies([tf.group(*opg)]):
                            if self.config.method == "curl":
                                new_g_grads = curl()
                            else:
                                new_g_grads = tf.gradients(self.loss[0], d_vars) + tf.gradients(self.loss[1], g_vars)
                            with tf.get_default_graph().control_dependencies([restore]):
                                new_grads = []
                                for _gboth, _gd, _gg, _g, _orig_g in zip(gboth,new_d_grads,new_g_grads,(d_grads+g_grads), orig_grads):
                                    a = (_gg - _g) / self._lr_t # d2f/dx2i
                                    b = (_gboth - _gg) / (2*self._lr_t)+(_gd-_g)/(2*self._lr_t) # d2f/dx1dx2
                                    c = (_gboth - _gd) / (2*self._lr_t)+(_gg-_g)/(2*self._lr_t) # d2f/dx1dx2
                                    c = -c
                                    d = -(_gd - _g) / self._lr_t # d2f/dx2j
                                    if self.config.form == 5:
                                        a = (_gg - _g) / self._lr_t # d2f/dx2i
                                        b = (_gboth - _gg) / (2*self._lr_t)+(_gd-_g)/(2*self._lr_t) # d2f/dx1dx2
                                        c = (_gboth - _gd) / (2*self._lr_t)+(_gg-_g)/(2*self._lr_t) # d2f/dx1dx2
                                        d = (_gd - _g) / self._lr_t # d2f/dx2j
                                    J = np.array([[a, b], [c,d]])
                                    Jt = np.transpose(J)

                                    det = a*d-b*c+1e-8
                                    #h_1 = 1.0/det * (b+d-a-c)
                                    h_1_a = d/det
                                    h_1_b = -b/det
                                    h_1_c = -c/det
                                    h_1_d = a/det
                                    Jinv = np.array([[h_1_a,h_1_b],[h_1_c,h_1_d]])
                                    _j = Jt[0][0]*Jinv[0][0]*_g+Jt[1][0]*Jinv[1][0]*_g+Jt[0][1]*Jinv[0][1]*_g+Jt[1][1]*Jinv[1][1]*_g

                                    new_grads.append( alpha*_orig_g + beta*_j )

                                new_grads_and_vars = list(zip(new_grads, all_vars)).copy()
                                return self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step, name=name)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    all_vars = [ v for _,v in grads_and_vars]
    d_vars = []
    g_vars = []
    all_grads = [ g for g, _ in grads_and_vars ]
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
        elif var in self.gan.g_vars():
            g_vars += [var]
        else:
            raise("Couldn't find var in g_vars or d_vars")

    with ops.init_scope():
        self.optimizer._create_slots([v for g,v in grads_and_vars])
    self._prepare()

    d_grads = all_grads[:len(d_vars)]
    g_grads = all_grads[len(d_vars):]
    return self.finite_differences(grads_and_vars, global_step, name, d_vars, g_vars, d_grads, g_grads)
  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
