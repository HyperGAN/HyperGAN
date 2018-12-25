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

class JROptimizer(optimizer.Optimizer):
  """https://arxiv.org/pdf/1806.09235.pdf"""
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="SOSOptimizer", optimizer=None, alpha=1):
    super().__init__(use_locking, name)
    self._alpha = alpha
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate

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
    all_vars = [ v for _,v in grads_and_vars]
    all_grads = [ g for g, _ in grads_and_vars ]
    d_grads = all_grads[:len(d_vars)]
    g_grads = all_grads[len(d_vars):]
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
        [self._zeros_slot(v, "orig", self._name) for _,v in grads_and_vars]
        slots_list = []
        if self.config.include_slots:
            for name in self.optimizer.get_slot_names():
                for var in self.optimizer.variables():
                    slots_list.append(self.optimizer._zeros_slot(var, "orig", "orig"))

    v1 = [self.get_slot(v, "orig") for _,v in grads_and_vars]
    slots_list = []
    slots_vars = []

    restored_vars = all_vars + slots_vars
    tmp_vars = v1 + slots_list

    e1 = 0.0001
    e2 = 0.0001

    #gamma12
    save = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars, restored_vars)]) # store variables
    restore = tf.group(*[tf.assign(w, v) for w,v in zip(restored_vars, tmp_vars)]) # store variables

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
    with tf.get_default_graph().control_dependencies([save]):
        #opboth = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step, name=name)
        #opdp = self.optimizer.apply_gradients(grads_and_vars[:len(d_vars)], global_step=global_step, name=name)
        #opgp = self.optimizer.apply_gradients(grads_and_vars[len(d_vars):], global_step=global_step, name=name)
        opboth = tf.group(*[tf.assign_sub(w, self._lr_t * v) for w,v in zip(all_vars, all_grads)]) # store variables
        opd = tf.group(*[tf.assign_sub(w, self._lr_t * v) for w,v in zip(d_vars, d_grads)]) # store variables
        opg = tf.group(*[tf.assign_sub(w, self._lr_t * v) for w,v in zip(g_vars, g_grads)]) # store variables
        with tf.get_default_graph().control_dependencies([opboth]):
            gboth = curl()#tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
            with tf.get_default_graph().control_dependencies([restore]):
                with tf.get_default_graph().control_dependencies([opd]):
                    #new_d_grads = [tf.zeros_like(_d) for _d in d_vars]+tf.gradients(self.gan.trainer.g_loss, g_vars)
                    new_d_grads = curl()#tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
                    with tf.get_default_graph().control_dependencies([restore]):
                        with tf.get_default_graph().control_dependencies([opg]):
                            #new_g_grads = tf.gradients(self.gan.trainer.d_loss, d_vars) + [tf.zeros_like(_g) for _g in g_vars]
                            new_g_grads = curl()#tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
                            with tf.get_default_graph().control_dependencies([restore]):
                                new_grads = []
                                for _gboth, _gd, _gg, _g in zip(gboth,new_d_grads,new_g_grads,d_grads):
                                    det = tf.square(_gboth)-(_gg*_gd)+1e-8
                                    h_1 = 1.0/det * (2*_gboth - _gd - _gg)
                                    if self.config.hessian:
                                        #v = (g(x + hjej)-g(x)))/(2hj) + \
                                        #    (g(x + hiei)-g(x))/(2hi)
                                        a = (_gboth - _g) / self._lr_t # d2f/dx2i
                                        c = (_gboth - _g) / self._lr_t # d2f/dx2j
                                        b = (_gg - _g) / (2*self._lr_t)+(_gd-_g)/(2*self._lr_t) # d2f/dx1dx2
                                        d = b # d2f/dx2dx1
                                        det = a*d-b*c+1e-8
                                        #h_1 = 1.0/det * (b+d-a-c)
                                        h_1_a = d/det
                                        h_1_b = -b/det
                                        h_1_c = -c/det
                                        h_1_d = a/det

                                        h_1 = h_1_a*h_1_d-h_1_b*h_1_c
                                    new_grads.append( _g*h_1 )

                                for _gboth, _gd, _gg, _g in zip(gboth[len(d_vars):],new_d_grads[len(d_vars):],new_g_grads[len(d_vars):],g_grads):
                                    det = tf.square(_gboth)-(_gg*_gd)+1e-8
                                    h_1 = 1.0/det * (2*_gboth - _gd - _gg)
                                    if self.config.hessian:
                                        #v = (g(x + hjej)-g(x)))/(2hj) + \
                                        #    (g(x + hiei)-g(x))/(2hi)
                                        a = (_gboth - _g) / self._lr_t # d2f/dx2i
                                        c = (_gboth - _g) / self._lr_t # d2f/dx2j
                                        b = (_gg - _g) / (2*self._lr_t)+(_gd-_g)/(2*self._lr_t) # d2f/dx1dx2
                                        d = b # d2f/dx2dx1
                                        det = a*d-b*c+1e-8
                                        #h_1 = 1.0/det * (b+d-a-c)
                                        h_1_a = d/det
                                        h_1_b = -b/det
                                        h_1_c = -c/det
                                        h_1_d = a/det
                                        h_1 = h_1_a*h_1_d-h_1_b*h_1_c
                                    new_grads.append( _g*h_1 )

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
    if self.config.finite_differences:
        return self.finite_differences(grads_and_vars, global_step, name, d_vars, g_vars, d_grads, g_grads)
    dc_grads = sum([tf.reduce_sum(tf.square(d)) for d in d_grads])
    gc_grads = sum([tf.reduce_sum(tf.square(g)) for g in g_grads])
    gamma12 = tf.gradients(gc_grads, d_vars) + [tf.zeros_like(g) for g in g_vars]
    gamma21 = [tf.zeros_like(d) for d in d_vars] + tf.gradients(dc_grads, g_vars)

    gamma12 = [ tf.zeros_like(ddg) if _dg is None else _dg for ddg, _dg in zip(all_vars, gamma12) ]
    gamma21 = [ tf.zeros_like(ddg) if _dg is None else _dg for ddg, _dg in zip(all_vars, gamma21) ]
    __gamma12 = [ tf.reduce_sum(_gamma12) for _gamma12 in gamma12 ]
    __gamma21 = [ tf.reduce_sum(_gamma21) for _gamma21 in gamma21 ]
    #gamma12_metric = self.gan.ops.squash(sum(gamma12))
    gamma12_metric = self.gan.ops.squash(sum(__gamma12))
    self.gan.add_metric('gamma12', gamma12_metric)
    gamma21_metric = self.gan.ops.squash(sum(__gamma21))
    self.gan.add_metric('gamma21', gamma21_metric)
   
    new_grads = []
    for _gamma12, _gamma21, _grads in zip(gamma12, gamma21, all_grads):
        Eo = _grads - \
             0.5*self._alpha*_gamma21 +\
             0.5*self._alpha*_gamma12
        new_grads += [ Eo ]

    new_grads_and_vars = list(zip(new_grads, all_vars)).copy()

    return self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step, name=name)
  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
