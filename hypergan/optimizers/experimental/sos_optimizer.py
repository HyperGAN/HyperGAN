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

class SOSOptimizer(optimizer.Optimizer):
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
        [self._get_or_make_slot(v, v, "restore", self._name) for v in all_vars]
        self.optimizer._create_slots([v for g,v in grads_and_vars])
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                self._zeros_slot(var, "restore", self._name)

    self._prepare()
    restore_slots = [self.get_slot(v, "restore") for v in all_vars]
    restore_vars = all_vars
    for name in self.optimizer.get_slot_names():
        for var in self.optimizer.variables():
            restore_vars += [var]
            restore_slots += [self._zeros_slot(var, "restore", self._name)]


    d_grads = all_grads[:len(d_vars)]
    g_grads = all_grads[len(d_vars):]
    dc_grads = [tf.reduce_sum(tf.square(d)) for d in d_grads]
    gc_grads = [tf.reduce_sum(tf.square(g)) for g in g_grads]
    l1 = self.gan.trainer.d_loss# + lookahead_l2
    l2 = self.gan.trainer.g_loss# + lookahead_l1
    #h11 = tf.gradients(d_grads, d_vars) + tf.gradients(d_grads, g_vars, stop_gradients=g_vars)
    h11 = tf.gradients(d_grads, d_vars) + [tf.zeros_like(g) for g in g_vars]
    h12 = tf.gradients(gc_grads, d_vars) + [tf.zeros_like(g) for g in g_vars]
    h21 = [tf.zeros_like(d) for d in d_vars] + tf.gradients(dc_grads, g_vars)
    #h22 = tf.gradients(g_grads, d_vars) + tf.gradients(g_grads, g_vars, stop_gradients=d_vars)
    h22 = [tf.zeros_like(d) for d in d_vars] + tf.gradients(g_grads, g_vars)
    #h22 = [tf.zeros_like(d) for d in all_vars]
    h11 = [ tf.zeros_like(_dg) if ddg is None else _dg for ddg, _dg in zip(all_vars, h11) ]
    h12 = [ tf.zeros_like(_dg) if ddg is None else _dg for ddg, _dg in zip(all_vars, h12) ]
    h21 = [ tf.zeros_like(_dg) if ddg is None else _dg for ddg, _dg in zip(all_vars, h21) ]
    h22 = [ tf.zeros_like(_dg) if ddg is None else _dg for ddg, _dg in zip(all_vars, h22) ]
    h21 = [ -_dg for _dg in h12 ]
    h22 = [ -_dg for _dg in h12 ]
    __h11 = [ tf.reduce_sum(_h11) for _h11 in h11 ]
    __h12 = [ tf.reduce_sum(_h12) for _h12 in h12 ]
    __h21 = [ tf.reduce_sum(_h21) for _h21 in h21 ]
    __h22 = [ tf.reduce_sum(_h22) for _h22 in h22 ]
    #h12_metric = self.gan.ops.squash(sum(h12))
    h11_metric = self.gan.ops.squash(sum(__h11))
    self.gan.add_metric('h11', h11_metric)
    h12_metric = self.gan.ops.squash(sum(__h12))
    self.gan.add_metric('h12', h12_metric)
    h21_metric = self.gan.ops.squash(sum(__h21))
    self.gan.add_metric('h21', h21_metric)
    h22_metric = self.gan.ops.squash(sum(__h22))
    self.gan.add_metric('h22', h22_metric)
    ho = []
    for _h12, _h21 in zip(h12, h21):
        zero = tf.zeros_like(_h12)
        shape = self.gan.ops.shape(_h12)
        #_ho = tf.stack([zero, _h11, _h22, zero])
        _ho = tf.stack([tf.stack([zero, _h12]),tf.stack([_h21, zero])])
        ho.append(_ho)

   
    eps = []
    m = tf.constant(0.0)
    for _h12, _h21, _grads, _ho in zip(h12, h21, all_grads, ho):
        # (I - alpha*Ho)
        Eo = 2.0 * _grads - \
             self._alpha*_h21 * _grads -\
             self._alpha*_h12 * _grads
        m += tf.reduce_sum(Eo)
        eps += [ Eo ]

    self.gan.add_metric('m', m)
    new_grads = eps
    new_grads_and_vars = list(zip(new_grads, all_vars)).copy()

    diag = []
    ho_t = []


    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(restore_slots, restore_vars)]) # store variables
    with tf.get_default_graph().control_dependencies([op1]):
        #lookahead = DepthOptimizer(learning_rate=0.001, decay=1.0, gan=None, config=None, use_locking=False, name="depthOptimizer", optimizer=self.optimizer, depth=1)
        #op2 = lookahead.apply_gradients(eps)
        op2 = self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op2]):
            new_grads2 = tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
            new_grads3 = []
            p1s = tf.constant(0.0)
            p2s = tf.constant(0.0)
            p1_masks = tf.constant(0.0)
            p2_masks = tf.constant(0.0)
            p1v2s = tf.constant(0.0)
            ps = tf.constant(0.0)
            for _grad1, _grad2, _ho in zip(eps, new_grads2, ho):
                axis = [i for i in range(len(self.gan.ops.shape(_ho)))]
                _X = (_ho[1][0] + _ho[0][1])*_grad2
                _Xr = tf.reshape(_X, [-1])
                _grad1r = tf.reshape(_grad1, [-1])
                angle = tf.tensordot(-self._alpha * _Xr, _grad1r, 1)
                p1_mask = (tf.sign(angle) + 1.0) / 2.0
                aeo = - self.config.a * (tf.square(tf.norm(_grad1)))/(angle+1e-8)
                p1 = tf.minimum(1.0, aeo)
                p1 = p1_mask + (1.0-p1_mask) * p1
                p2_mask = (tf.sign((tf.norm(_grad1)) - self.config.b) + 1.0) / 2.0
                p2 = tf.square(tf.norm(_grad1)) * p2_mask + (1.0-p2_mask)
                p = tf.minimum(p1, p2)
                p1v2 = tf.reduce_sum(tf.sign(p1-p2))
                ps+=tf.reduce_sum(p)
                p1s+=tf.reduce_sum(p1)
                p2s+=tf.reduce_sum(p2)
                p1v2s+=tf.reduce_sum(p1v2)
                p1_masks+=tf.reduce_sum(p1_mask)
                p2_masks+=tf.reduce_sum(p2_mask)

                if self.config.lola:
                    new_grads3 += [_grad1 - _X * self._alpha]
                else:
                    new_grads3 += [_grad1 - p * _X * self._alpha]

            self.gan.add_metric("p1m",p1_masks)
            self.gan.add_metric("p2m",p2_masks)
            self.gan.add_metric("p1v2",p1v2s)
            self.gan.add_metric("p",ps)
            self.gan.add_metric("p1",p1s)
            self.gan.add_metric("p2",p2s)
            new_grads_and_vars3 = list(zip(new_grads3, all_vars)).copy()
            op3 = tf.group(*[tf.assign(w, v) for w,v in zip(restore_vars, restore_slots)]) # store variables
            with tf.get_default_graph().control_dependencies([op3]):
                return self.optimizer.apply_gradients(new_grads_and_vars3, global_step=global_step, name=name)

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
