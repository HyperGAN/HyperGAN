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

class GigaWolfOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="GigaWolfOptimizer", optimizer=None, optimizer2=None):
    super().__init__(use_locking, name)
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
    self.g_rho = gan.configurable_param(self.config.g_rho)
    self.d_rho = gan.configurable_param(self.config.d_rho)
    if tf.contrib.framework.is_tensor(self.g_rho):
        self.gan.add_metric("g_rho", self.g_rho)
    if tf.contrib.framework.is_tensor(self.d_rho):
        self.gan.add_metric("d_rho", self.d_rho)
    def create_optimizer(klass, options):
        options['gan']=self.gan
        options['config']=options
        defn = {k: v for k, v in options.items() if k in inspect.getargspec(klass).args}
        learn_rate = options.learn_rate or options.learning_rate
        if 'learning_rate' in options:
            del defn['learning_rate']
        return klass(learn_rate, **defn)

    optimizer = hc.lookup_functions(optimizer)
    self.optimizer = create_optimizer(optimizer['class'], optimizer)
    optimizer2 = hc.lookup_functions(optimizer2)
    self.optimizer2 = create_optimizer(optimizer['class'], optimizer2)
 
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
        zt = [self._get_or_make_slot(v, v, "zt", self._name) for _,v in grads_and_vars]
        xt = [self._get_or_make_slot(v, v, "xt", self._name) for _,v in grads_and_vars]
        xt = [self._get_or_make_slot(v, v, "tmp", self._name) for _,v in grads_and_vars]
        slots_list = []
        if self.config.include_slots:
            for name in self.optimizer.get_slot_names():
                for var in self.optimizer.variables():
                    self._get_or_make_slot(var, var, "zt", "zt")
                    self._get_or_make_slot(var, var, "xt", "xt")
                    self._get_or_make_slot(var, var, "tmp", "tmp")
    self._prepare()

    zt = [self.get_slot(v, "zt") for _,v in grads_and_vars]
    xt = [self.get_slot(v, "xt") for _,v in grads_and_vars]
    tmp = [self.get_slot(v, "tmp") for _,v in grads_and_vars]
    xslots_list = []
    zslots_list = []
    tmpslots_list = []
    slots_vars = []
    if self.config.include_slots:
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                slots_vars += [var]
                xslots_list.append(self._get_or_make_slot(var, var, "zt", "zt"))
                zslots_list.append(self._get_or_make_slot(var, var, "xt", "xt"))
                tmpslots_list.append(self._get_or_make_slot(var, var, "tmp", "tmp"))


    restored_vars = var_list + slots_vars
    zt_vars = zt + zslots_list
    xt_vars = xt + xslots_list
    tmp_vars = tmp + tmpslots_list
    all_grads = [ g for g, _ in grads_and_vars ]
    # store variables for resetting

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars, restored_vars)]) # store tmp_vars

    with tf.get_default_graph().control_dependencies([op1]):
        op2 = self.optimizer.apply_gradients(grads_and_vars.copy(), global_step=global_step, name=name)
        with tf.get_default_graph().control_dependencies([op2]):
            op3 = tf.group(*[tf.assign(w, v) for w,v in zip(xt_vars, restored_vars)]) # store xt^+1 in xt_vars
            with tf.get_default_graph().control_dependencies([op3]):
                op4 = tf.group(*[tf.assign(w, v) for w,v in zip(restored_vars, zt_vars)]) # restore vars to zt (different weights)
                with tf.get_default_graph().control_dependencies([op4]):
                    op5 = self.optimizer2.apply_gradients(grads_and_vars.copy(), global_step=global_step, name=name) # zt+1
                    with tf.get_default_graph().control_dependencies([op5]):
                        zt1_xt1 = [_restored_vars - _xt1_vars for _restored_vars, _xt1_vars in zip(restored_vars, xt_vars)]
                        St1 = [tf.minimum(1.0, tf.norm(_zt1_vars-_zt_vars) / tf.norm(_zt1_xt1)) for _zt1_vars, _zt_vars, _zt1_xt1 in zip(restored_vars, zt_vars, zt1_xt1)]
                        self.gan.add_metric('st1',tf.reduce_mean(St1[0]))
                        self.gan.add_metric('xzt1',tf.norm(xt_vars[0]-zt_vars[0]))
                        nextw = [_xt_t1 + _St1 * _zt1_xt1 for _xt_t1, _St1, _zt1_xt1 in zip(xt_vars, St1, zt1_xt1)]
                        op6 = tf.group(*[tf.assign(w, v) for w,v in zip(zt_vars, restored_vars)]) # new step
                        with tf.get_default_graph().control_dependencies([op6]):
                            op7 = tf.group(*[tf.assign(w, v) for w,v in zip(restored_vars, nextw)]) # store zt+1
                            with tf.get_default_graph().control_dependencies([op7]):
                                return tf.no_op()

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer2.variables()+ self.optimizer.variables()
