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

class CurlOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.001, p=0.1, gan=None, config=None, use_locking=False, name="CurlOptimizer", optimizer=None, rho=1, beta=1, gamma=1):
    super().__init__(use_locking, name)
    self._beta = beta
    self._rho = rho
    self._gamma = gamma
    self.gan = gan
    self.config = config
    self._lr_t = learning_rate
    def create_optimizer(klass, options):
        options['gan']=self.gan
        options['config']=options
        defn = {k: v for k, v in options.items() if k in inspect.getargspec(klass).args}
        return klass(options.learn_rate, **defn)

    optimizer = hc.lookup_functions(optimizer)
    self.optimizer = create_optimizer(optimizer['class'], optimizer)
 
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
        gswap = [self._zeros_slot(v, "gswap", self._name) for _,v in grads_and_vars]
        v1 = [self._zeros_slot(v, "v1", self._name) for _,v in grads_and_vars]
        slots_list = []
        self.optimizer._create_slots(v1)
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                print("FIND VAR", var)
                slots_list.append(self.optimizer._zeros_slot(var, name+"_curl", self.optimizer._name))
    self._prepare()

    gswap = [self.get_slot(v, "gswap") for _,v in grads_and_vars]
    v1 = [self.get_slot(v, "v1") for _,v in grads_and_vars]
    slots_list = []
    slots_vars = []
    for name in self.optimizer.get_slot_names():
        for var in self.optimizer.variables():
            slots_vars += [var]
            slots_list.append(self.optimizer.get_slot(var, name+"_curl"))


    restored_vars = var_list + slots_vars
    tmp_vars = v1 + slots_list
    tmp_grads = gswap
    all_grads = [ g for g, _ in grads_and_vars ]
    step1 = grads_and_vars
    # store variables for resetting
    tmp_vars = restored_vars

    consensus_reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in all_grads[:len(d_vars)] if g is not None
    )
    Jgrads = tf.gradients(consensus_reg, d_vars)+[tf.zeros_like(g) for g in g_vars]

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars, restored_vars)]) # store variables

    with tf.get_default_graph().control_dependencies(op1):
        op2 = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_grads, all_grads)]) # store gradients
        with tf.get_default_graph().control_dependencies(op2):
            # step 1
            op3 = self.optimizer.apply_gradients(step1, global_step=global_step, name=name)
            with tf.get_default_graph().control_dependencies(op3):
                # store g2

                grads2 = tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)

                def curlcombine(g1,g2,_v1,_v2):
                    return self._gamma*g1-self._rho*(g2-g1)/((_v2-_v1)+1e-8)*g1
                g1s = gswap
                g2s = grads2
                g3s = [curlcombine(g1,g2,v1,v2) for g1,g2,v1,v2 in zip(g1s,g2s,v1,var_list)]
                op4 = tf.group(*[tf.assign(w, v) for w,v in zip(gswap, g3s)])
                with tf.get_default_graph().control_dependencies(op4):
                    # restore v1, slots
                    op5 = tf.group(*[ tf.assign(w,v) for w,v in zip(restored_vars, tmp_vars)])
                    with tf.get_default_graph().control_dependencies(op5):
                        # step 3
                        flin = gswap
                        flin = []
                        for grad, jg in zip(gswap, Jgrads):
                            if jg is None:
                                print("JG NONE", grad)
                                flin += [grad]
                            else:
                                flin += [grad + jg * self._beta]
                            
                        step3 = zip(flin, var_list)
                        op6 = self.optimizer.apply_gradients(step3, global_step=global_step, name=name)
                        with tf.get_default_graph().control_dependencies(op6):
                            return tf.no_op()

  
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer.variables()
