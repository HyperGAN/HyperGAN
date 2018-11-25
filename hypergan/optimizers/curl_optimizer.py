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
        if self.config.include_slots:
            for name in self.optimizer.get_slot_names():
                for var in self.optimizer.variables():
                    slots_list.append(self.optimizer._zeros_slot(var, "curl", "curl"))
    self._prepare()

    gswap = [self.get_slot(v, "gswap") for _,v in grads_and_vars]
    v1 = [self.get_slot(v, "v1") for _,v in grads_and_vars]
    slots_list = []
    slots_vars = []
    if self.config.include_slots:
        for name in self.optimizer.get_slot_names():
            for var in self.optimizer.variables():
                slots_vars += [var]
                slots_list.append(self.optimizer._zeros_slot(var, "curl", "curl"))


    restored_vars = var_list + slots_vars
    tmp_vars = v1 + slots_list
    tmp_grads = gswap
    all_grads = [ g for g, _ in grads_and_vars ]
    # store variables for resetting

    d_grads = all_grads[:len(d_vars)]
    if self.config.beta_type == 'sga':
        Jgrads = tf.gradients(d_grads, d_vars, grad_ys=d_grads, stop_gradients=d_vars) + [tf.zeros_like(g) for g in g_vars]
    elif self.config.beta_type == 'magnitude':
        consensus_reg = [tf.square(g) for g in d_grads if g is not None]
        Jgrads = tf.gradients(consensus_reg, d_vars) + [tf.zeros_like(g) for g in g_vars]
    else:
        consensus_reg = 0.5 * sum(
                tf.reduce_sum(tf.square(g)) for g in d_grads if g is not None
        )
        Jgrads = tf.gradients(consensus_reg, d_vars, stop_gradients=d_vars) + [tf.zeros_like(g) for g in g_vars]

    op1 = tf.group(*[tf.assign(w, v) for w,v in zip(tmp_vars, restored_vars)]) # store variables
    op2 = tf.group(*[tf.assign(w, v) for w,v in zip(gswap, all_grads)]) # store gradients

    with tf.get_default_graph().control_dependencies([op1, op2]):
        # store g2
        op3 = self.optimizer.apply_gradients(list(grads_and_vars).copy(), global_step=global_step, name=name)
        #op3 = tf.group(*[tf.assign_sub(v, self._lr_t*grad) for grad,v in grads_and_vars])
        with tf.get_default_graph().control_dependencies([op3]):

            def curlcombine(g1,g2,_v1,_v2):
                J = (g2-g1)/((_v2-_v1)+1e-8)
                if self.config.curl == 'reverse':
                    return self._gamma*g1-self._rho*(g1-g2)/((_v1-_v2)+1e-8)*g1
                elif self.config.curl == "certain":
                    p1 = tf.nn.relu(tf.sign(g1))
                    p2 = tf.nn.relu(tf.sign(g2))
                    isone = p1 * p2
                    iszero = (1-p1)*(1-p2)
                    move = iszero * (g2-g1)
                    move += isone * (g2-g1)
                    #move = tf.sign(g2-g1) * tf.square(g2 - g1)
                    m=move/((_v2-_v1) +1e-8)
                    v = self._gamma*g1-self._rho*m*g1
                    return tf.nn.softmax(v)*g1
                elif self.config.curl == "softmax":
                    return self._gamma*g1-tf.nn.softmax(J)*g1*self._rho
                elif self.config.curl == "softmax-abs":
                    return self._gamma*g1-(1.0-tf.nn.softmax(J))*g1*self._rho
                elif self.config.curl == "mirror":
                    return self._gamma*(g1 + 2*g2)
                else:
                    return self._gamma*g1-self._rho*tf.abs((g2-g1)/((_v2-_v1)+1e-8))*g1
            g2s = tf.gradients(self.gan.trainer.d_loss, d_vars) + tf.gradients(self.gan.trainer.g_loss, g_vars)
            g3s = [curlcombine(g1,g2,v1,v2) for g1,g2,v1,v2 in zip(gswap,g2s,v1,var_list)]
            op4 = tf.group(*[tf.assign(w, v) for w,v in zip(gswap, g3s)])
            with tf.get_default_graph().control_dependencies([op4]):
                # restore v1, slots
                op5 = tf.group(*[ tf.assign(w,v) for w,v in zip(restored_vars, tmp_vars)])
                with tf.get_default_graph().control_dependencies([op5]):
                    flin = []
                    for grad, jg in zip(gswap, Jgrads):
                        if jg is None:
                            print("JG NONE", grad)
                            flin += [grad]
                        else:
                            flin += [grad + jg * self._beta]

                    step3 = list(zip(flin, var_list))
                    op6 = self.optimizer.apply_gradients(step3.copy(), global_step=global_step, name=name)
                    with tf.get_default_graph().control_dependencies([op6]):
                        return tf.no_op()

                    # Flin = gamma * IF - rho * JF + beta * JtF
                    #op7 = tf.group(*[tf.assign_add(gsw, (jg * self._beta)) if jg is not None else tf.no_op() for gsw, jg in zip(gswap, Jgrads)])
                    #with tf.get_default_graph().control_dependencies([op7]):
                    #    flin_grads_and_vars = zip(gswap, var_list)
                    #    # step 1
                    #    op8 = self.optimizer.apply_gradients(list(flin_grads_and_vars).copy(), global_step=global_step, name=name)
                    #    with tf.get_default_graph().control_dependencies([op8]):
                    #        return tf.no_op()
  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")
  def variables(self):
      return super().variables() + self.optimizer.variables()
