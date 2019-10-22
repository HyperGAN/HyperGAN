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
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

class CompetitiveOptimizer(optimizer.Optimizer):
  """https://github.com/devzhk/Implicit-Competitive-Regularization/blob/master/optimizers.py ACGD"""
  def __init__(self, learning_rate=0.001, decay=0.9, gan=None, config=None, use_locking=False, name="CompetitiveOptimizer", optimizer=None):
    super().__init__(use_locking, name)
    self._decay = decay
    self.gan = gan
    self.config = config
    self.name = name
    self.learning_rate = learning_rate
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
    d_vars = []
    d_grads = []
    g_vars = []
    g_grads = []
    for grad,var in grads_and_vars:
        if var in self.gan.d_vars():
            d_vars += [var]
            d_grads += [grad]
        elif var in self.gan.g_vars():
            g_vars += [var]
            g_grads += [grad]
        else:
            raise("Couldn't find var in g_vars or d_vars")
    min_params = d_vars
    max_params = g_vars
    grad_x = g_grads
    grad_y = d_grads
    lr = self.learning_rate

    # best
    grad_x_rev = tf.gradients(self.gan.loss.sample[0], max_params)#, grad_ys=self.gan.loss.sample[0], stop_gradients=max_params)
    grad_y_rev = tf.gradients(self.gan.loss.sample[1], min_params)#, grad_ys=self.gan.loss.sample[1], stop_gradients=min_params)

    f = self.gan.loss.sample[0]
    g = self.gan.loss.sample[1]
    
    #hyp_x = self.hvpvec(grad_y, max_params, grad_x_rev)
    #hyp_y = self.hvpvec(grad_x, min_params, grad_y_rev)

    # best
    #hyp_x = self.hvpvec(self.gan.loss.sample[0], max_params, grad_x_rev)
    #hyp_y = self.hvpvec(self.gan.loss.sample[1], min_params, grad_y_rev)

    # pytorch
    #hyp_x = self.hvpvec(grad_y, max_params, [_g*lr for _g in grad_y])
    #hyp_y = self.hvpvec(grad_x, min_params, [_g*lr for _g in grad_x])

    #hyp_x = self.hvpvec(self.gan.loss.sample[0], max_params, grad_x_rev)
    #hyp_y = self.hvpvec(self.gan.loss.sample[1], min_params, grad_y_rev)

    #hyp_x = self.hvpvec(grad_x, max_params, grad_x_rev)
    #hyp_y = self.hvpvec(grad_y, min_params, grad_y_rev)
    hyp_x = self.hvpvec([_g*lr for _g in grad_x], max_params, grad_x_rev)
    hyp_y = self.hvpvec([_g*lr for _g in grad_y], min_params, grad_y_rev)
    # explicit
    #dy_g = tf.gradients(g, min_params)#, grad_ys=self.gan.loss.sample[0], stop_gradients=max_params)
    #dx_f = tf.gradients(f, max_params)#, grad_ys=self.gan.loss.sample[1], stop_gradients=min_params)
    #dy_g = [_g * lr for _g in dy_g]
    #dx_f = [_g * lr for _g in dx_f]
    #hyp_x = self.d2xy_dy(f=f, dy=dy_g, y=min_params, x=max_params)
    #hyp_y = self.d2xy_dy(f=g, dy=dx_f, y=max_params, x=min_params)

    #hyp_x = self.hvpvec(grad_x, max_params, grad_x_rev)
    #hyp_y = self.hvpvec(grad_y, min_params, grad_y_rev)

    #hyp_x = self.hvpvec(self.gan.loss.sample[0], max_params, scaled_grad_x)
    #hyp_y = self.hvpvec(self.gan.loss.sample[1], min_params, scaled_grad_y)

    #hyp_x = tf.gradients(grad_y, max_params)
    #hyp_x = [_h * _g for _h, _g in zip(hyp_x, scaled_grad_x)]
    #hyp_y = tf.gradients(grad_x, min_params)
    #hyp_y = [_h * _g for _h, _g in zip(hyp_y, scaled_grad_y)]
    self.gan.add_metric('hyp_x', sum([ tf.reduce_mean(_p) for _p in hyp_x]))
    self.gan.add_metric('hyp_y', sum([ tf.reduce_mean(_p) for _p in hyp_y]))
    rhs_x = [g + lr*hyp for g, hyp in zip(grad_x, hyp_x)]
    rhs_y = [g - lr*hyp for g, hyp in zip(grad_y, hyp_y)]
    #old_y = [tf.zeros_like(_d * tf.math.sqrt(self.learning_rate)) for _d in d_grads]
    #
    #p_y2 = [_c * tf.math.sqrt(self.learning_rate) for _c in p_y]


    #------------
    if self.config.conjugate:
        cg_y = self.mgeneral_conjugate_gradient(grad_x=grad_y, grad_y=grad_y_rev,
                x_params=min_params, y_params=max_params, b=rhs_y, x=None, nsteps=(self.config.nsteps or 3),
                lr_x=self.learning_rate, lr_y=self.learning_rate)

        #cg_y = [c * lr for c in cg_y]
        #cg_x = [c * lr for c in cg_x]
        #rhs_x = cg_x
        #hcg = self.hvpvec(grad_y, max_params, cg_y)
        #hcg = [lr * _p + _g for _p,_g in zip(hcg, rhs_x)]

        rhs_y = cg_y

        #rhs_x = [_cg_x * _rhs_x for _cg_x, _rhs_x in zip(cg_x, rhs_x)]
        #rhs_y = [_cg_y * _rhs_y for _cg_y, _rhs_y in zip(cg_y, rhs_y)]
    #-----
    if self.config.conjugate2:
        cg_x = self.mgeneral_conjugate_gradient(grad_x=grad_x, grad_y=grad_x_rev,
                x_params=max_params, y_params=min_params, b=rhs_x, x=None, nsteps=(self.config.nsteps or 3),
                lr_x=self.learning_rate, lr_y=self.learning_rate)
        rhs_x = cg_x
    if self.config.sum_ak:
        ak_x = self.sum_ak(x=x_params, y=y_params, f=f, g=g)
        ak_y = self.sum_ak(x=y_params, y=x_params, f=g, g=f)
        new_grad_y = [_ak * _rhs for _ak, _rhs in zip(ak_y, rhs_y)]
        new_grad_x = [_ak * _rhs for _ak, _rhs in zip(ak_x, rhs_x)]

        hcg_y = self.hvpvec(grad_x, min_params, cg_x)
        new_grad_y = [g+hcg for g, hcg in zip(grad_y, hcg_y)]

    

    new_grad_x = rhs_x
    new_grad_y = rhs_y

    new_grads = new_grad_y + new_grad_x

    all_vars = d_vars + g_vars
    new_grads_and_vars = list(zip(new_grads, all_vars)).copy()
    return self.optimizer.apply_gradients(new_grads_and_vars)

  def sum_ak(self, x, y, f, g, nsteps=1):
    for i in range(nsteps):
        grads = tf.gradients(g, x)
        grads2 = tf.gradients(grads, y)
        g = self.d2xy_dy(f=f, dy=grads2, y=y, x=x)
    return g


  def d2xy_dy(self, f, dy, y, x):
    grads = tf.gradients(f, y)
  
    elemwise_products = [
        math_ops.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, dy)
        if grad_elem is not None
    ]
  
    result = tf.gradients(elemwise_products, x)
    return result

  def mgeneral_conjugate_gradient(self, grad_x, grad_y, x_params, y_params, b, lr_x, lr_y, x=None, nsteps=10):
    if x is None:
        x = [tf.zeros_like(_b) for _b in b]
    eps = 1e-8
    r = [tf.identity(_b) for _b in b]
    p = [tf.identity(_r) for _r in r]
    rdotr = [self.dot(_r, _r) for _r in r]
    lr_x_mul_p = [lr_x * _p for _p in p]
    for i in range(nsteps):
        #self.gan.add_metric("hp", sum([ tf.reduce_mean(_p) for _p in lr_x_mul_p]))
        h_1_v = self.hvpvec([_g*lr_x for _g in grad_x], y_params, lr_x_mul_p)
        h_1 = [lr_y * v for v in h_1_v]
        #self.gan.add_metric("h_1", sum([ tf.reduce_mean(_p) for _p in h_1]))
        h_2_v = self.hvpvec([_g*lr_y for _g in grad_y], x_params, h_1)
        h_2 = [lr_x * v for v in h_2_v]
        #self.gan.add_metric("h_2", sum([ tf.reduce_mean(_p) for _p in h_2]))
        Avp_ = [_p + _h_2 for _p, _h_2 in zip(p, h_2)]
        alpha = [_rdotr / (self.dot(_p, _avp_)+eps) for _rdotr, _p, _avp_ in zip(rdotr, p, Avp_)]
        x = [_alpha * _p for _alpha, _p in zip(alpha, p)]

        r = [_alpha * _avp_ for _alpha, _avp_ in zip(alpha, Avp_)]
        new_rdotr = [self.dot(_r, _r) for _r in r]
        beta = [_new_rdotr / (_rdotr+eps) for _new_rdotr, _rdotr in zip(new_rdotr, rdotr)]
        p = [_r + _beta * _p for _r, _beta, _p in zip(r,beta,p)]
        rdotr = new_rdotr
    return x

  def dot(self, r, p):
    return r * p
    s = r.shape
    r = tf.reshape(r, [-1])
    p = tf.reshape(p, [-1])
    result = tf.tensordot(r, p, axes=0)
    return tf.reshape(result, s)

  def hvpvec(self, ys, xs, vs):
    #print("VS", len(vs), "XS", len(xs), "YS", len(ys))
    #return _hessian_vector_product(ys, xs, vs)
    #result = tf.gradients(ys, xs)
    #for r,v in zip(result, vs):
    #    print("R", r)
    #    print("V", v)
    result = self.fwd_gradients(ys, xs, stop_gradients=xs)
    #print("R", len(result))
    #print("V", len(vs))
    result = [ r * v for r, v in zip(result, vs) ]
    return result

  def fwd_gradients(self, ys, xs, grad_xs=None, stop_gradients=None, colocate_gradients_with_ops=True, us=None):
    if us is None:
        us = [tf.zeros_like(y) + float('nan') for y in ys]
    print("YS", len(ys), "US", len(us))
    dydxs = tf.gradients(ys, xs, grad_ys=us,stop_gradients=stop_gradients,colocate_gradients_with_ops=colocate_gradients_with_ops)
    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs, colocate_gradients_with_ops=colocate_gradients_with_ops)
    return dysdx

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
