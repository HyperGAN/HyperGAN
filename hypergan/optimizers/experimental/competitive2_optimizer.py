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

    x_loss = self.gan.loss.sample[1]
    y_loss = self.gan.loss.sample[0]

    x_grads = grad_x
    y_grads = grad_y

    if self.config.sga_lambda == 0:
        rhs_x = grad_x
        rhs_y = grad_y
    else:
        if self.config.sga_t2:
            hyp_y = self.hessian_vector_product(f, max_params, min_params, [lr * _g for _g in grad_x_rev])
            hyp_x = self.hessian_vector_product(g, min_params, max_params, [lr * _g for _g in grad_y_rev])
        else:
            hyp_x = self.hvpvec([_g*lr for _g in grad_x], max_params, grad_x_rev)
            hyp_y = self.hvpvec([_g*lr for _g in grad_y], min_params, grad_y_rev)
        self.gan.add_metric('hyp_x', sum([ tf.reduce_mean(_p) for _p in hyp_x]))
        self.gan.add_metric('hyp_y', sum([ tf.reduce_mean(_p) for _p in hyp_y]))
        if self.config.neg:
            rhs_x = [g - (self.config.sga_lambda or lr)*hyp for g, hyp in zip(grad_x, hyp_x)]
        else:
            rhs_x = [g + (self.config.sga_lambda or lr)*hyp for g, hyp in zip(grad_x, hyp_x)]
        rhs_y = [g - (self.config.sga_lambda or lr)*hyp for g, hyp in zip(grad_y, hyp_y)]

    if self.config.con5:
        cg_x = self.mgeneral_conjugate_gradient(grad_x=rhs_y, grad_y=rhs_x, x_loss=x_loss, y_loss=y_loss,
                x_params=min_params, y_params=max_params, x=rhs_x, b=rhs_x, nsteps=(self.config.nsteps or 3),
                lr=self.learning_rate)

        cg_y = self.mgeneral_conjugate_gradient(grad_x=rhs_x, grad_y=rhs_y, x_loss=y_loss, y_loss=x_loss,
                x_params=max_params, y_params=min_params, x=rhs_y, b=rhs_y, nsteps=(self.config.nsteps or 3),
                lr=self.learning_rate)
 
    elif self.config.con6:
        cg_x = self.mgeneral_conjugate_gradient(grad_x=rhs_y, grad_y=rhs_x, x_loss=x_loss, y_loss=y_loss,
                x_params=min_params, y_params=max_params, b=rhs_x, nsteps=(self.config.nsteps or 3),
                lr=self.learning_rate)

        cg_y = self.mgeneral_conjugate_gradient(grad_x=rhs_x, grad_y=rhs_y, x_loss=y_loss, y_loss=x_loss,
                x_params=max_params, y_params=min_params, b=rhs_y, nsteps=(self.config.nsteps or 3),
                lr=self.learning_rate)

    elif self.config.con4:
        cg_x, cg_y = self.sum_ak(x_grads=x_grads, y_grads=y_grads, x_loss=y_loss, y_loss=x_loss, x_params=min_params, y_params=max_params, nsteps=(self.config.nsteps or 3), lr=self.learning_rate)
    else:
        cg_x, cg_y = self.sum_ak(x_grads=x_grads, y_grads=y_grads, x_loss=x_loss, y_loss=y_loss, x_params=min_params, y_params=max_params, nsteps=(self.config.nsteps or 3), lr=self.learning_rate)

    self.gan.add_metric('cg_x', sum([ tf.reduce_mean(_p) for _p in cg_x]))
    self.gan.add_metric('cg_y', sum([ tf.reduce_mean(_p) for _p in cg_y]))

    new_grad_x = cg_x
    new_grad_y = cg_y

    new_grads = new_grad_y + new_grad_x
    print("grads", len(new_grads))

    all_vars = d_vars + g_vars
    print("allv", len(all_vars))
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
  
  def conjugate_gradient(self, grad_x, grad_y, x_params, y_params, lr_x, lr_y, x, nsteps=10):
      gy = grad_y
      for i in range(nsteps):
          gx = [_g*lr_x for _g in x]
          h_1_v = self.fwd_gradients(gx, y_params, stop_gradients=y_params)
          gy = [ _h_1_v + _g  for _h_1_v, _g in zip(h_1_v, gy)]
          h_2_v = self.fwd_gradients(gy, x_params, stop_gradients=x_params)
          x = [_h_2_v + _x for _h_2_v, _x in zip(h_2_v, x)]
      return x
  
  def conjugate_gradient2(self, grad_x, grad_y, x_params, y_params, lr_x, lr_y, x, nsteps=10):
      for i in range(nsteps):
          h_1_v = self.hvpvec([_g*lr_x for _g in grad_x], y_params, [lr_x * _p for _p in x])
          h_1 = [lr_y * v for v in h_1_v]
          h_2 = self.hvpvec([_g*lr_y for _g in grad_y], x_params, h_1)
          x = [_h_2 + _x for _h_2, _x in zip(h_2, x)]
      return x
  
  def hessian_vector_product(self, ys, xs, xs2, vs, grads=None):
    # Validate the input
      if len(vs) != len(xs):
          raise ValueError("xs and v must have the same length.")

      # First backprop
      if grads is None:
          grads = tf.gradients(ys, xs)

      assert len(grads) == len(xs)
      elemwise_products = [
              math_ops.multiply(grad_elem, tf.stop_gradient(v_elem))
              for grad_elem, v_elem in zip(grads, vs)
              if grad_elem is not None
              ]

      # Second backprop
      return tf.gradients(elemwise_products, xs2)



  def sum_ak(self, x_loss, y_loss, x_grads, y_grads, x_params, y_params, lr, nsteps=10):
    def adjust(v):
        return [_v * lr for _v in v]

    for i in range(nsteps):
        move1 = self.hessian_vector_product(x_loss, y_params, x_params, adjust(x_grads))
        y_grads = [_yg + _m1 for _yg, _m1 in zip(y_grads, move1)]
        #self.gan.add_metric('m1', sum([ tf.reduce_sum(tf.abs(_m1)) for _m1 in move1]))

        move2 = self.hessian_vector_product(y_loss, x_params, y_params, adjust(y_grads))
        x_grads = [_xg + _m2 for _xg, _m2 in zip(x_grads, move2)]
        #self.gan.add_metric('m2', sum([ tf.reduce_sum(tf.abs(_m2)) for _m2 in move2]))
    return x_grads, y_grads

  def mgeneral_conjugate_gradient(self, x_loss, y_loss, grad_x, grad_y, x_params, y_params, b, lr, x=None, nsteps=10):
    if x is None:
        x = [tf.zeros_like(_b) for _b in b]
    eps = 1e-12
    r = [tf.identity(_b) for _b in b]
    p = [tf.identity(_r) for _r in r]
    rdotr = self.dot(r, r)
    for i in range(nsteps):
        h_1_v = self.hessian_vector_product(x_loss, y_params, x_params, [lr * _p for _p in p])
        h_2_v = self.hessian_vector_product(y_loss, x_params, y_params, [lr * _h for _h in h_1_v])

        z = h_2_v

        #alpha = [self.dot(_z, _z) / (self.dot(_p, _p)+1e-32) for _rdotr, _p, _z in zip(rdotr, p, z)]
        alpha = 1.0
        x = [alpha * _p + _x for _p, _x in zip(p, x)]

        r_1 = r
        r = z
        new_rdotr = self.dot(r, r)
        #beta_pk =[ tf.nn.relu(self.dot(_r, ( _r - _p)) / (self.dot(_p, _p)+1e-32)) for _r, _p in zip(z, p) ]
        y = [_r - _p for _r, _p in zip(r, p)]
        beta_bycd = new_rdotr / (tf.maximum( self.dot( p, y ), self.dot( [-_p for _p in p], r_1 ) )+1e-32)
        p = [_r + beta_bycd * _p for _r, _p in zip(r,p)]
        self.gan.add_metric('new_rdotr', new_rdotr)
        self.gan.add_metric('rdotr', rdotr)

        rdotr = new_rdotr
    return x

  def dot(self, x, y):
    x = [tf.reshape(_x, [-1]) for _x in x]
    y = [tf.reshape(_y, [-1]) for _y in y]
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)
    return tf.reduce_sum(tf.multiply(x,y))



  def hvpvec(self, ys, xs, vs):
    #print("VS", len(vs), "XS", len(xs), "YS", len(ys))
    #return _hessian_vector_product(xs, ys, vs)
    #return tf.gradients(ys, xs, grad_ys=vs)
    #result = tf.gradients(ys, xs)
    #for r,v in zip(result, vs):
    #    print("R", r)
    #    print("V", v)
    result = self.fwd_gradients(ys, xs, stop_gradients=xs)
    #print("R", len(result))
    #print("V", len(vs))

    res = [ ]
    print("LEN", len(xs), len(result), len(ys))
    for y, r, v in zip(ys, result, vs):
        if r is not None:
            res += [v+r]
        else:
            print("[Warning] Null value in hvpvec ", v)
            res += [tf.zeros_like(y)]
    return res

  def fwd_gradients(self, ys, xs, grad_xs=None, stop_gradients=None, colocate_gradients_with_ops=True, us=None):
    if us is None:
        us = [tf.zeros_like(y) + float('nan') for y in ys]
    print("YS", len(ys), "US", len(us))
    dydxs = tf.gradients(ys, xs, grad_ys=us,stop_gradients=stop_gradients,colocate_gradients_with_ops=colocate_gradients_with_ops, unconnected_gradients='zero')
    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs, colocate_gradients_with_ops=colocate_gradients_with_ops, unconnected_gradients='zero')
    return dysdx

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def variables(self):
      return super().variables() + self.optimizer.variables()
