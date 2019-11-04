import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
import collections

from tensorflow.python.ops import math_ops
from hypergan.trainers.base_trainer import BaseTrainer
from tensorflow.python.ops import gradients_impl

TINY = 1e-12

cg_state = collections.namedtuple("CGState", ["i", "x", "r", "p", "rdotr", "alpha"])
def update_vars(state1, state2):
  ops = []
  for name in state1._fields:
    state1_vs = getattr(state1, name)
    if isinstance(state1_vs, list):
      ops += [tf.assign(_v1, _v2) for _v1, _v2 in zip(state1_vs, getattr(state2, name))]
    else:
      ops += [tf.assign(state1_vs, getattr(state2, name))]
  return tf.group(*ops)

def build_vars(state):
  args = []
  variables = []
  for name in state._fields:
    vs = getattr(state, name)
    if name == 'alpha':
        sv = tf.Variable(vs, trainable=False, name=name+"_sv_dontsave")
        variables += [sv]
    elif isinstance(vs, list):
        sv = [tf.Variable(tf.zeros_like(v), trainable=False, name=name+"_sv_dontsave") for v in vs]
        variables += sv
    else:
        sv = tf.Variable(tf.zeros_like(vs), trainable=False, name=name+"_sv_dontsave")
        variables += [sv]
    args.append(sv)
  return cg_state(*args), variables

def tf_conjugate_gradient(operator,
                       rhs,
                       tol=1e-4,
                       max_iter=20,
                       alpha=1.0,
                       config={},
                       loss=None,
                       params=None,
                       name="conjugate_gradient"):
    r"""
        modified from tensorflow/contrib/solvers/python/ops/linear_equations.py
    """
    def dot(x, y):
      x = [tf.reshape(_x, [-1]) for _x in x]
      y = [tf.reshape(_y, [-1]) for _y in y]
      x = tf.concat(x, axis=0)
      y = tf.concat(y, axis=0)
      return tf.reduce_sum(tf.multiply(x,y))

    def cg_step(state):  # pylint: disable=missing-docstring
      #z = [p * h2 for p, h2 in zip(state.p, operator.apply(state.p))]
      grad = operator.apply(state.p)
      if config.z == "p+grad":
        z = [_grad + _p for _grad, _p in zip(grad, state.p)]
      else:
        z = grad

      if config.alpha == "hager_zhang":
          ValueAndGradient = collections.namedtuple('ValueAndGradient', ['x', 'f', 'df'])
          def value_and_gradients_function(x):
              return ValueAndGradient(x=params, f=loss, df=tf.gradients(loss, params))
          ls_result = tfp.optimizer.linesearch.hager_zhang(value_and_gradients_function, initial_step_size=1.0)
          alpha = ls_result.left
      else:
          #alpha = dot(z,z) / (dot(state.p, state.p)+1e-32)
          alpha = state.alpha#tf.Print(state.alpha, [state.alpha], "Alpha:")

      #x = [_alpha * _p + _x for _alpha, _p, _x in zip(alpha, state.p, state.x)]
      print("X", state.x)
      print("P", state.p)
      print("alpha", alpha)
      x = [alpha * _p + _x for  _p, _x in zip(state.p, state.x)]
      r_1 = state.r
      r = grad#z#[alpha * _z + _r for _z, _r in zip(z, state.r)]

      new_rdotr = dot(r, r)
      beta_fr = new_rdotr / (rdotr+1e-32)
      y = [_r - _p for _r, _p in zip(r, state.p)]
      beta_pk = tf.nn.relu(dot(r,  y) / (dot(state.p, state.p)+1e-32))
      beta_bycd = dot(r, r) / tf.maximum( dot( state.p, y ), dot( [-_p for _p in state.p], r_1 ) )
      p = [_r + beta_bycd * _p for _r, _p in zip(r,state.p)]
      i = state.i + 1

      return cg_state(i, x, r, p, new_rdotr, alpha)

    with tf.name_scope(name):
      x = [tf.zeros_like(h) for h in rhs]
      rdotr = dot(rhs, rhs)
      #p = [-_p for _p in rhs]
      p = rhs
      state = cg_state(i=0, x=x, r=p, p=p, rdotr=rdotr, alpha=alpha)
      state, variables = build_vars(state)
      def update_op(state):
        return update_vars(state, cg_step(state))
      def reset_op(state, rhs):
        return update_vars(state, cg_step(cg_state(i=0, x=x, r=[state.alpha * _g for _g in rhs], p=[state.alpha * _g for _g in rhs], rdotr=rdotr, alpha=state.alpha)))
      return [reset_op(state, rhs), update_op(state), variables, state]

class CGOperator:
    def __init__(self, hvp, x_loss, y_loss, x_params, y_params, lr):
        self.hvp = hvp
        self.x_loss = x_loss
        self.y_loss = y_loss
        self.x_params = x_params
        self.y_params = y_params
        self.lr = lr

    def apply(self, p):
        lr_x = self.lr#tf.sqrt(self.lr)
        lr_y = self.lr
        h_1_v = self.hvp(self.x_loss, self.y_params, self.x_params, [lr_x * _p for _p in p])
        for _x, _h in zip(self.x_params, h_1_v):
            if _h is None:
                print("X none", _x)
        h_2_v = self.hvp(self.y_loss, self.x_params, self.y_params, [lr_y * _h for _h in h_1_v])
        return h_2_v
        #return [lr_x * _g for _g in h_2_v]

class CompetitiveTrainer(BaseTrainer):
    def hessian_vector_product(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")

        if grads is None:
            grads = tf.gradients(ys, xs)

        assert len(grads) == len(xs)
        elemwise_products = [
                math_ops.multiply(grad_elem, tf.stop_gradient(v_elem))
                for grad_elem, v_elem in zip(grads, vs)
                if grad_elem is not None
                ]

        return tf.gradients(elemwise_products, xs2)

    def _create(self):
        gan = self.gan
        config = self.config
        lr = self.config.learn_rate

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        config.optimizer["loss"] = loss.sample

        self.optimizer = self.gan.create_optimizer(config.optimizer)

        d_grads = tf.gradients(d_loss, gan.d_vars())
        g_grads = tf.gradients(g_loss, gan.g_vars())

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        d_params = gan.d_vars()
        g_params = gan.g_vars()
        clarified_d_grads = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_sv_dontsave") for v in d_grads]
        clarified_g_grads = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_sv_dontsave") for v in g_grads]

        clarified_grads = clarified_d_grads + clarified_g_grads
        operator_g = CGOperator(hvp=self.hessian_vector_product, x_loss=d_loss, y_loss=g_loss, x_params=d_params, y_params=g_params, lr=lr)
        reset_g_op, cg_g_op, var_g, state_g = tf_conjugate_gradient( operator_g, clarified_g_grads, max_iter=(self.config.nsteps or 10), config=(self.config.conjugate or hc.Config({})), alpha=(self.config.initial_alpha or 1.0), loss=g_loss, params=g_params)
        operator_d = CGOperator(hvp=self.hessian_vector_product, x_loss=g_loss, y_loss=d_loss, x_params=g_params, y_params=d_params, lr=lr)
        reset_d_op, cg_d_op, var_d, state_d = tf_conjugate_gradient( operator_d, clarified_d_grads, max_iter=(self.config.nsteps or 10), config=(self.config.conjugate or hc.Config({})), alpha=(self.config.initial_alpha or 1.0), loss=d_loss, params=d_params)
        self._variables = var_g + var_d + clarified_g_grads + clarified_d_grads

        assign_d = [tf.assign(c, x) for c, x in zip(clarified_d_grads, d_grads)]
        assign_g = [tf.assign(c, y) for c, y in zip(clarified_g_grads, g_grads)]
        self.reset_clarified_gradients = tf.group(*(assign_d+assign_g))

        self.reset_conjugate_tracker = tf.group(reset_g_op, reset_d_op)
        self.conjugate_gradient_descend_t_1 = tf.group(cg_g_op, cg_d_op)

        assign_d2 = [tf.assign(c, x) for c, x in zip(clarified_d_grads, state_d.x)]
        assign_g2 = [tf.assign(c, y) for c, y in zip(clarified_g_grads, state_g.x)]

        self.conjugate_gradient_descend_t_2 = tf.group(*(assign_d2+assign_g2))
        self.gan.add_metric('cg_g', sum([ tf.reduce_sum(tf.abs(_p)) for _p in clarified_g_grads]))

        if self.config.sga_lambda:
            dyg = tf.gradients(g_loss, g_params)
            dxf = tf.gradients(d_loss, d_params)
            hyp_d = self.hessian_vector_product(d_loss, g_params, d_params, [self.config.sga_lambda * _g for _g in dyg])
            hyp_g = self.hessian_vector_product(g_loss, d_params, g_params, [self.config.sga_lambda * _g for _g in dxf])
            sga_g_op = [tf.assign_sub(_g, _h) for _g, _h in zip(clarified_g_grads, ([state_g.alpha * _g for _g in hyp_g]))]
            sga_d_op = [tf.assign_sub(_g, _h) for _g, _h in zip(clarified_d_grads, ([state_d.alpha * _g for _g in hyp_d]))]
            self.sga_step_t = tf.group(*(sga_d_op + sga_g_op))
            self.gan.add_metric('hyp_g', sum([ tf.reduce_mean(_p) for _p in hyp_g]))
            self.gan.add_metric('hyp_d', sum([ tf.reduce_mean(_p) for _p in hyp_d]))

        #self.clarification_metric_g = sum(state_g.rdotr)
        #self.clarification_metric_d = sum(state_d.rdotr)
        def _metric(r):
            #return tf.reduce_max(tf.convert_to_tensor([tf.reduce_max(tf.norm(_r)) for _r in r]))
            return [tf.reduce_max(tf.norm(_r)) for _r in r][0]
        self.clarification_metric_g = state_g.rdotr
        self.clarification_metric_d = state_d.rdotr

        all_vars = d_params + g_params
        new_grads_and_vars = list(zip(clarified_grads, all_vars)).copy()
    
        self.last_mx = 1e12
        self.last_my = 1e12

        self.alpha = state_g.alpha
        self.alpha_value= tf.constant(1.0)
        set_alpha_d = tf.assign(state_d.alpha, self.alpha_value)
        set_alpha_g = tf.assign(state_g.alpha, self.alpha_value)
        self.set_alpha = tf.group(set_alpha_d, set_alpha_g)
        self.optimize_t = self.optimizer.apply_gradients(new_grads_and_vars)

    def required(self):
        return "".split()

    def variables(self):
        return super().variables() + self._variables

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        d_loss, g_loss = loss.sample

        self.before_step(self.current_step, feed_dict)
        sess.run(self.reset_clarified_gradients, feed_dict)
        sess.run(self.reset_conjugate_tracker, feed_dict)
        i=0
        sess.run(self.set_alpha, {self.alpha_value: (self.config.initial_alpha or 1.0)})
        alpha = sess.run(self.alpha)

        if self.config.sga_lambda:
            sess.run(self.sga_step_t, feed_dict)

        while True:
            i+=1
            mx, my, _ = sess.run([self.clarification_metric_d, self.clarification_metric_g, self.conjugate_gradient_descend_t_1], feed_dict)
            if i == 1:
                initial_clarification_metric_g = my
                initial_clarification_metric_d = mx
                dy = my
                dx = mx
                my_t1 = my
                mx_t1 = mx
            else:
                dy = my_t1 - my
                dx = mx_t1 - mx
                my_t1 = my
                mx_t1 = mx

            if self.config.log_level == "info":
                alpha = sess.run(self.alpha)
                print("-MD %e MG %e alpha %e" % (mx, my, alpha))

            if mx > initial_clarification_metric_d or my > initial_clarification_metric_g or np.isnan(mx) or np.isnan(my) or np.isinf(mx) or np.isinf(my):
                sess.run(self.set_alpha, {self.alpha_value: alpha / 2.0})
                sess.run(self.reset_clarified_gradients, feed_dict)
                sess.run(self.reset_conjugate_tracker, feed_dict)
                if self.config.sga_lambda:
                    sess.run(self.sga_step_t, feed_dict)
                new_alpha = sess.run(self.alpha)
                if new_alpha < 1e-10:
                    print("Alpha too low, exploding")
                    break
                print(i, "Explosion detected, reduced alpha from ", alpha, "to", new_alpha)
                alpha = new_alpha
                i=0
                continue

            threshold_x =  mx / (initial_clarification_metric_d+1e-12)
            threshold_y = my / (initial_clarification_metric_g+1e-12)
            threshold = threshold_x + threshold_y

            if self.config.max_steps and i > self.config.max_steps:
               if self.config.verbose:
                   print("Max steps ", self.config.max_steps, "threshold", threshold, "max", self.config.threshold, "mx", mx, "my", my)
               break
            #print( "p", i, dx / mx, dy / my, self.config.threshold, dy / my > self.config.threshold)
            if i % 100 == 0 and i != 0:
                print("Threshold at", i, threshold_x, threshold_y, "last", self.last_mx, self.last_my, "m", mx, my, "d", dx, dy)
            #self.last_mx = (self.last_mx + 1e-16) * 1.05
            #self.last_my = (self.last_my + 1e-16) * 1.05
            #if ((dx / (initial_clarification_metric_d+1e-12)) < self.config.threshold and (dy / (initial_clarification_metric_g+1e-12)) < self.config.threshold and my <= self.last_my and mx <= self.last_mx):
            if mx < self.config.threshold and my < self.config.threshold:
                #self.last_mx = mx
                #self.last_my = my
                sess.run(self.conjugate_gradient_descend_t_2)
                if self.config.verbose:
                    print("Found in ", i, "threshold", threshold, "mx", mx, "my", my, "alpha", alpha)
                break
        metric_values = sess.run([self.optimize_t] + self.output_variables(metrics), feed_dict)[1:]
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

