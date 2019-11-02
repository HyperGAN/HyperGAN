import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
import collections

from tensorflow.python.ops import math_ops
from hypergan.trainers.base_trainer import BaseTrainer
from tensorflow.python.ops import gradients_impl

TINY = 1e-12

cg_state = collections.namedtuple("CGState", ["i", "x", "r", "p", "lr", "rdotr"])
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
    if isinstance(vs, list):
        sv = [tf.Variable(tf.zeros_like(v), trainable=False, name=name+"_sv_dontsave") for v in vs]
        variables += sv
    else:
        print(vs, name)
        sv = tf.Variable(tf.zeros_like(vs), trainable=False, name=name+"_sv_dontsave")
        variables += [sv]
    args.append(sv)
  return cg_state(*args), variables

def tf_conjugate_gradient(operator,
                       rhs,
                       lr=1e-4,
                       tol=1e-4,
                       max_iter=20,
                       name="conjugate_gradient"):
    r"""
        modified from tensorflow/contrib/solvers/python/ops/linear_equations.py to work with arrays
    """
    def dot(x, y):
      return tf.reduce_sum(tf.conj(x) * y)

    def cg_step(state):  # pylint: disable=missing-docstring
      h_2_v = operator.apply(state.p)
      Avp_ = [_p + state.lr*_h_2 for _p, _h_2 in zip(state.p, h_2_v)]

      alpha = [_rdotr / (dot(_p, _avp_)+1e-8) for _rdotr, _p, _avp_ in zip(state.rdotr, state.p, Avp_)]
      x = [_alpha * _p + _x for _alpha, _p, _x in zip(alpha, state.p, state.x)]

      r = [-_alpha * _avp_+_r for _alpha, _avp_,_r in zip(alpha, Avp_, state.r)]
      new_rdotr = [dot(_r, _r) for _r in r]
      beta = [_new_rdotr / (_rdotr+1e-8) for _new_rdotr, _rdotr in zip(new_rdotr, state.rdotr)]
      p = [_r + _beta * _p for _r, _beta, _p in zip(r,beta,state.p)]
      i = state.i + 1

      return cg_state(i, x, r, p, lr, new_rdotr)

    with tf.name_scope(name):
      x = [tf.zeros_like(h) for h in rhs]
      rdotr = [dot(_r, _r) for _r in rhs]
      state = cg_state(i=0, x=x, r=rhs, p=rhs, lr=lr, rdotr=rdotr)
      state, variables = build_vars(state)
      def update_op(state):
        return update_vars(state, cg_step(state))
      def reset_op(state, rhs):
        return update_vars(state, cg_step(cg_state(i=0, x=x, r=rhs, p=rhs, lr=lr, rdotr=rdotr)))
      return [reset_op(state, rhs), update_op(state), variables, state]

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

    """ Steps G and D simultaneously """
    def _create(self):
        gan = self.gan
        config = self.config

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))
        config.optimizer["loss"] = loss.sample

        self.optimizer = self.gan.create_optimizer(config.optimizer)

        d_grads = tf.gradients(d_loss, gan.d_vars())
        g_grads = tf.gradients(g_loss, gan.g_vars())
        dx = d_grads
        dy = g_grads
        x_loss = d_loss
        y_loss = g_loss

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        min_params = gan.d_vars()
        max_params = gan.g_vars()
        clarified_d_grads = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_sv_dontsave") for v in d_grads]
        clarified_g_grads = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_sv_dontsave") for v in g_grads]
        rhs_y = clarified_d_grads
        rhs_x = clarified_g_grads
        #rhs_y = d_grads
        #hs_x = g_grads
        clarified_grads = clarified_d_grads + clarified_g_grads
        lr = self.config.learn_rate
        class CGOperator:
            def __init__(self, hvp, x_loss, y_loss, x_params, y_params):
                self.hvp = hvp
                self.x_loss = x_loss
                self.y_loss = y_loss
                self.x_params = x_params
                self.y_params = y_params

            def apply(self, p):
                h_1_v = self.hvp(self.x_loss, self.y_params, self.x_params, [lr * _p for _p in p])
                for _x, _h in zip(self.x_params, h_1_v):
                    if _h is None:
                        print("X none", _x)
                return self.hvp(self.y_loss, self.x_params, self.y_params, [lr * _h for _h in h_1_v])

        operator_x = CGOperator(hvp=self.hessian_vector_product, x_loss=x_loss, y_loss=y_loss, x_params=min_params, y_params=max_params)
        reset_x_op, cg_x_op, var_x, state_x = tf_conjugate_gradient( operator_x, rhs_x, lr=lr, max_iter=(self.config.nsteps or 10) )
        operator_y = CGOperator(hvp=self.hessian_vector_product, x_loss=y_loss, y_loss=x_loss, x_params=max_params, y_params=min_params)
        reset_y_op, cg_y_op, var_y, state_y = tf_conjugate_gradient( operator_y, rhs_y, lr=lr, max_iter=(self.config.nsteps or 10) )
        self._variables = var_x + var_y + clarified_g_grads + clarified_d_grads

        assign_x = [tf.assign(c, x) for c, x in zip(clarified_d_grads, d_grads)]
        assign_y = [tf.assign(c, y) for c, y in zip(clarified_g_grads, g_grads)]
        self.reset_clarified_gradients = tf.group(*(assign_x+assign_y))
        self.reset_conjugate_tracker = tf.group(reset_x_op, reset_y_op)
        self.conjugate_gradient_descend_t_1 = tf.group(cg_x_op, cg_y_op)
        assign_x = [tf.assign(c, x) for c, x in zip(clarified_d_grads, state_y.x)]
        assign_y = [tf.assign(c, y) for c, y in zip(clarified_g_grads, state_x.x)]
        self.conjugate_gradient_descend_t_2 = tf.group(*(assign_x+assign_y))
        self.gan.add_metric('cg_g', sum([ tf.reduce_sum(tf.abs(_p)) for _p in clarified_g_grads]))

        f = d_loss
        g = g_loss
        dyg = tf.gradients(g, max_params)
        dxf = tf.gradients(f, min_params)

        if self.config.sga_lambda:
            hyp_y = self.hessian_vector_product(f, max_params, min_params, [self.config.sga_lambda * _g for _g in dyg])
            hyp_x = self.hessian_vector_product(g, min_params, max_params, [self.config.sga_lambda * _g for _g in dxf])
            sga_x_op = [tf.assign_sub(_g, _h) for _g, _h in zip(clarified_g_grads, hyp_x)]
            sga_y_op = [tf.assign_sub(_g, _h) for _g, _h in zip(clarified_d_grads, hyp_y)]
            self.sga_step_t = tf.group(*(sga_x_op + sga_y_op))
            self.gan.add_metric('hyp_x', sum([ tf.reduce_mean(_p) for _p in hyp_x]))
            self.gan.add_metric('hyp_y', sum([ tf.reduce_mean(_p) for _p in hyp_y]))

        self.clarification_metric_x = sum(state_x.rdotr)
        self.clarification_metric_y = sum(state_y.rdotr)
        self.state_x = state_x

        all_vars = min_params + max_params
        new_grads_and_vars = list(zip(clarified_grads, all_vars)).copy()

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
        if self.config.sga_lambda:
            sess.run(self.sga_step_t, feed_dict)
        while True:
            i+=1
            mx, my, _ = sess.run([self.clarification_metric_x, self.clarification_metric_y, self.conjugate_gradient_descend_t_1], feed_dict)
            if self.config.max_steps and i > self.config.max_steps:
               if self.config.verbose:
                   print("Max steps ", self.config.max_steps, "mx", mx, "my", my)
               break
            if self.config.trim_threshold is not None and (mx > self.config.trim_threshold or my > self.config.trim_threshold):
                print("Trimming MX = %.2e MY = %.2e" % (mx, my))
                self.trim.before_step(self.current_step, feed_dict)
                result = self._step(feed_dict)
                self.trim.after_step(self.current_step, feed_dict)
                return result
            #if self.config.verbose:
            #    print("MX = %.2e MY = %.2e" % (mx, my))
            if mx < (threshold_g) and \
               my < (threshold_d):
                   sess.run(self.conjugate_gradient_descend_t_2)
                   print("Found in ", i)
                   break
        metric_values = sess.run([self.optimize_t] + self.output_variables(metrics), feed_dict)[1:]
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            #print("METRICS", list(zip(sorted(metrics.keys()), metric_values)))
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

