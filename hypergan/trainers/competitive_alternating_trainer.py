import numpy as np
import hyperchamber as hc
import inspect
import collections

from tensorflow.python.ops import math_ops
from hypergan.trainers.base_trainer import BaseTrainer
from tensorflow.python.ops import gradients_impl

TINY = 1e-12

cg_state = collections.namedtuple("CGState", ["i", "x", "r", "p", "lr", "rdotr", "metric"])
def update_vars(state1, state2):
  ops = []
  for name in state1._fields:
    state1_vs = getattr(state1, name)
    print("NN", name)
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
        sv = tf.Variable(tf.zeros_like(vs), trainable=False, name=name+"_sv_dontsave")
        variables += [sv]
    args.append(sv)
  return cg_state(*args), variables

def dot(x, y):
  x = [tf.reshape(_x, [-1]) for _x in x]
  y = [tf.reshape(_y, [-1]) for _y in y]
  x = tf.concat(x, axis=0)
  y = tf.concat(y, axis=0)
  return tf.reduce_sum(tf.multiply(x,y))

def tf_conjugate_gradient(operator,
                       rhs,
                       x,
                       lr=1e-4,
                       tol=1e-4,
                       max_iter=20,
                       config={},
                       gan=None,
                       name="conjugate_gradient"):
    if config.method == "optimizer":
      optimizer = gan.create_optimizer(config.optimizer)
    r"""
        modified from tensorflow/contrib/solvers/python/ops/linear_equations.py to work with arrays
    """
    def cg_step(state):  # pylint: disable=missing-docstring
      def _conjugate(state):
          h_2_v = operator.apply(state.p)
          #alpha = dot(state.r, state.p) / (dot(state.p, Hp)+1e-8)
          #alpha = tf.Print(alpha, [alpha], 'alpha')
          alpha = 1.0
          x = [alpha * _p + _x for _p, _x in zip(state.p, state.x)]
          Hp = h_2_v
          Avp_ = [_p + _h_2 for _p, _h_2 in zip(state.p, h_2_v)]
          Hp = Avp_

          r = [-alpha * _avp_+_r for _avp_,_r in zip(Avp_, state.r)]
          #beta = dot(r, Hp) / (dot(state.r, Hp)+1e-8)
          #beta = dot(r, r) / (dot(state.r, state.r)+1e-8)
          beta = 0.1
          y = [_r - _r_1 for _r, _r_1 in zip(r, state.r)]
          #beta = -dot(r, r) / (tf.maximum( dot( state.p, y ), dot( [-_p for _p in state.p], state.r ) )+1e-32)
          #beta = tf.Print(beta, [beta], 'beta')
          p = [_r + beta * _p for _r, _p in zip(r,state.p)]
          i = state.i + 1
          metric = dot(r,r)

          return cg_state(i, x, r, p, lr, dot(r,r), metric)
      def _sum(state):
          h_2_v = operator.apply(state.p)
          if(config.normalize == True):
              norm_factor = tf.sqrt(dot(state.p,state.p)) / (tf.sqrt(dot(h_2_v, h_2_v))+1e-32)
              h_2_v = [norm_factor * _h for _h in h_2_v]
          if(config.normalize == 2):
              norm_factor = tf.sqrt(tf.abs(dot(h_2_v,state.p))) / (tf.sqrt(dot(h_2_v, h_2_v))+1e-32)
              h_2_v = [norm_factor * _h for _h in h_2_v]
          if(config.normalize == 3):
              norm_factor = tf.sqrt(tf.abs(dot(rhs,rhs))) / (tf.sqrt(dot(h_2_v, h_2_v))+1e-32)
              h_2_v = [norm_factor * _h for _h in h_2_v]
          if(config.normalize == 4):
              rhs_mean = [tf.reduce_mean(tf.abs(_g)) for _g in rhs]
              h_2_v_mean = [tf.reduce_mean(tf.abs(_g)) for _g in h_2_v]
              rhs_mean = sum(rhs_mean)  / len(rhs_mean)
              h_2_v_mean = sum(h_2_v_mean)  / len(h_2_v_mean)
              norm_factor = rhs_mean / h_2_v_mean
              h_2_v = [norm_factor * _h for _h in h_2_v]
          if(config.normalize == 5):
              rhs_mean = [tf.reduce_mean(tf.abs(_g)) for _g in rhs]
              h_2_v_mean = [tf.reduce_mean(tf.abs(_g)) for _g in h_2_v]
              norm_factor = [_r / (_h+1e-32) for _r, _h in zip(rhs_mean, h_2_v_mean)]
              h_2_v = [_n * _h for _n, _h in zip(norm_factor, h_2_v)]
              norm_factor = sum(norm_factor) / len(norm_factor)


          if config.force:
              p = h_2_v
          else:
              p = [_p + (config.decay or 0.05)*_h_2_v for _p, _h_2_v, _r1 in zip(state.p, h_2_v, state.r)]

          x = p
          r = h_2_v
          #rdotr = dot(r, r)
          #rdotr = tf.Print(rdotr, [rdotr], "rdotr")
          if config.metric == "ddr":
            rdotr = (dot(r,r)-dot(state.r,state.r)) / (dot(rhs,rhs)+1e-32)
            metric = rdotr - state.rdotr
          else:
            rdotr = dot(r,r)/ (dot(rhs,rhs)+1e-32)
            metric = rdotr

          #rdotr = sum([tf.reduce_sum(tf.abs(_x)) for _x in x])
          #metric = tf.abs((rdotr-state.rdotr) / rdotr)

          #metric = tf.Print(metric, [metric], "metric")
          #metric = tf.Print(metric, [state.rdotr], "staterdotr")
          #metric = tf.Print(metric, [rdotr], "dotr")
          i = state.i + 1

          return cg_state(i, x, r, p, lr, rdotr, metric)
    
      def _optimizer(state):
        grads = operator.apply(state.x)
        variables = state.x
        r = grads
        x = state.x
        p = x
        i = state.i + 1
        rdotr = dot(r,r)/ (dot(rhs,rhs)+1e-32)
        metric = rdotr

        grads_and_vars = list(zip(grads, state.x)).copy()

        return [cg_state(i, x, r, p, lr, rdotr, metric), grads_and_vars]

      method = _conjugate
      if config.method == "sum":
          method = _sum
      if config.method == "optimizer":
          method = _optimizer
      return method(state)

    with tf.name_scope(name):
      rdotr = 0.0#dot(rhs, rhs)
      if config.method == "optimizer":
          state = cg_state(i=0, x=rhs, r=rhs, p=rhs, lr=lr, rdotr=rdotr, metric=1.0)
          state, variables = build_vars(state)
          variables += optimizer.variables()
          def update_op(state):
            step = cg_step(state)
            op1= optimizer.apply_gradients(step[1])
            with tf.control_dependencies([op1]):
                return update_vars(state, step[0])
    
          def reset_op(state, rhs):
            update_state = update_vars(state, cg_step(cg_state(i=0, x=rhs, r=rhs, p=rhs, lr=lr, rdotr=rdotr, metric=1.0))[0])
            reset_optimizer = tf.variables_initializer(optimizer.variables())
            return tf.group(update_state, reset_optimizer)
      else:
          r = operator.apply(rhs)
          state = cg_state(i=0, x=x, r=r, p=rhs, lr=lr, rdotr=rdotr, metric=1.0)
          state, variables = build_vars(state)
          def update_op(state):
            return update_vars(state, cg_step(state))
          def reset_op(state, rhs):
            return update_vars(state, cg_step(cg_state(i=0, x=x, r=r, p=r, lr=lr, rdotr=rdotr, metric=1.0)))
      return [reset_op(state, rhs), update_op(state), variables, state]

class CompetitiveAlternatingTrainer(BaseTrainer):
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
    def fwd_gradients(self, ys, xs, grad_xs=None, stop_gradients=None, colocate_gradients_with_ops=True, us=None):
      if us is None:
          us = [tf.zeros_like(y) + float('nan') for y in ys]
      print("YS", len(ys), "US", len(us))
      dydxs = tf.gradients(ys, xs, grad_ys=us,stop_gradients=stop_gradients)
      dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs)
      return dysdx


    def hessian_vector_product9(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]
        if grads is None:
            grads = tf.gradients(ys, xs)
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        grads = self.fwd_gradients(grads, xs, grad_xs=vs)
        grads = tf.gradients(grads, xs2)
        return grads


    def hessian_vector_product5(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        grads = self.fwd_gradients(grads, xs)
        grads = tf.gradients(grads, xs2, vs)
        return grads


    def hessian_vector_product10(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs2)
        grads = self.fwd_gradients(grads, xs, grad_xs=vs)
        return grads

    def hessian_vector_product13(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        print("vS", vs)
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones, grad_ys=vs)
        rop = tf.gradients(rop, xs2)
        print("ROP", rop)
        return rop



    def hessian_vector_product15(self, ys, xs, xs2, vs, grads=None):
        if len(vs) != len(xs):
            raise ValueError("xs and v must have the same length.")
        #offset = [_x + _o for _x, _o in zip(xs, vs)]

        grads = tf.gradients(ys, xs)
        grads = [_g * (self.config.hvp_lambda or self.config.learn_rate) for _g in grads]
        #grads = [_g + _v for _g, _v in zip(grads, vs)]
        print("vS", vs)
        ones = [tf.ones_like(_v) for _v in grads]
        lop = tf.gradients(grads, xs, grad_ys=ones)
        rop = tf.gradients(lop, ones)
        rop = tf.gradients(rop, xs2, vs)
        print("ROP", rop)
        return rop





    """ Steps G and D simultaneously """
    def _create(self):
        gan = self.gan
        config = self.config

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        g_optimizer = config.g_optimizer or config.optimizer
        d_optimizer = config.d_optimizer or config.optimizer
        d_optimizer["loss"] = d_loss
        g_optimizer["loss"] = g_loss
        g_optimizer = self.gan.create_optimizer(g_optimizer)
        d_optimizer = self.gan.create_optimizer(d_optimizer)



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
                print("D %d %d %d %d" % (len(self.y_params), len(self.x_params), len(p), len(gan.d_vars())))
                h_1_v = self.hvp(self.x_loss, self.x_params, self.y_params, [lr * _p for _p in p])
                for _x, _h in zip(self.x_params, h_1_v):
                    if _h is None:
                        print("X none", _x)
                return self.hvp(self.y_loss, self.y_params, self.x_params, [lr * _h for _h in h_1_v])

        old_y = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_oldx_dontsave") for v in d_grads]
        old_x = [tf.Variable(tf.zeros_like(v), trainable=False, name=v.name.split(":")[0]+"_oldy_dontsave") for v in g_grads]
        #old_y = [tf.zeros_like(v) for v in d_grads]
        #old_x = [tf.zeros_like(v) for v in g_grads]

        hvp = self.hessian_vector_product9
        if self.config.hvp == 5:
            hvp = self.hessian_vector_product5
        if self.config.hvp == 10:
            hvp = self.hessian_vector_product10
        if self.config.hvp == 11:
            hvp = self.hessian_vector_product11
        if self.config.hvp == 13:
            hvp = self.hessian_vector_product13
        if self.config.hvp == 14:
            hvp = self.hessian_vector_product14
        if self.config.hvp == 15:
            hvp = self.hessian_vector_product15
        if self.config.hvp == 12:
            hvp = self.hessian_vector_product


        operator_x = CGOperator(hvp=hvp, x_loss=x_loss, y_loss=y_loss, x_params=min_params, y_params=max_params)
        reset_x_op, cg_x_op, var_x, state_x = tf_conjugate_gradient( operator_x, rhs=rhs_y, x=old_y, lr=lr, max_iter=(self.config.nsteps or 10), config=self.config.conjugate, gan=self.gan )
        operator_y = CGOperator(hvp=hvp, x_loss=y_loss, y_loss=x_loss, x_params=max_params, y_params=min_params)
        reset_y_op, cg_y_op, var_y, state_y = tf_conjugate_gradient( operator_y, rhs=rhs_x, x=old_x, lr=lr, max_iter=(self.config.nsteps or 10), config=self.config.conjugate, gan=self.gan )
        self._variables = var_x + var_y + clarified_g_grads + clarified_d_grads + old_x + old_y

        self.reset_y = [tf.assign(c, x) for c, x in zip(clarified_d_grads, d_grads)]
        self.reset_x = [tf.assign(c, y) for c, y in zip(clarified_g_grads, g_grads)]
        self.reset_conjugate_x = reset_x_op
        self.reset_conjugate_y = reset_y_op
        self.conjugate_gradient_descend_x1 = cg_x_op
        self.conjugate_gradient_descend_y1 = cg_y_op
        if self.config.final_hessian:
            final_y = [ _g * _s for _g, _s in zip(d_grads, state_y.x) ]
            final_x = [ _g * _s for _g, _s in zip(g_grads, state_x.x) ]
            final_x2 = tf.gradients(final_y, max_params)
            final_y2 = tf.gradients(final_x, min_params)
            hcg_x = [self.config.final_lambda * _g + _cg for _g, _cg in zip(final_x2, g_grads)]
            hcg_y = [self.config.final_lambda * _g + _cg for _g, _cg in zip(final_y2, d_grads)]
            self.gan.add_metric('hcg_x', sum([ tf.reduce_sum(tf.abs(_p)) for _p in hcg_x]))
            self.gan.add_metric('hcg_y', sum([ tf.reduce_sum(tf.abs(_p)) for _p in hcg_y]))
            assign_old_x = [tf.assign(c, x) for c, x in zip(old_x, hcg_x)]
            assign_old_y = [tf.assign(c, y) for c, y in zip(old_y, hcg_y)]
            assign_y = [tf.assign(c, x) for c, x in zip(clarified_d_grads, state_y.x)]
            assign_x = [tf.assign(c, y) for c, y in zip(clarified_g_grads, state_x.x)]
            self.conjugate_gradient_descend_x2 = tf.group(*(assign_x + assign_old_y))
            self.conjugate_gradient_descend_y2 = tf.group(*(assign_y + assign_old_x))
        else:
            final_y = [((self.config.risk or 0.0) * _g) + ((1.0-(self.config.risk or 0.0)) * _s) for _g, _s in zip(clarified_g_grads, state_y.x)]
            final_x = [((self.config.risk or 0.0) * _g) + ((1.0-(self.config.risk or 0.0)) * _s) for _g, _s in zip(clarified_d_grads, state_x.x)]
            assign_y = [tf.assign(c, x) for c, x in zip(clarified_g_grads, final_y)]
            assign_x = [tf.assign(c, y) for c, y in zip(clarified_d_grads, final_x)]
            self.conjugate_gradient_descend_x2 = assign_x
            self.conjugate_gradient_descend_y2 = assign_y

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
            self.sga_step_x = tf.group(*sga_x_op)
            self.sga_step_y = tf.group(*sga_y_op)
            self.gan.add_metric('hyp_x', sum([ tf.reduce_mean(_p) for _p in hyp_x]))
            self.gan.add_metric('hyp_y', sum([ tf.reduce_mean(_p) for _p in hyp_y]))

        self.clarification_metric_x = state_x.metric
        self.clarification_metric_y = state_y.metric
        self.state_x = state_x
        self.state_y = state_y

        all_vars = min_params + max_params
        new_grads_and_vars = list(zip(clarified_grads, all_vars)).copy()

        if self.config.trim:
            self.trim = self.gan.create_component(self.config.trim, name='trim')
            self.trim.after_create()

        apply_vec_g = list(zip(clarified_g_grads, max_params)).copy()
        apply_vec_d = list(zip(clarified_d_grads, min_params)).copy()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self
        g_optimizer_t = g_optimizer.apply_gradients(apply_vec_g)
        d_optimizer_t = d_optimizer.apply_gradients(apply_vec_d)

        self.d_optimizer_t = d_optimizer_t
        self.g_optimizer_t = g_optimizer_t
 
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

        ### Update D
        sess.run(self.reset_y, feed_dict)
        sess.run(self.reset_conjugate_y, feed_dict)
        i = 0
        while True:
            i+=1
            my, _ = sess.run([self.clarification_metric_y, self.conjugate_gradient_descend_y1], feed_dict)
            if self.config.verbose:
                print("My = %.2e" % (my))
            if (i >= (self.config.max_steps or 1e10)) or (i > 1 and my < (self.config.threshold or 1e-4)):
                   sess.run(self.conjugate_gradient_descend_y2)
                   if self.current_step % 10 == 0:
                       print("Y Found in ", i)
                   break
        for i in range(config.d_update_steps or 1):
            sess.run([self.d_optimizer_t], feed_dict)

        ### Update G
        sess.run(self.reset_x, feed_dict)
        sess.run(self.reset_conjugate_x, feed_dict)
        i=0
        if self.config.sga_lambda:
            sess.run(self.sga_step_x, feed_dict)
        while True:
            i+=1
            mx, _ = sess.run([self.clarification_metric_x, self.conjugate_gradient_descend_x1], feed_dict)
            if self.config.verbose:
                print("MX = %.2e" % (mx), i)
            if (i >= (self.config.max_steps or 1e10)) or (i > 1 and mx < (self.config.threshold or 1e-4)):
                   sess.run(self.conjugate_gradient_descend_x2)
                   if self.current_step % 10 == 0:
                       print("X Found in ", i)
                   break

        metric_values = sess.run([self.g_optimizer_t] + self.output_variables(metrics), feed_dict)[1:]

        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            #print("METRICS", list(zip(sorted(metrics.keys()), metric_values)))
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)))

