"""AdaBound for TensorFlow."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.ops.clip_ops import clip_by_value
import tensorflow as tf

"""Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
        arad(boolean, optional): use arad
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

class AdaBoundOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
            beta1_power="decay(range=0.1:0 steps=10000 metric=b1)",
            beta2_power="decay(range=0.999:0 steps=10000 metric=b2)",
            lower_bound="decay(range=0:1 steps=10000 metric=lower)", 
            upper_bound="decay(range=1000:1 steps=10000 metric=upper)", 
            epsilon=1e-8, amsbound=False, gan=None, arad=False,
                 config={},
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(use_locking, name)
        self._lr = gan.configurable_param(learning_rate)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._gan = gan

        self._lower_bound=gan.configurable_param(lower_bound)
        self._upper_bound=gan.configurable_param(upper_bound)
        self._beta1_power=gan.configurable_param(beta1_power)
        self._beta2_power=gan.configurable_param(beta2_power)
        self._amsbound = amsbound
        self._arad = arad
        self.config = config

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        graph = None if context.executing_eagerly() else ops.get_default_graph()
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)


    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._base_lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr_t = lr_t * tf.sqrt(1-beta2_t)/(1-beta1_t)

        lower_bound = lr_t * self._lower_bound
        upper_bound = lr_t * self._upper_bound

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound :
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else :
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)


        # Compute the bounds
        step_size_bound = lr_t / (v_sqrt + epsilon_t)
        if isinstance(self.config.lower_bound, int) and self.config.lower_bound < 0:
            bounded_lr = m_t * step_size_bound
        else:
            bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

        if self._arad:
            bounded_lr *= (self.config.arad_lambda or 1.0) * tf.abs(m_t)

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_dense(self, grad, var):
        return self._apply(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self._apply(grad, var)
