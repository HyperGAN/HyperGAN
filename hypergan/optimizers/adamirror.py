# From https://gist.github.com/lgeiger/10a3b1a0b94b52bc64d14d949ad74595
# Training GANs with Optimism using Tensorflow (https://github.com/vsyrgkanis/optimistic_GAN_training/) 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class AdamirrorOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adamirror"):
        super(AdamirrorOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _get_beta_accumulators(self):
        if context.executing_eagerly():
            graph = None
        else:
            graph = ops.get_default_graph()
        return (self._get_non_slot_variable("beta1_power", graph=graph),
                self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._beta1,
                                       name="beta1_power",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2,
                                       name="beta2_power",
                                       colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        old_update = lr * m / (math_ops.sqrt(v) + epsilon_t)
        var_update = state_ops.assign_add(var, old_update, use_locking=self._use_locking)

        with ops.control_dependencies([var_update]):
            m_t = state_ops.assign(m,
                                   m * beta1_t + (grad * (1 - beta1_t)),
                                   use_locking=self._use_locking)

            v_t = state_ops.assign(v,
                                   v * beta2_t + ((grad * grad) * (1 - beta2_t)),
                                   use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var,
                                          2 * lr * m_t / (math_ops.sqrt(v_t) + epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)

