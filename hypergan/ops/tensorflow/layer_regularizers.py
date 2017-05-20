class batch_norm_second_half(object):
    """Code modification of http://stackoverflow.com/a/33950177

    """
    def __init__(self, epsilon=1e-5,  name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon

            self.name=name

    def __call__(self, x):

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            second_half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])

            self.mean, self.variance = tf.nn.moments(second_half, [0, 1, 2])

            out =  tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

class batch_norm_first_half(object):
    """Code modification of http://stackoverflow.com/a/33950177

    """
    def __init__(self, epsilon=1e-5,  name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon

            self.name=name

    def __call__(self, x):

        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.random_normal_initializer(1., 0.02,dtype=config['dtype']))
            self.beta = tf.get_variable("beta", [shape[-1]],dtype=config['dtype'],
                                initializer=tf.constant_initializer(0.,dtype=config['dtype']))

            first_half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])

            self.mean, self.variance = tf.nn.moments(first_half, [0, 1, 2])

            out =  tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


