import tensorflow as tf
import warnings


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar
