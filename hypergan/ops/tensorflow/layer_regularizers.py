import tensorflow as tf

def layer_norm_1(component, net):
    ops = component.ops
    scope = ops.generate_name()
    with tf.variable_scope(scope, reuse=ops._reuse):
        net = tf.contrib.layers.layer_norm(net, scope=scope, center=True, scale=True, variables_collections=tf.GraphKeys.LOCAL_VARIABLES)
        vars = lookup_vars(scope)
    if not ops._reuse:
        ops.add_weights(vars)

    return net

def batch_norm_1(component, net):
    config = component.config
    ops = component.ops

    dtype = ops.dtype
    shape = ops.shape(net)

    epsilon = config.epsilon or 0.001
    batch_norm_gamma_stddev = config.batch_norm_gamma_stddev or 0.02

    decay = config.batch_norm_decay or 0.999
    center = config.batch_norm_center or True
    scale = config.batch_norm_scale or False
    epsilon = config.batch_norm_epsilon or 0.001
    scope = ops.generate_name()
    with tf.variable_scope(scope, reuse=ops._reuse):
        net = tf.contrib.layers.batch_norm(net, 
                decay = decay,
                center = center,
                scale = scale,
                epsilon = epsilon,
                is_training = True,
                scope=scope
                )
        vars = lookup_vars(scope)
    if not ops._reuse:
        ops.add_weights(vars)
    return net


def lookup_vars(name):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    filtered = []
    for var in vars:
        if var.name.startswith(name):
            filtered.append(var)
    return filtered

