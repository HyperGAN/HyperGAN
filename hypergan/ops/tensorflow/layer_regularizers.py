import tensorflow as tf

def layer_norm_1(component, net):
    ops = component.ops
    with tf.variable_scope(ops.generate_name(), reuse=ops._reuse):
        layer = tf.contrib.layers.layer_norm(net, scope='layer_norm', center=True, scale=True, variables_collections=tf.GraphKeys.LOCAL_VARIABLES)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_norm')
    ops.add_weights(vars)

    return layer

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
    name = ops.generate_name()
    with tf.variable_scope(name, reuse=ops._reuse):
        net = tf.contrib.layers.batch_norm(net, 
                decay = decay,
                center = center,
                scale = scale,
                epsilon = epsilon,
                is_training = True, #TODO: research
                scope=name
                )
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        filtered = []
        for var in vars:
            if var.name.startswith(name):
                filtered.append(var)
    ops.add_weights(filtered)
    return net
