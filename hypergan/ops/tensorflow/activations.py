def decayer(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [1], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        decay_scale = tf.get_variable("decay_scale", [1], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))

def decayer2(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.,dtype=config['dtype']),dtype=config['dtype'])
        decay_scale = tf.get_variable("decay_scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.,dtype=config['dtype']), dtype=config['dtype'])
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))


