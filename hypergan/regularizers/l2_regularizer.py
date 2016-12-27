import tensorflow as tf

def get(config):
    ws = None
    with tf.variable_scope("generator"):
        with tf.variable_scope("g_lin_proj"):
            tf.get_variable_scope().reuse_variables()
            ws = tf.get_variable('Matrix',dtype=config['dtype'])
            tf.get_variable_scope().reuse_variables()
        lam = config['generator.regularizers.l2.lambda']
        return [lam*tf.nn.l2_loss(ws)]



