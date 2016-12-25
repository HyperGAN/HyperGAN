#todo: broken
def get_zs(config, x,y):
    return approximate_z(config, x, [y])

def approximate_z(config, x, y):
    y = tf.concat(1, y)
    x_dims = config['x_dims']
    batch_size = config["batch_size"]
    transfer_fct = config['transfer_fct']
    x = tf.reshape(x, [config["batch_size"], -1,config['channels']])
    noise_dims = int(x.get_shape()[1])-int(y.get_shape()[1])
    n_z = int(config['z_dim'])
    channels = (config['channels']+1)

    result = build_reshape(int(x.get_shape()[1]), [y], config['d_project'], batch_size, config['dtype'])
    result = tf.reshape(result, [batch_size, -1, 1])
    result = tf.concat(2, [result, x])

    result = tf.reshape(result, [config["batch_size"], x_dims[0],x_dims[1],channels])

    if config['g_encode_layers']:
        result = build_conv_tower(result, 
                    config['g_encode_layers'], 
                    config['e_conv_size'], 
                    config['batch_size'], 
                    config['e_batch_norm'], 
                    config['e_batch_norm_last_layer'], 
                    'v_', 
                    transfer_fct
                    )

    result = transfer_fct(result)
    last_layer = result
    result = tf.reshape(result, [config['batch_size'], -1])

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    out_mean= tf.get_variable('v_out_mean', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    mu = tf.add(tf.matmul(result, out_mean),b_out_mean)

    out_log_sigma=tf.get_variable('v_out_logsigma', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    b_out_log_sigma= tf.get_variable('v_b_out_logsigma', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    sigma = tf.add(tf.matmul(result, out_log_sigma),b_out_log_sigma)

    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=config['dtype'])
    set_tensor('eps', eps)

    z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(sigma)), eps))

    e_z = tf.random_normal([config['batch_size'], n_z], mu, tf.exp(sigma), dtype=config['dtype'])

    if(config['e_last_layer']):
        z = config['e_last_layer'](z)
        e_z = config['e_last_layer'](e_z)
    return e_z, z, mu, sigma

