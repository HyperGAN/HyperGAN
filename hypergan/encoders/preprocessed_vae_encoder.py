#todo, this doesn't work
def z_from_f(config, f, categories):
    batch_size = config["batch_size"]
    transfer_fct = config['transfer_fct']
    n_z = int(config['z_dim'])
    n_c = sum(config['categories'])
    batch_norm = config['generator.regularizers.layer']

    result = f
    print("RESULT IS", result)
    if(config['f_skip_fc']):
        pass
    else:
        result = tf.reshape(result, [config['batch_size'], 2048])
        result = linear(result, config['f_hidden_1'], scope="v_f_hidden")
        result = batch_norm(config['batch_size'], name='v_f_hidden_bn')(result)
        result = transfer_fct(result)
        result = linear(result, config['f_hidden_2'], scope="v_f_hidden2")
        result = batch_norm(config['batch_size'], name='v_f_hidden_bn2')(result)
        result = transfer_fct(result)
    last_layer = result
    result = linear(result, n_z, scope="v_f_hidden3")
    result = batch_norm(config['batch_size'], name='v_f_hidden_bn3')(result)
    result = transfer_fct(result)

    result = tf.reshape(result, [config['batch_size'], -1])

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=config['dtype']))
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

    if config['category_loss']:
        e_c = linear(e_z,n_c, 'v_ez_lin')
        e_c = [tf.nn.softmax(x) for x in split_categories(e_c, config['batch_size'], categories)]
    else:
        e_c = []




    if(config['e_last_layer']):
        z = config['e_last_layer'](z)
        e_z = config['e_last_layer'](e_z)
    return e_z, e_c, z, mu, sigma


