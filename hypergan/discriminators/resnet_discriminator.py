
def discriminator_vanilla(config, x):
    x_dims = config['x_dims']
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    channels = (config['channels'])
    activation = config['discriminator.activation']

    result = x
    if config['conv_d_layers']:
        result = build_conv_tower(result, config['conv_d_layers'][:1], config['d_pre_res_filter'], config['batch_size'], config['d_batch_norm'], True, 'd_', activation, stride=config['d_pre_res_stride'])
        if(config['d_pool']):
            result = tf.nn.max_pool(result, [1, 3, 3, 1], [1, 2,2,1],padding='SAME')
        result = activation(result)
        result = build_resnet(result, config['d_resnet_depth'], config['d_resnet_filter'], 'd_conv_res_', activation, config['batch_size'], config['d_batch_norm'], conv=True)
        result = build_conv_tower(result, config['conv_d_layers'][1:], config['d_conv_size'], config['batch_size'], config['d_batch_norm'], config['d_batch_norm_last_layer'], 'd_2_', activation)
        result = tf.reshape(result, [batch_size, -1])

    return result


