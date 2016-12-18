def discriminator_wide_resnet(config, x):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_wide_resnet_depth']
    batch_norm = config['generator.regularizers.layer']
    result = x
    result = conv2d(result, layers[0], name='d_expand1a', k_w=3, k_h=3, d_h=1, d_w=1)
    result = batch_norm(config['batch_size'], name='d_expand_bn1a')(result)
    result = activation(result)
    result = conv2d(result, layers[0], name='d_expand1b', k_w=1, k_h=1, d_h=1, d_w=1)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_2')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_3')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_4')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_5')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_6')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_7')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_8')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_9')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_10')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_11')
    print("DRESULT", result)
    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
    print("RESULT SIZE IS", result)
    result = tf.reshape(result, [batch_size, -1])

    return result


