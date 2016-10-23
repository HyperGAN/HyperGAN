def discriminator_densenet(config, x):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_densenet_layers']
    depth = config['d_densenet_block_depth']
    k = config['d_densenet_k']
    result = x
    result = conv2d(result, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)
    for i in range(layers):
        if i != layers-1:
            result = dense_block(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i))
        else:
          for j in range(depth):
            result = dense_block(result, k, activation, batch_size, 'layer', 'd_layers_'+str(i)+"_"+str(j))
            print("densenet size", result)


    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
    result = tf.reshape(result, [batch_size, -1])

    return result


