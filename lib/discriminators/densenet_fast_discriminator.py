
def discriminator_fast_densenet(config, x):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_densenet_layers']
    depth = config['d_densenet_block_depth']
    k = config['d_densenet_k']
    result = x
    result = conv2d(result, 64, name='d_expand', k_w=3, k_h=3, d_h=2, d_w=2)
    result = batch_norm(config['batch_size'], name='d_expand_bn1a')(result)
    result = activation(result)
    result = conv2d(result, 128, name='d_expand2', k_w=3, k_h=3, d_h=2, d_w=2)
    for i in range(layers):
      for j in range(depth):
        result = dense_block(result, k, activation, batch_size, 'layer', 'd_layers_'+str(i)+"_"+str(j))
        print("Discriminator densenet layer:", result)
      result = dense_block(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i))


    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    while filter_size_h > 1:
      result = dense_block(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i+10))
      filter_size_w = int(result.get_shape()[1])
      filter_size_h = int(result.get_shape()[2])
      print("densenet size", result)
      i+=1
    result = tf.reshape(result, [batch_size, -1])

    return result


