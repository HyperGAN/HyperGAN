def conv_layer(result, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    size = int(result.get_shape()[-1])
    result = activation(result)
    if(batch_norm is not None):
        result = batch_norm(batch_size, name=name+'bn')(result)
    s = result.get_shape()
    if(sigmoid_gate is not None):
        mask = linear(sigmoid_gate, s[1]*s[2]*s[3], scope=name+"lin_proj_mask")
        mask = tf.reshape(mask, result.get_shape())
        result *= tf.nn.sigmoid(mask)

    if(resize):
        result = tf.image.resize_images(result, resize, 1)

    if reshaped_z_proj is not None:
        result = tf.concat(axis=3,values=[result, reshaped_z_proj])

    if(noise_shape):
      noise = tf.random_uniform(noise_shape,-1, 1,dtype=dtype)
      result = tf.concat(axis=3, values=[result, noise])
    if(id=='conv'):
        result = conv2d(result, int(result.get_shape()[3]), name=name, k_w=filter, k_h=filter, d_h=stride, d_w=stride)
    elif(id=='identity'):
        result = conv2d(result, output_channels, name=name, k_w=filter, k_h=filter, d_h=1, d_w=1)

    return result


def standard_layer(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    return conv_layer(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

def inception_layer(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    print("OUTPUT CHANLLL", output_channels)
    if output_channels == 3:
        return conv_layer(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)
    size = int(net.get_shape()[-1])
    if(batch_norm is not None):
        net = batch_norm(batch_size, name=name+'bn')(net)

    net = activation(net)
    s = net.get_shape()
    if(sigmoid_gate is not None):
        mask = linear(sigmoid_gate, s[1]*s[2]*s[3], scope=name+"lin_proj_mask")
        mask = tf.reshape(mask, net.get_shape())
        net *= tf.nn.sigmoid(mask)

    if output_channels == 3:
        return conv2d(net, output_channels, name=name, k_w=filter, k_h=filter, d_h=1, d_w=1)

    net1 = conv2d(net, output_channels//3, name=name+'1', k_w=1, k_h=1, d_h=1, d_w=1)
    net2 = conv2d(net1, output_channels//3, name=name+'2', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net3 = conv2d(net2, output_channels//3, name=name+'3', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net = tf.concat(axis=3, values=[net1, net2, net3])
    return net

def dense_layer(net,config,  activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return conv_layer(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

    net1 = conv_layer(net, activation, batch_size, 'identity', name, output_channels=max(output_channels-16, 16), filter=filter, batch_norm=config.layer_regularizer)
    net2 = conv_layer(net, activation, batch_size, 'identity', name+'2', output_channels=16, filter=filter, batch_norm=config.layer_regularizer)
    net = tf.concat(axis=3, values=[net1, net2])
    return net


def build_reshape(output_size, nodes, method, batch_size, dtype):
    node_size = sum([int(x.get_shape()[1]) for x in nodes])
    dims = output_size-node_size
    if(method == 'noise'):
        noise = tf.random_uniform([batch_size, dims],-1, 1, dtype=dtype)
        result = tf.concat(axis=1, values=nodes+[noise])
    elif(method == 'tiled'):
        t_nodes = tf.concat(axis=1, values=nodes)
        dims =  int(t_nodes.get_shape()[1])
        result= tf.tile(t_nodes,[1, output_size//dims])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            #zeros = tf.zeros([batch_size, width])
            slice = tf.slice(result, [0,0],[batch_size,width])
            result = tf.concat(axis=1, values=[result, slice])


    elif(method == 'zeros'):
        result = tf.concat(axis=1, values=nodes)
        result = tf.pad(result, [[0, 0],[dims//2, dims//2]])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            zeros = tf.zeros([batch_size, width],dtype=dtype)
            result = tf.concat(axis=1, values=[result, zeros])
    elif(method == 'linear'):
        result = tf.concat(axis=1, values=[y, z])
        result = linear(result, dims, 'g_input_proj')

    else:
        assert 1 == 0
    return result



