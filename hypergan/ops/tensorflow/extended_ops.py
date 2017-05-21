def constrained_conv2d(input_, output_dim,
           k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    assert k_h % d_h == 0
    assert k_w % d_w == 0
    # constrained to have stride be a factor of kernel width
    # this is intended to reduce convolution artifacts
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],dtype=config.dtype,
                            initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=config.dtype))

        # This is meant to reduce boundary artifacts
        padded = tf.pad(input_, [[0, 0],
            [k_h-1, 0],
            [k_w-1, 0],
            [0, 0]])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0,dtype=config.dtype),dtype=config.dtype)
        conv = tf.nn.bias_add(conv, biases)

        return conv

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis=3, values=[x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def special_deconv2d(input_, output_shape,
             k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False,
             init_bias=0.):
    # designed to reduce padding and stride artifacts in the generator

    # If the following fail, it is hard to avoid grid pattern artifacts
    assert k_h % d_h == 0
    assert k_w % d_w == 0

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],dtype=config['dtype'],
                            initializer=tf.random_normal_initializer(stddev=stddev,dtype=config['dtype']))

        def check_shape(h_size, im_size, stride):
            if h_size != (im_size + stride - 1) // stride:
                print( "Need h_size == (im_size + stride - 1) // stride")
                print( "h_size: ", h_size)
                print( "im_size: ", im_size)
                print( "stride: ", stride)
                print( "(im_size + stride - 1) / float(stride): ", (im_size + stride - 1) / float(stride))
                raise ValueError()

        check_shape(int(input_.get_shape()[1]), output_shape[1] + k_h, d_h)
        check_shape(int(input_.get_shape()[2]), output_shape[2] + k_w, d_w)

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[output_shape[0],
            output_shape[1] + k_h, output_shape[2] + k_w, output_shape[3]],
                                strides=[1, d_h, d_w, 1])
        deconv = tf.slice(deconv, [0, k_h // 2, k_w // 2, 0], output_shape)


        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias,dtype=config['dtype']),dtype=config['dtype'])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


