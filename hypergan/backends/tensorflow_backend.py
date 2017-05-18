class TensorflowBackend:
    def __init__(self, dtype="float32", initializer='orthogonal', orthogonal_gain=1.0, random_stddev=0.02):
        self.dtype = self.parse_dtype(dtype)
        self.scope_count = 0
        if initializer == 'orthogonal':
            self.initializer = self.orthogonal_initializer(orthogonal_gain)
        else:
            self.initializer = self.random_initializer(random_stddev)

    def random_initializer(self, stddev):
        def _build():
            return tf.random_normal_initializer(0, stddev, dtype=self.dtype)
        return _build

    def orthogonal_initializer(self, gain):
        def _build():
            return tf.orthogonal_initializer(gain)
        return _build

    def generate_scope(self):
        self.scope_count += 1
        return str(self.scope_count)

    def parse_dtype(self, dtype):
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16'
            return tf.float16
        else
            raise "dtype not defined: "+name

    def conv2d(net, filter_w, filter_h, stride_w, stride_h, output_dim):
        initializer = self.initializer()
        with tf.variable_scope(name):
            with tf.device("/cpu:0"):
                w = tf.get_variable('w', [filter_h, filter_w, net.get_shape()[-1], output_dim],dtype=self.dtype, regularizer=regularizer,
                                    initializer=initializer)
            conv = tf.nn.conv2d(net, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0, dtype=self.dtype), dtype=self.dtype)
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_shape):
        initializer = self.initializer()
        with tf.variable_scope(self.generate_scope()):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [filter_h, filter_w, output_shape[-1], input_.get_shape()[-1]], dtype=self.dtype, initializer=initializer)

            try:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])

            # Support for versions of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], dtype=self.dtype,initializer=tf.constant_initializer(init_bias, dtype=self.dtype))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            return deconv


    def linear(self, net, output_dim):
        initializer = self.initializer()
        #initializer = tf.constant_initializer(1)
        with tf.variable_scope(self.generate_scope()):
          with tf.device('/cpu:0'):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], dtype=config['dtype'],
                                       initializer=initializer, regularizer=regularizer
                                    )
          bias = tf.get_variable("bias", [output_size],dtype=config['dtype'],
              initializer=tf.constant_initializer(bias_start,dtype=config['dtype']))
          if with_w:
              # import ipdb; ipdb.set_trace()
            return tf.matmul(input_, matrix) + bias, matrix, bias
          else:
            return tf.matmul(input_, matrix) + bias


    def reshape(self, net, shape):
        return tf.reshape(net, shape)

    def concat(values=[], axis=0):
        return tf.concat(values=values, axis=axis)

    def resize_images(net, dims, op_type):
        return tf.image.resize_images(net, dims, op_type)

    def slice(net, x, y):
        return tf.slice(net, x, y)

    def shape(net):
        return [int(x) for x in net.get_shape()]

    def lookup(symbol):
        if symbol == 'tanh':
            return tf.nn.tanh
        if symbol == 'sigmoid':
            return tf.nn.sigmoid
        #TODO if symbol starts with function:
        return None
