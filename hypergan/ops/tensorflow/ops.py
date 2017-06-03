import tensorflow as tf
import types
from hypergan.ops.tensorflow import layer_regularizers

class TensorflowOps:
    def __init__(self, dtype="float32", initializer='orthogonal', orthogonal_gain=1.0, random_stddev=0.02):
        self.dtype = self.parse_dtype(dtype)
        self.scope_count = 0
        self.description = ''
        self.weights = []
        self.biases = []
        if initializer == 'orthogonal':
            self.initializer = self.orthogonal_initializer(orthogonal_gain)
        else:
            self.initializer = self.random_initializer(random_stddev)

    def assert_tensor(self, net):
        if type(net) != tf.Tensor:
            raise Exception("Expected a Tensor but received", net)

    def variables(self):
        return self.biases + self.weights

    def random_initializer(self, stddev):
        def _build():
            return tf.random_normal_initializer(0, stddev, dtype=self.dtype)
        return _build

    def orthogonal_initializer(self, gain):
        def _build():
            return tf.orthogonal_initializer(gain)
        return _build

    def describe(self, description):
        self.description = description

    def generate_scope(self):
        self.scope_count += 1
        return str(self.scope_count)

    def generate_name(self):
        if self.description == "":
            return self.generate_scope()
        return self.description + "_" + self.generate_scope()

    def parse_dtype(self, dtype):
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16':
            return tf.float16
        else:
            raise Exception("dtype not defined: "+dtype)

    def describe(self, description):
        self.description = description

    def get_weight(self, shape):
        weight = tf.get_variable('w', shape, dtype=self.dtype, initializer=self.initializer())
        self.weights.append(weight)
        return weight

    def get_bias(self, shape):
        bias = tf.get_variable('b', shape, initializer=tf.constant_initializer(0.0, dtype=self.dtype), dtype=self.dtype)
        self.biases.append(bias)
        return bias
    
    def parse_dtype(self, dtype):
        if dtype == 'float32':
            return tf.float32
        elif dtype == 'float16':
            return tf.float16
        else:
            raise Exception("dtype not defined: "+dtype)

    def conv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        self.assert_tensor(net)
        with tf.variable_scope(self.generate_name()):
            w = self.get_weight([filter_h, filter_w, net.get_shape()[-1], output_dim])
            conv = tf.nn.conv2d(net, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
            biases = self.get_bias([output_dim])
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d(self, net, filter_w, filter_h, stride_w, stride_h, output_dim):
        self.assert_tensor(net)
        initializer = self.initializer()
        shape = self.shape(net)
        output_shape = [shape[0], shape[1]*stride_h, shape[2]*stride_w, output_dim]
        init_bias = 0.
        with tf.variable_scope(self.generate_name()):
            # filter : [height, width, output_channels, in_channels]
            w = self.get_weight([filter_h, filter_w, output_dim, output_dim])

            deconv = tf.nn.conv2d_transpose(net, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])

            biases = self.get_bias([output_shape[-1]])
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            return deconv

    def linear(self, net, output_dim):
        self.assert_tensor(net)
        initializer = self.initializer()
        shape = self.shape(net)
        with tf.variable_scope(self.generate_name()):
            w = self.get_weight([output_dim, output_dim])
            bias = self.get_bias([output_dim])
            return tf.matmul(net, w) + bias


    def layer_regularizer(self, net, symbol, epsilon):
        self.assert_tensor(net)
        op = self.lookup(symbol)
        if op:
            net = op(epsilon, self.generate_name())(net, self.dtype)
        return net

    def reshape(self, net, shape):
        self.assert_tensor(net)
        return tf.reshape(net, shape)

    def concat(self, values=[], axis=0):
        return tf.concat(values=values, axis=axis)

    def resize_images(self, net, dims, op_type):
        self.assert_tensor(net)
        return tf.image.resize_images(net, dims, op_type)

    def slice(self, net, x, y):
        self.assert_tensor(net)
        return tf.slice(net, x, y)

    def shape(self, net):
        self.assert_tensor(net)
        return [int(x) for x in net.get_shape()]

    def lookup(self, symbol):
        if symbol == None:
            return None
        if type(symbol) == types.FunctionType:
            return symbol
        if symbol == 'tanh':
            return tf.nn.tanh
        if symbol == 'sigmoid':
            return tf.nn.sigmoid
        if symbol == 'batch_norm':
            return layer_regularizers.batch_norm_1
        if symbol == 'layer_norm':
            return layer_regularizers.layer_norm_1
        #TODO if symbol starts with function:

        print("lookup failed for ", self.description, symbol)
        return None

    def init_session(self, device):
        # Initialize tensorflow
        with tf.device(device):
            self.sess = tf.Session(config=tf.ConfigProto())

    def create_graph(self, graph_type, device):
        tf_graph = hg.graph.Graph(self)
        graph = self.graph
        with tf.device(device):
            if 'y' in graph:
                # convert to one-hot
                graph.y=tf.cast(graph.y,tf.int64)
                graph.y=tf.one_hot(graph.y, self.config['y_dims'], 1.0, 0.0)

            if graph_type == 'full':
                tf_graph.create(graph)
            elif graph_type == 'generator':
                tf_graph.create_generator(graph)
            else:
                raise Exception("Invalid graph type")

    def initialize_graph(self):
        print(" |= Initializing new network")
        with tf.device(self.device):
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def load_or_initialize_graph(self, save_file):
        save_file = os.path.expanduser(save_file)
        if os.path.isfile(save_file) or os.path.isfile(save_file + ".index" ):
            print(" |= Loading network from "+ save_file)
            dir = os.path.dirname(save_file)
            print(" |= Loading checkpoint from "+ dir)
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(self.sess, save_file)
                loadedFromSave = True
                print("Model loaded")
            else:
                print("No checkpoint file found")
        else:
            self.initialize_graph()

    def save(self, save_file):
        saver = tf.train.Saver()
        saver.save(self.sess, save_file)
