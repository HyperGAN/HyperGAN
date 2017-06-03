import tensorflow as tf
from hypergan.ops.tensorflow.ops import TensorflowOps

from unittest.mock import MagicMock

ops = TensorflowOps()
tanh_str = "function:tensorflow.python.ops.math_ops.tanh"
class OpsTest(tf.test.TestCase):
    def test_lookup_function(self):
        with self.test_session():
            self.assertEqual(ops.lookup_function(tanh_str), tf.nn.tanh)

    def test_lookup_class(self):
        with self.test_session():
            self.assertEqual(ops.lookup_class('tf.python.training.rmsprop.RMSPropOptimizer'), tf.python.training.rmsprop.RMSPropOptimizer)

    def test_lookup(self):
        with self.test_session():
            self.assertEqual(ops.lookup('tanh'), tf.nn.tanh)
            self.assertEqual(ops.lookup(tanh_str), tf.nn.tanh)
            self.assertEqual(ops.lookup(None), None)

    def test_dtype(self):
        with self.test_session():
            self.assertEqual(ops.parse_dtype('float32'), tf.float32)

    def test_shape(self):
        with self.test_session():
            self.assertEqual(ops.shape(tf.constant(1)), [])

    def test_slice(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.slice(tf.constant(1,shape=[1]), [0],[1])), [1])

    def test_resize(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.resize_images(tf.constant(1, shape=[1,1,1]), [2,2], 1))[1], 2)

    def test_concat(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.concat([tf.constant(1)])), [])

    def test_reshape(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.reshape(tf.constant(1), [1])), [1])

    def test_linear(self):
        with self.test_session():
            net = ops.linear(tf.constant(1., shape=[1, 3]), 3)
            self.assertEqual(ops.shape(net)[1], 3)

    def test_conv2d(self):
        with self.test_session():
            net = ops.conv2d(tf.constant(1., shape=[1, 3, 3, 3]), 3, 3, 1, 1, 3)
            self.assertEqual(ops.shape(net)[1], 3)

    def test_deconv2d(self):
        with self.test_session():
            net = ops.deconv2d(tf.constant(1., shape=[1, 3, 3, 3]), 3, 3, 1, 1, 3)
            self.assertEqual(ops.shape(net)[1], 3)

    def test_generate_scope(self):
        with self.test_session():
            ops = TensorflowOps()
            self.assertEqual(ops.generate_scope(), "1")
            self.assertEqual(ops.generate_scope(), "2")

    def test_generate_name(self):
        with self.test_session():
            ops = TensorflowOps()
            ops.describe("generator")
            self.assertEqual(ops.generate_name(), "generator_1")
            self.assertEqual(ops.generate_name(), "generator_2")

    def test_variable_constructor(self):
        with self.test_session():
            ops = TensorflowOps()
            self.assertEqual(len(ops.weights), 0)
            self.assertEqual(len(ops.biases), 0)
            self.assertEqual(len(ops.variables()), 0)

    def test_get_weight(self):
        with self.test_session():
            ops = TensorflowOps({'dtype':'float32'})
            ops.get_weight([1,1])
            self.assertTrue('float32' in str(ops.weights[0].dtype))
            self.assertEqual(ops.shape(ops.weights[0]), [1, 1])

    def test_get_bias(self):
        with self.test_session():
            ops = TensorflowOps({'dtype':'float32'})
            ops.get_bias([1,1])
            self.assertTrue('float32' in str(ops.biases[0].dtype))
            self.assertEqual(ops.shape(ops.biases[0]), [1, 1])

if __name__ == "__main__":
    tf.test.main()
