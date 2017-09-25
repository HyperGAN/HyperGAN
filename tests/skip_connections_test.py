import hypergan as hg
import tensorflow as tf
from hypergan.skip_connections import SkipConnections

sc = SkipConnections()
class SkipConnectionsTest(tf.test.TestCase):
    def test_get_none(self):
        self.assertIsNone(sc.get('none'))

    def test_get(self):
        a = tf.zeros([1,2,3])
        b = tf.zeros([3,2,1])

        sc.set('layer_filter', a)
        sc.set('layer_filter', b)
        self.assertEqual(a, sc.get('layer_filter', a.get_shape()))
        self.assertEqual(b, sc.get('layer_filter', b.get_shape()))

if __name__ == "__main__":
    tf.test.main()
