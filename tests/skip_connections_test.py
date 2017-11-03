import hypergan as hg
import tensorflow as tf
from hypergan.skip_connections import SkipConnections

class SkipConnectionsTest(tf.test.TestCase):
    def test_get_none(self):
        sc = SkipConnections()
        self.assertIsNone(sc.get('none'))

    def test_get(self):
        sc = SkipConnections()
        a = tf.zeros([1,2,3])
        b = tf.zeros([3,2,1])

        sc.set('layer_filter', a)
        sc.set('layer_filter', b)
        self.assertEqual(a, sc.get('layer_filter', a.get_shape()))
        self.assertEqual(b, sc.get('layer_filter', b.get_shape()))

    def test_get_closest(self):
        sc = SkipConnections()
        b = tf.zeros([1,2,2])
        a = tf.zeros([1,4,4])

        sc.set('layer_filter', b)
        sc.set('layer_filter', a)
        self.assertEqual(a, sc.get_closest('layer_filter', a.get_shape()))
        self.assertEqual(b, sc.get_closest('layer_filter', b.get_shape()))
        self.assertEqual(a, sc.get_closest('layer_filter', [1,3,3]))
        self.assertEqual(b, sc.get_closest('layer_filter', [1,1,1]))
        self.assertEqual(None, sc.get_closest('layer_filter', [1,5,5]))


    def test_array(self):
        sc = SkipConnections()
        a = tf.zeros([1,2,3])
        b = tf.zeros([3,2,1])

        self.assertEqual([], sc.get_array('layer_filter', a.get_shape()))
        sc.set('layer_filter', a)
        self.assertEqual([a], sc.get_array('layer_filter', a.get_shape()))
        sc.set('layer_filter', a)
        self.assertEqual([a, a], sc.get_array('layer_filter', a.get_shape()))

    def test_get_shapes(self):
        sc = SkipConnections()
        a = tf.zeros([1,2,3])
        b = tf.zeros([3,2,1])

        sc.set('layer_filter', a)
        sc.set('layer_filter', b)
        shapes = sc.get_shapes('layer_filter')
        print("SHAPES", shapes)
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], [1,2,3])
        self.assertEqual(shapes[1], [3,2,1])
        self.assertEqual(sc.get_shapes('non-existant'), None)

    def test_clear(self):
        sc = SkipConnections()
        a = tf.zeros([1,2,3])
        b = tf.zeros([3,2,1])

        sc.set('layer_filter', a)
        sc.set('layer_filter', b)
        sc.clear('layer_filter', shape=a.get_shape())
        print("C?LAEAR", sc.connections)
        self.assertEqual(None, sc.get('layer_filter', a.get_shape()))
        self.assertEqual([], sc.get_array('layer_filter', a.get_shape()))
        self.assertEqual(b, sc.get('layer_filter', b.get_shape()))


if __name__ == "__main__":
    tf.test.main()
