
from hypergan.util.ops import *

from hypergan.samplers.common import *

z = None
def sample(gan, sample_file):
    sess = gan.sess
    config = gan.config
    global z
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z

    x = np.linspace(0,1, 4)
    y = np.linspace(0,1, 6)

    z = np.random.uniform(-1, 1, [config['batch_size'], int(z_t[0].get_shape()[1])])

    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        sample = sess.run(generator, feed_dict={z_t[0]: z})
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks), sample_file)

    return [{'image':sample_file, 'label':'grid'}]
