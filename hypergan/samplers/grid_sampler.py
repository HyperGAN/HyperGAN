from hypergan.util.ops import *
from hypergan.samplers.common import *

def sample(gan, sample_file):
    sess = gan.sess
    config = gan.config
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z[0]

    x = np.linspace(0,1, 4)
    y = np.linspace(0,1, 6)

    z = np.mgrid[-0.999:0.999:0.6, -0.999:0.999:0.26].reshape(2,-1).T
    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        sample = sess.run(generator, feed_dict={z_t: z})#, categories_t: categories})
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks), sample_file)

    return [{'image':sample_file, 'label':'grid'}]
