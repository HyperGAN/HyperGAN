
from hypergan.util.ops import *
from hypergan.util.globals import *

from hypergan.samplers.common import *

#mask_noise = None
z = None
def sample(sample_file, sess, config):
    global z
    generator = get_tensor("g")[0]
    y_t = get_tensor("y")
    z_t = get_tensor("z")
    #mask_noise_t = get_tensor("mask_noise")
    #categories_t = get_tensor("categories")[0]

    x = np.linspace(0,1, 4)
    y = np.linspace(0,1, 6)

    z = np.random.uniform(-1, 1, [config['batch_size'], config['generator.z']])

    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        sample = sess.run(generator, feed_dict={z_t: z})
        #plot(self.config, sample, sample_file)
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks), sample_file)

    return [{'image':sample_file, 'label':'grid'}]
