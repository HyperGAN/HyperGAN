from hypergan.samplers.common import *
import tensorflow as tf


class StaticBatchSampler:
    pass
#mask_noise = None
z = None
y = None
x = None
def sample(gan, sample_file):
    sess = gan.sess
    config = gan.config
    global z, y, x
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z[0] # TODO support multiple z
    x_t = gan.graph.x


    if z is None:
        z = sess.run(z_t)
        y = sess.run(y_t)
        x = sess.run(x_t)


    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        sample = sess.run(generator, feed_dict={z_t: z, y_t: y, x_t: x})
        mean = sess.run(tf.reduce_mean(gan.graph.z_encoded), feed_dict={z_t: z})
        width = min(gan.config.batch_size, 8)
        stacks = [np.hstack(sample[i*width:i*width+width]) for i in range(gan.config.batch_size//width)]
        plot(config, np.vstack(stacks), sample_file)

    return [{'image':sample_file, 'label':'grid'}]
