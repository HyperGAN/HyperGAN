from hypergan.samplers.common import *

def sample_tensor(sess,generator, feed_dict, sample_file):
    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        sample = sess.run(generator, feed_dict=feed_dict)
        #plot(self.config, sample, sample_file)
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks), sample_file)



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

    if x is None:
        x = gan.sess.run(x_t)

    if z is None:
        z = sess.run(z_t)
        y = sess.run(y_t)


    x_file = sample_file+'x.png'
    autoencoded_x_file = sample_file+'autox.png'
    autoencoded_g_file = sample_file+'autog.png'
    feed_dict = {z_t: z, y_t: y, x_t: x}
    sample_tensor(sess,generator, feed_dict, sample_file)
    sample_tensor(sess,x_t, feed_dict, x_file)
    sample_tensor(sess,gan.graph.dx, feed_dict, autoencoded_x_file)
    sample_tensor(sess,gan.graph.dg, feed_dict, autoencoded_g_file)
    samples = []
    samples.append({'image':sample_file, 'label':'g'})
    samples.append({'image':x_file, 'label':'x'})
    samples.append({'image':autoencoded_g_file, 'label':'autoencoded_g'})
    samples.append({'image':autoencoded_x_file, 'label':'autoencoded_x'})

    return samples
