
from hypergan.util.ops import *

from hypergan.samplers.common import *

def sample_tensor(sess,generator, feed_dict, sample_file):
    if isinstance(generator, list):
        generator = generator[0]
    if generator is None:
        return
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
xb = None
def sample(gan, sample_file):
    sess = gan.sess
    config = gan.config
    global z, y, x, xb
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z[0] # TODO support multiple z
    x_t = gan.graph.xa
    xb_t = gan.graph.xb
    gb_t = gan.graph.gb

    if x is None:
        x = gan.sess.run(x_t)
        xb = gan.sess.run(xb_t)

    if z is None:
        z = sess.run(z_t)
        y = sess.run(y_t)


    x_file = sample_file+'x.png'
    xb_file = sample_file+'xb.png'
    autoencoded_x_file = sample_file+'autox.png'
    autoencoded_g_file = sample_file+'autog.png'
    autoencoded_hx_file = sample_file+'autohx.png'
    autoencoded_hg_file = sample_file+'autohg.png'
    autoencoded_gb_file = sample_file+'autogg.png'
    autoencoded_xb_file = sample_file+'autoxb.png'
    ga_file = sample_file+'ga.png'
    gb_file = sample_file+'gb.png'
    feed_dict = {z_t: z, y_t: y, x_t: x, xb_t: xb}
    sample_tensor(sess,generator, feed_dict, sample_file)
    print(x_t)
    sample_tensor(sess,gan.graph.xa, feed_dict, x_file)
    sample_tensor(sess,gan.graph.xb, feed_dict, xb_file)
    sample_tensor(sess,gan.graph.ga, feed_dict, ga_file)
    sample_tensor(sess,gan.graph.gb, feed_dict, gb_file)
    sample_tensor(sess,gan.graph.gba, feed_dict, autoencoded_x_file)
    sample_tensor(sess,gan.graph.gab, feed_dict, autoencoded_hg_file)
    sample_tensor(sess,gan.graph.gabba, feed_dict, autoencoded_g_file)
    sample_tensor(sess,gan.graph.gbaab, feed_dict, autoencoded_hx_file)
    sample_tensor(sess,gan.graph.hx, feed_dict, autoencoded_gb_file)
    sample_tensor(sess,gan.graph.rxa, feed_dict, autoencoded_xb_file)
    samples = []
    samples.append({'image':x_file, 'label':'xa'})
    samples.append({'image':autoencoded_hg_file, 'label':'gab'})
    samples.append({'image':autoencoded_g_file, 'label':'gabba'})
    samples.append({'image':xb_file, 'label':'xb'})
    samples.append({'image':autoencoded_x_file, 'label':'gba'})
    samples.append({'image':autoencoded_hx_file, 'label':'gbaab'})
    samples.append({'image':ga_file, 'label':'ga'})
    samples.append({'image':gb_file, 'label':'gb'})

    return samples
