
from hypergan.util.ops import *

from hypergan.samplers.common import *

#mask_noise = None
z = None
y = None
def sample(gan, sample_file):
    sess = gan.sess
    config = gan.config
    global z, y
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z[0] # TODO support multiple z

    x = np.linspace(0,1, 4)

    if z is None:
        z = sess.run(z_t)
        y = sess.run(y_t)


    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        class_mappings = gan.graph.class_mappings[0](gan.graph.x)
        sample, sample_x, sample_g = sess.run([class_mappings,gan.graph.x,generator], feed_dict={z_t: z, y_t: y})
        #plot(self.config, sample, sample_file)
        stacks1 = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        stacks2 = [np.hstack(sample_x[x*8:x*8+8]) for x in range(4)]
        stacks3 = [np.hstack(sample_g[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks2), sample_file)
        plot(config, np.vstack(stacks1), sample_file+'2.png')
        plot(config, np.vstack(stacks3), sample_file+'3.png')

    return [{'image':sample_file, 'label':'grid'}, 
            {'image':sample_file+'2.png', 'label':'grid'},
            {'image':sample_file+'3.png', 'label':'grid'}
            ]
