
from hypergan.util.ops import *
from hypergan.util.globals import *

#mask_noise = None
def sample(sample_file, sess, config):
    generator = get_tensor("g")[0]
    y_t = get_tensor("y")
    z_t = get_tensor("z")
    dropout_t = get_tensor("dropout")
    #mask_noise_t = get_tensor("mask_noise")
    #categories_t = get_tensor("categories")[0]

    x = np.linspace(0,1, 4)
    y = np.linspace(0,1, 6)

    #z = np.mgrid[-3:3:0.75, -3:3:0.38*3].reshape(2,-1).T
    #z = np.mgrid[-3:3:0.6*3, -3:3:0.38*3].reshape(2,-1).T
    #z = np.mgrid[-6:6:0.6*6, -6:6:0.38*6].reshape(2,-1).T

    #z = np.random.uniform(-1, 1, [config['batch_size'], 2])
    z = np.mgrid[-0.999:0.999:0.6, -0.999:0.999:0.26].reshape(2,-1).T
    #z = np.square(1/z) * np.sign(z)
    #z = np.mgrid[0:1000:300, 0:1000:190].reshape(2,-1).T
    #z = np.mgrid[-0:1:0.3, 0:1:0.19].reshape(2,-1).T
    #z = np.mgrid[0.25:-0.25:-0.15, 0.25:-0.25:-0.095].reshape(2,-1).T
    #z = np.mgrid[-0.125:0.125:0.075, -0.125:0.125:0.095/2].reshape(2,-1).T
    #z = np.zeros(z_t.get_shape())
    #z.fill(0.2)

    #categories = np.zeros(categories_t.get_onshape())
    #global mask_noise
    #if mask_noise is None:
    #    s=mask_noise_t.get_shape()
    #    mask_noise = np.random.uniform(0, 1, [1, s[1], s[2], s[3]])
    #    mask_noise = np.tile(mask_noise, [config['batch_size'], 1, 1, 1])
    #    #ask_noise = np.ones(mask_noise_t.get_shape())
    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(1)
        print("seed",g.seed)
        sample = sess.run(generator, feed_dict={z_t: z})#, categories_t: categories})
        print(np.shape(sample), np.min(sample), np.max(sample))
        #plot(self.config, sample, sample_file)
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(config, np.vstack(stacks), sample_file)


