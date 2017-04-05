import tensorflow as tf
import hyperchamber as hc

def D(gan, net, reuse=None):
    for i, discriminator in enumerate(gan.config.discriminators):
        discriminator = hc.Config(hc.lookup_functions(discriminator))
        with(tf.variable_scope("discriminator", reuse=reuse)):
            ds = discriminator.create(gan, discriminator, net, net, gan.graph.xs, gan.graph.gs,prefix="d_"+str(i))
            bs = gan.config.batch_size
            net = ds
            net = tf.slice(net, [0,0],[bs, -1])
            print('net is', net)
            return net

def gradient_penalty(gan, gradient_penalty):
    x = gan.graph.x
    g = gan.graph.gs[0]
    shape = [1 for t in g.get_shape()]
    shape[0] = gan.config.batch_size
    uniform_noise = tf.random_uniform(shape=shape,minval=0.,maxval=1.)
    interpolates = x + (1 - uniform_noise)*g
    gradients = tf.gradients(D(gan, interpolates), [interpolates])[0]
    penalty = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    penalty = tf.reduce_mean(tf.square(penalty-1.))
    return float(gradient_penalty) * penalty

