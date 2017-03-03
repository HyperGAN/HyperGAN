import tensorflow as tf
from hypergan.util.ops import *

def config():
    selector = hc.Selector()
    selector.set('create', discriminator)
    selector.set('kernels', 20)
    selector.set('kernel_dims', 200)

    return selector.random_config()

def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]


    net = tf.concat(axis=0, values=[d_real, d_fake])

    n_kernels = int(config.kernels)
    dim_per_kernel = int(config.kernel_dims)
    minis= get_minibatch_features(config, net, config.batch_size*2,config.dtype,prefix, n_kernels, dims_per_kernel)
    return minis

# This is openai's implementation of minibatch regularization
def get_minibatch_features(h, batch_size,dtype, prefix, n_kernels, dims_per_kernel):
    single_batch_size = batch_size//2
    print("[discriminator] minibatch from", h, "to", n_kernels*dim_per_kernel)
    x = linear(h, n_kernels * dim_per_kernel, scope=prefix+"h")
    activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

    big = np.zeros((batch_size, batch_size))
    big += np.eye(batch_size)
    big = tf.expand_dims(big, 1)
    big = tf.cast(big,dtype=dtype)

    abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation,3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
    mask = 1. - big
    masked = tf.exp(-abs_dif) * mask
    def half(tens, second):
        m, n, _ = tens.get_shape()
        m = int(m)
        n = int(n)
        return tf.slice(tens, [0, 0, second * single_batch_size], [m, n, single_batch_size])

    # TODO: speedup by allocating the denominator directly instead of constructing it by sum
    #       (current version makes it easier to play with the mask and not need to rederive
    #        the denominator)
    f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
    f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

    return [f1, f2]
