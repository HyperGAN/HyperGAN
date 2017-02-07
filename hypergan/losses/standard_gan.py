import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config():
    selector = hc.Selector()
    selector.set("reduce", [tf.reduce_mean])#reduce_sum, reduce_logexp work
    selector.set("label_smooth", list(np.linspace(0.15, 0.35, num=100)))

    selector.set('create', create)

    return selector.random_config()

def create(config, gan):
    d_real = self.gan.graph.d_real
    d_fake = self.gan.graph.d_fake 

    d_real = config.reduce(d_real, axis=1)
    d_fake = config.reduce(d_fake, axis=1)

    zeros = tf.zeros_like(d_fake, dtype=gan.config.dtype)
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, zeros)
    d_loss = sigmoid_kl_with_logits(d_real_log, 1.-config.label_smooth)

    return [d_loss, g_loss]
