import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config():
    selector = hc.Selector()
    selector.set("reduce", [tf.reduce_mean])#reduce_sum, reduce_logexp work

    selector.set('create', create)
    selector.set('batch_norm', layer_norm_1)

    return selector.random_config()

def create(config, gan):
    batch_norm = config.batch_norm
    batch_size = gan.config.batch_size

    num_classes = gan.config.y_dims
    net = gan.graph.d_real
    net = linear(net, num_classes, scope="d_fc_end", stddev=0.003)
    net = batch_norm(batch_size, name='d_bn_end')(net)

    d_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net,labels=gan.graph.y)

    gan.graph.d_class_loss=tf.reduce_mean(d_class_loss)

    return [d_class_loss, None]

