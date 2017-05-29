import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc
from hypergan.generators.resize_conv_generator import minmaxzero
from hypergan.losses.common import *

def config(
        include_recdistance=False,
        include_recdistance2=False,
        include_grecdistance=False,
        include_grecdistance2=False,
        include_distance=False,
        ):
    selector = hc.Selector()
    selector.set('create', create)

    if include_recdistance:
        selector.set('include_recdistance', True)
    if include_recdistance2:
        selector.set('include_recdistance2', True)
    if include_grecdistance:
        selector.set('include_grecdistance', True)
    if include_grecdistance2:
        selector.set('include_grecdistance2', True)
    if include_distance:
        selector.set('include_distance', True)
    return selector.random_config()


def dist(x1, x2):
    bs = int(x1.get_shape()[0])
    return tf.reshape(tf.abs(x1 - x2), [bs, -1])

def create(config, gan):
    x = gan.graph.x
    g_losses = []

    if 'include_recdistance' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.rxabba, gan.graph.rxa),
            dist(gan.graph.rxbaab, gan.graph.rxb)
            ])
        reconstruction *= config.alignment_lambda
        g_losses.append(tf.reduce_mean(reconstruction))

    if 'include_recdistance2' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.rxabba, gan.graph.xa),
            dist(gan.graph.rxbaab, gan.graph.xb)
            ])
        reconstruction *= config.alignment_lambda
        g_lossses.append(tf.reduce_mean(reconstruction))


    if 'include_grecdistance' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.rgabba, gan.graph.rga),
            dist(gan.graph.rgbaab, gan.graph.rgb)
            ])
        reconstruction *= config.alignment_lambda
        g_loss.append(tf.reduce_mean(reconstruction))

    if 'include_grecdistance2' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.rgabba, gan.graph.ga),
            dist(gan.graph.rgbaab, gan.graph.gb)
            ])
        reconstruction *= config.alignment_lambda
        g_losses.append(tf.reduce_mean(reconstruction))


    if 'include_distance' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.xabba, gan.graph.xa),
            dist(gan.graph.xbaab, gan.graph.xb)
            ])
        reconstruction *= config.alignment_lambda
        g_losses.append(tf.reduce_mean(reconstruction))
        print("- - - -- - Reconstruction loss added.")


    g_loss = tf.add_n(g_losses)
    return [None, g_loss]
