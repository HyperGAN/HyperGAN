import tensorflow as tf
import numpy as np
import hyperchamber as hc
from .common import *

def config(
        d_learn_rate=1e-3,
        g_learn_rate=1e-3,
        d_clipped_weights=False,
        clipped_gradients=False
    ):
    selector = hc.Selector()

    selector.set('create', create)
    selector.set('run', run)

    selector.set('d_learn_rate', d_learn_rate)
    selector.set('g_learn_rate', g_learn_rate)
    selector.set('clipped_gradients', clipped_gradients)
    selector.set('d_clipped_weights', d_clipped_weights)

    return selector.random_config()

def create(config, gan, d_vars, g_vars):
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr = np.float32(config.g_learn_rate)
    d_lr = np.float32(config.d_learn_rate)

    gan.graph.d_vars = d_vars
    g_optimizer = tf.train.GradientDescentOptimizer(g_lr)
    d_optimizer = tf.train.GradientDescentOptimizer(d_lr)

    if(config.clipped_gradients):
        g_optimizer = capped_optimizer(g_optimizer, config.clipped_gradients, g_loss, g_vars)
        d_optimizer = capped_optimizer(d_optimizer, config.clipped_gradients, d_loss, d_vars)
    else:
        g_optimizer = g_optimizer.minimize(g_loss, var_list=g_vars)
        d_optimizer = d_optimizer.minimize(d_loss, var_list=d_vars)

    gan.graph.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]

    return g_optimizer, d_optimizer

iteration = 0
def run(gan):
    sess = gan.sess
    config = gan.config
    x_t = gan.graph.x
    g_t = gan.graph.g
    d_log_t = gan.graph.d_log
    g_loss = gan.graph.g_loss
    d_loss = gan.graph.d_loss
    d_fake_loss = gan.graph.d_fake_loss
    d_real_loss = gan.graph.d_real_loss
    g_optimizer = gan.graph.g_optimizer
    d_optimizer = gan.graph.d_optimizer
    d_class_loss = gan.graph.d_class_loss
    d_vars = gan.graph.d_vars

    _, d_cost, d_log = sess.run([d_optimizer, d_loss, d_log_t])

    # in WGAN paper, values are clipped.  This might not work, and is slow.
    if(config.d_clipped_weights):
        sess.run(gan.graph.clip)

    if(d_class_loss is not None):
        _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
        #print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_class %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_class, d_log ))
    else:
        _, g_cost,d_fake,d_real = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss])
        #print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_log ))


    global iteration
    iteration+=1

    return d_cost, g_cost


