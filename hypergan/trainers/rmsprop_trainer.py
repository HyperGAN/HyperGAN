import tensorflow as tf
import numpy as np
import hyperchamber as hc
from .common import *

def config(g_momentum=0.01, 
           d_momentum=0.00001, 
           g_decay=0.999, 
           d_decay=0.995, 
           d_learn_rate=0.0005, 
           g_learn_rate=0.0004, 
           clipped_gradients=False,
           clipped_d_weights=0.01):
    selector = hc.Selector()
    selector.set('create', create)
    selector.set('run', run)

    selector.set('g_momentum', g_momentum)
    selector.set('d_momentum', d_momentum)
    selector.set('g_decay', g_decay)
    selector.set('d_decay', d_decay)
    selector.set('clipped_gradients', clipped_gradients)
    selector.set("d_learn_rate", d_learn_rate)
    selector.set("g_learn_rate", g_learn_rate)

    selector.set("clipped_d_weights", clipped_d_weights)
    return selector.random_config()

def create(config, gan, d_vars, g_vars):
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr = np.float32(config.g_learn_rate)
    d_lr = np.float32(config.d_learn_rate)
    gan.graph.d_vars = d_vars

    g_optimizer = tf.train.RMSPropOptimizer(g_lr, decay=config.g_decay, momentum=config.g_momentum)
    d_optimizer = tf.train.RMSPropOptimizer(d_lr, decay=config.d_decay, momentum=config.d_momentum)
    if(config.clipped_gradients):
        g_optimizer = capped_optimizer(g_optimizer, config.clipped_gradients, g_loss, g_vars)
        d_optimizer = capped_optimizer(d_optimizer, config.clipped_gradients, d_loss, d_vars)
    else:
        g_optimizer = g_optimizer.minimize(g_loss, var_list=g_vars)
        d_optimizer = d_optimizer.minimize(d_loss, var_list=d_vars)

    return g_optimizer, d_optimizer

iteration = 0
clip = None
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
    if config.clipped_d_weights:
        global clip
        if(clip == None):
            clip = [tf.assign(d,tf.clip_by_value(d, -config.clipped_d_weights, config.clipped_d_weights))  for d in d_vars]
        sess.run(clip)

    global iteration
    if(d_class_loss is not None):
        _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
        if iteration % 100 == 0:
            print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_class %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_class, d_log ))
    else:
        _, g_cost,d_fake,d_real = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss])
        if iteration % 100 == 0:
            print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_log ))

    iteration+=1

    return d_cost, g_cost


