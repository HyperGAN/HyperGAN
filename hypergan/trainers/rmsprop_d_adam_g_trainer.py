import tensorflow as tf
import numpy as np
from .common import *

def config():
    selector = hc.Selector()
    selector.set('create', create)
    selector.set('run', run)

    selector.set("d_learn_rate", 1e-3)
    selector.set("g_learn_rate", 1e-3)

    selector.set("clipped_discriminator", False)
    return selector.random_config()

def create(config, gan, d_vars, g_vars):
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr = np.float32(config.g_learn_rate)
    d_lr = np.float32(config.d_learn_rate)
    gan.graph.d_vars = d_vars
    g_optimizer = tf.train.AdamOptimizer(g_lr).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.RMSPropOptimizer(d_lr).minimize(d_loss, var_list=d_vars)
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
    if config.clipped_discriminator:
        clip = [tf.assign(d,tf.clip_by_value(d, -config.clipped_discriminator, config.clipped_discriminator))  for d in d_vars]
        sess.run(clip)

    if(d_class_loss is not None):
        _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
        #print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_class %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_class, d_log ))
    else:
        _, g_cost,d_fake,d_real = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss])
        #print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_log %.2f" % (iteration, g_cost,d_cost, d_real, d_log ))
    global iteration
    iteration+=1

    return d_cost, g_cost 

