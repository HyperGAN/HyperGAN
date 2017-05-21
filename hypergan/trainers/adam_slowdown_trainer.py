import tensorflow as tf
import numpy as np
import hyperchamber as hc
from .common import *

def config(g_momentum=0, d_momentum=0, g_decay=0.999, d_decay=0.999, 
        d_learn_rate=1e-4, g_learn_rate=1e-4, clipped_gradients=False,
        clipped_d_weights=False):
    selector = hc.Selector()
    selector.set('create', create)
    selector.set('run', run)

    selector.set('g_momentum', g_momentum)
    selector.set('d_momentum', d_momentum)
    selector.set('g_decay', g_decay)
    selector.set('d_decay', d_decay)
    selector.set('clipped_gradients', clipped_gradients)
    selector.set("discriminator_learn_rate", d_learn_rate)
    selector.set("generator_learn_rate", g_learn_rate)

    selector.set("clipped_d_weights", clipped_d_weights)
    return selector.random_config()

def create(config, gan, d_vars, g_vars):
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr =tf.Variable(config.generator_learn_rate, trainable=False)
    d_lr =tf.Variable(config.discriminator_learn_rate, trainable=False)
    gan.graph.d_vars = d_vars

    g_decay = tf.Variable(config.g_decay, trainable=False)
    d_decay = tf.Variable(config.d_decay, trainable=False)
    g_momentum = tf.Variable(config.g_momentum, trainable=False)
    d_momentum = tf.Variable(config.d_momentum, trainable=False)
    g_beta1 = tf.Variable(0.01, trainable=False)
    d_beta1 = tf.Variable(0.01, trainable=False)
    g_beta2 = tf.Variable(0.999, trainable=False)
    d_beta2 = tf.Variable(0.999, trainable=False)
    g_epsilon = tf.Variable(1e-8, trainable=False)
    d_epsilon = tf.Variable(1e-8, trainable=False)
    g_optimizer = tf.train.AdamOptimizer(g_lr, beta1=g_beta1, beta2=g_beta2,epsilon=g_epsilon)
    d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=d_beta1, beta2=d_beta2,epsilon=d_epsilon)
    if config.clipped_d_weights:
        gan.graph.clip = [tf.assign(d,tf.clip_by_value(d, -config.clipped_d_weights, config.clipped_d_weights))  for d in d_vars]
    if(config.clipped_gradients):
        g_optimizer = capped_optimizer(g_optimizer, config.clipped_gradients, g_loss, g_vars)
        d_optimizer = capped_optimizer(d_optimizer, config.clipped_gradients, d_loss, d_vars)
    else:
        g_optimizer = g_optimizer.minimize(g_loss, var_list=g_vars)
        d_optimizer = d_optimizer.minimize(d_loss, var_list=d_vars)

    multiplier = 1.0000001
    assigns = []
    #assigns.append(tf.assign(d_decay, 1-(1-d_decay)/multiplier))
    #assigns.append(tf.assign(d_momentum, d_momentum/multiplier))
    #assigns.append(tf.assign(d_beta1, d_beta1/multiplier))
    #assigns.append(tf.assign(d_lr, d_lr/1.0001))

    #assigns.append(tf.assign(g_decay, 1-(1-g_decay)/multiplier))
    #assigns.append(tf.assign(g_momentum, g_momentum/multiplier))
    #assigns.append(tf.assign(g_lr, g_lr/1.0001))
    gan.graph.slowdown_step = tf.stack(assigns)
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
    clip = gan.graph.clip
    slowdown_step = gan.graph.slowdown_step

    _, d_cost, d_log = sess.run([d_optimizer, d_loss, d_log_t])
    if config.clipped_d_weights:
        sess.run(clip)

    global iteration
    if iteration % 100 == 0 and iteration > 0:
        sess.run(slowdown_step)
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


