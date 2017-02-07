import tensorflow as tf
import numpy as np
from .common import *

def initialize(gan, d_vars, g_vars):
    config = gan.config
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr = np.float32(config['trainer.rmsprop.generator.lr'])
    d_lr = np.float32(config['trainer.rmsprop.discriminator.lr'])
    gan.graph.d_vars = d_vars
    g_optimizer = tf.train.RMSPropOptimizer(g_lr).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.RMSPropOptimizer(d_lr).minimize(d_loss, var_list=d_vars)
    return g_optimizer, d_optimizer

iteration = 0
def train(gan):
    sess = gan.sess
    config = gan.config
    x_t = gan.graph.x
    g_t = gan.graph.g
    g_loss = gan.graph.g_loss
    d_loss = gan.graph.d_loss
    d_fake_loss = gan.graph.d_fake_loss
    d_real_loss = gan.graph.d_real_loss
    g_optimizer = gan.graph.g_optimizer
    d_optimizer = gan.graph.d_optimizer
    d_class_loss = gan.graph.d_class_loss
    d_vars = gan.graph.d_vars

    _, d_cost = sess.run([d_optimizer, d_loss])
    #clip = [tf.assign(d,tf.clip_by_value(d, -0.1, 0.1))  for d in d_vars]
    #sess.run(clip)

    _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
    print("%2d: g cost %.2f d_loss %.2f d_real %.2f d_class %.2f" % (iteration, g_cost,d_cost, d_real, d_class ))

    global iteration
    iteration+=1

    return d_cost, g_cost


