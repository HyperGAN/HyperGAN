import tensorflow as tf
import numpy as np
from hypergan.util.globals import *
from .common import *

def initialize(config, d_vars, g_vars):
    d_loss = gan.graph.d_loss
    g_loss = gan.graph.g_loss
    g_lr = np.float32(config['trainer.sgd_adam.generator.lr'])
    d_lr = np.float32(config['trainer.sgd_adam.discriminator.lr'])
    g_optimizer = capped_optimizer(tf.train.AdamOptimizer, g_lr, g_loss, g_vars)
    d_optimizer = tf.train.GradientDescentOptimizer(d_lr).minimize(d_loss, var_list=d_vars)
    return g_optimizer, d_optimizer

iteration = 0
def train(gan):
    sess = gan.sess
    config = gan.config
    x_t = gan.graph.x
    g_t = gan.graph.g
    g_loss = gan.graph.g_loss_sig
    d_loss = gan.graph.d_loss
    d_fake_loss = gan.graph.d_fake_loss
    d_real_loss = gan.graph.d_real_loss
    g_optimizer = gan.graph.g_optimizer
    d_optimizer = gan.graph.d_optimizer
    d_class_loss = gan.graph.d_class_loss
    g_class_loss = gan.graph.g_class_loss

    _, d_cost = sess.run([d_optimizer, d_loss])
    _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
    print("%2d: g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, g_cost,d_fake, d_real, d_class ))

    global iteration
    iteration+=1

    return d_cost, g_cost


