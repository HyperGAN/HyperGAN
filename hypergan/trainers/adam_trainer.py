import tensorflow as tf
import numpy as np
from hypergan.util.globals import *

def initialize(config, d_vars, g_vars):
    d_loss = get_tensor('d_loss')
    g_loss = get_tensor('g_loss')
    g_lr = np.float32(config['trainer.adam.generator.lr'])
    d_lr = np.float32(config['trainer.adam.discriminator.lr'])
    d_beta1 = np.float32(config['trainer.adam.discriminator.beta1'])
    d_beta2 = np.float32(config['trainer.adam.discriminator.beta2'])
    d_epsilon = np.float32(config['trainer.adam.discriminator.epsilon'])
    g_beta1 = np.float32(config['trainer.adam.generator.beta1'])
    g_beta2 = np.float32(config['trainer.adam.generator.beta2'])
    g_epsilon = np.float32(config['trainer.adam.generator.epsilon'])
    #g_optimizer = tf.train.AdamOptimizer(g_lr, beta1=g_beta1, beta2=g_beta2, epsilon=g_epsilon).minimize(g_loss, var_list=g_vars)
    g_optimizer = capped_optimizer(tf.train.AdamOptimizer, g_lr, g_loss, g_vars)
    #d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=d_beta1, beta2=d_beta2, epsilon=d_epsilon).minimize(d_loss, var_list=d_vars)
    d_optimizer = capped_optimizer(tf.train.AdamOptimizer, d_lr, d_loss, d_vars)
    return g_optimizer, d_optimizer

iteration = 0
def train(sess, config):
    x_t = get_tensor('x')
    g_t = get_tensor('g')
    g_loss = get_tensor("g_loss_sig")
    d_loss = get_tensor("d_loss")
    d_fake_loss = get_tensor('d_fake_loss')
    d_real_loss = get_tensor('d_real_loss')
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")
    d_class_loss = get_tensor("d_class_loss")
    g_class_loss = get_tensor("g_class_loss")

    _, d_cost = sess.run([d_optimizer, d_loss])
    _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
    print("%2d: g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, g_cost,d_fake, d_real, d_class ))

    global iteration
    iteration+=1

    return d_cost, g_cost


