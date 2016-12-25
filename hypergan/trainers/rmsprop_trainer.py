import tensorflow as tf
from hypergan.util.globals import *
TINY = 1e-12

def initialize(config, d_vars, g_vars):
    d_loss = get_tensor('d_loss')
    g_loss = get_tensor('g_loss')
    d_lr = np.float32(config['trainer.rmsprop.discriminator.lr'])
    g_lr = np.float32(config['trainer.rmsprop.generator.lr'])

    g_optimizer = tf.train.RMSPropOptimizer(g_lr).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.RMSPropOptimizer(d_lr).minimize(d_loss, var_list=d_vars)
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
    print("[rms]%2d: g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, g_cost,d_fake, d_real, d_class ))
    while(g_cost > 0.7):
      _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
      print("[rms]%2d: g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, g_cost,d_fake, d_real, d_class ))

    global iteration
    iteration+=1

    return d_cost, g_cost


