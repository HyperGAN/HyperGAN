import tensorflow as tf
from lib.util.globals import *

def initialize(config, d_vars, g_vars):
    joint_loss = get_tensor('joint_loss')
    g_lr = np.float32(config['trainer.adam.generator.lr'])
    d_lr = np.float32(config['trainer.slowdown.discriminator.lr'])
    set_tensor("lr_value", d_lr)
    d_lr = tf.get_variable('lr', [], trainable=False, initializer=tf.constant_initializer(d_lr,dtype=config['dtype']),dtype=config['dtype'])
    set_tensor("lr", d_lr)

    g_optimizer = tf.train.AdamOptimizer(g_lr).minimize(joint_loss, var_list=g_vars+d_vars)
    return g_optimizer, None#d_optimizer

iteration = 0
def train(sess, config):
    joint_loss = get_tensor('joint_loss')
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
    lr = get_tensor("lr")

    d_lr = 1.4e-5
    if iteration % 2 == 0:
        lr_value = 0.1*d_lr
    else:
        lr_value = d_lr

    #_, d_cost = sess.run([d_optimizer, d_loss], feed_dict={lr:lr_value})
    _, d_cost,g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, d_loss,joint_loss, d_fake_loss, d_real_loss, d_class_loss], feed_dict={lr:lr_value})
    print("%2d: d_lr %.1e g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, lr_value, g_cost,d_fake, d_real, d_class ))

    global iteration
    iteration+=1

    return d_cost, g_cost


