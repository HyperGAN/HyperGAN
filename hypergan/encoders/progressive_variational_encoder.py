import tensorflow as tf

from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def encode(config, x,y):
    x_dims = config['x_dims']
    batch_size = config["batch_size"]
    noise_dims = int(x.get_shape()[1])-int(y.get_shape()[1])
    n_z = int(config['generator.z'])
    channels = (config['channels']+1)
    activation = config['encoder.activation']
    batch_norm = config['generator.regularizers.layer']
    net = tf.image.resize_images(x, [32, 32], 1)
    net += tf.random_normal(net.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])
    net = conv2d(net, 16, name='v_expand', k_w=3, k_h=3, d_h=1, d_w=1)
    depth_increase = 2
    depth = 3
    for i in range(depth):
      net = batch_norm(config['batch_size'], name='v_expanbn_'+str(i))(net)
      net = activation(net)
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name='v_expanlayer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]

      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')
    net = tf.reshape(net, [config['batch_size'], -1])

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    out_mean= tf.get_variable('v_out_mean', [net.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    mu = tf.add(tf.matmul(net, out_mean),b_out_mean)

    out_log_sigma=tf.get_variable('v_out_logsigma', [net.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    b_out_log_sigma= tf.get_variable('v_b_out_logsigma', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    sigma = tf.add(tf.matmul(net, out_log_sigma),b_out_log_sigma)

    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=config['dtype'])
    set_tensor('eps', eps)

    z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(sigma)), eps))

    e_z = tf.random_normal([config['batch_size'], n_z], mu, tf.exp(sigma), dtype=config['dtype'])

    z = tf.nn.tanh(z)
    e_z = tf.nn.tanh(e_z)
    return e_z, z, mu, sigma

