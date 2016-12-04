import tensorflow as tf
from lib.util.ops import *
from lib.util.globals import *
from lib.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    batch_norm = config['generator.regularizers.layer']
    batch_norm = batch_norm_1
    net = x
    net = conv2d(net, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      net = batch_norm(config['batch_size']*2, name='d_expand_bn_'+str(i))(net)
      net = activation(net)
      # APPEND xs[i] and gs[i]
      if(i < len(xs) and i > 0):
        xg = tf.concat(0, [xs[i], gs[i]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev']*i, dtype=config['dtype'])

        xgs.append(xg)

        mxg = conv2d(xg, 6*(i), name="d_add_xg"+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
        mxg = batch_norm(config['batch_size'], name='d_add_xg_bn_'+str(i))(mxg)
        mxg = activation(mxg)

        xgs_conv.append(mxg)
  
        net = tf.concat(3, [net, xg])

      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name='d_expand_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('Discriminator pyramid layer:', net)

    dropout = tf.Variable(0.5)
    k=-1
    net = batch_norm(config['batch_size']*2, name='d_expand_bn_end_'+str(i))(net)
    net = activation(net)
    net = tf.reshape(net, [batch_size, -1])
    net = linear(net, int(1024*1.5), scope="d_fc_end1")
    net = batch_norm(config['batch_size']*2, name='d_bn_end1')(net)
    net = activation(net)
    net = tf.nn.dropout(net, dropout)
    net = linear(net, 1024, scope="d_fc_end2")
    net = batch_norm(config['batch_size']*2, name='d_bn_end2')(net)
    net = activation(net)
    net = tf.nn.dropout(net, dropout)
 
    return net


