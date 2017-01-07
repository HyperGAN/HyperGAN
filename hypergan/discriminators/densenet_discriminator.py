import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    batch_norm = config['discriminator.regularizers.layer']
    net = x
    net = conv2d(net, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      #if batch_norm is not None:
      #    net = batch_norm(config['batch_size']*2, name='d_expand_bn_'+str(i))(net)
      #net = activation(net)
      # APPEND xs[i] and gs[i]
      if(i < len(xs) and i > 0):
        xg = tf.concat(0, [xs[i], gs[i]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev']*(i+1), dtype=config['dtype'])

        xgs.append(xg)
  
        s = [int(x) for x in xg.get_shape()]

        #net = tf.concat(3, [net, xg])
      filter_size_w = 2
      filter_size_h = 2
      if i == depth-1:
          filter_size_w = int(net.get_shape()[1])
          filter_size_h = int(net.get_shape()[2])
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      length = 2
      size_dense = 16
      #if i==0:
      #    length = 1
      for j in range(length):
          print( j)
          net_dense = net
          if i > 0:
              net_dense = activation(net_dense)
              if batch_norm is not None:
                net_dense = batch_norm(config['batch_size']*2, name='d_expand_bna_'+str(i*10+j))(net_dense)
          newnet = conv2d(net_dense, size_dense, name='d_expand_layear'+str(i*10+j), k_w=3, k_h=3, d_h=1, d_w=1)
          net = tf.concat(3, [net, newnet])

      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      net = activation(net)
      if batch_norm is not None:
        net = batch_norm(config['batch_size']*2, name='d_expand_b2na_'+str(i*10+j))(net)
      net = conv2d(net, int(net.get_shape()[3]), name='d_expand_la4year'+str(i*10+j), k_w=1, k_h=1, d_h=1, d_w=1)

      print('[discriminator] layer', net)

    k=-1
    net = activation(net)
    if batch_norm is not None:
        net = batch_norm(config['batch_size']*2, name='d_expand_bn_end_'+str(i))(net)
    net = tf.reshape(net, [batch_size, -1])
 
    return net


