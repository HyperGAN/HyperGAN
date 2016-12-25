import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    net = conv2d(x, 64, name='d_expand', k_w=3, k_h=3, d_h=2, d_w=2)
    batch_norm = config['generator.regularizers.layer']

    xgs = []
    xgs_conv = []
    for i in range(depth):
      print('--',i,net)
      net = batch_norm(config['batch_size']*2, name='d_expand_bn_'+str(i))(net)
      net = activation(net)
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1 and i > 0):
        xg = tf.concat(0, [xs[i+1], gs[i+1]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])

        xgs.append(xg)

        mxg = conv2d(xg, 6*(i), name="d_add_xg"+str(i), k_w=3, k_h=3, d_h=2, d_w=2)
        mxg = batch_norm(config['batch_size']*2, name='d_add_xg_bn_'+str(i))(mxg)
        mxg = activation(mxg)

        xgs_conv.append(mxg)
  
        net = tf.concat(3, [net, xg])

      fltr=3
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name='d_expand_layer'+str(i), k_w=fltr, k_h=fltr, d_h=2, d_w=2)

      print('Discriminator pyramid layer:', net)

    net = tf.reshape(net, [config['batch_size']*2, -1])
    net = batch_norm(config['batch_size']*2, name='d_expand_bn_end_'+str(i))(net)
    net = activation(net)
    net = linear(net, int(1024), scope="d_fc_end1")
    net = batch_norm(config['batch_size']*2, name='d_bn_end1')(net)
    net = activation(net)

    return net


