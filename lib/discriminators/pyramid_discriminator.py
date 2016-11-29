import tensorflow as tf
from lib.util.ops import *
from lib.util.globals import *
from lib.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    result = x
    result = conv2d(result, 64, name='d_expand', k_w=3, k_h=3, d_h=2, d_w=2)
    batch_norm = config['generator.regularizers.layer']

    xgs = []
    xgs_conv = []
    for i in range(depth):
      print('--',i,result)
      result = batch_norm(config['batch_size'], name='d_expand_bn_'+str(i))(result)
      result = activation(result)
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1 and i > 0):
        xg = tf.concat(0, [xs[i+1], gs[i+1]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])

        xgs.append(xg)

        mxg = conv2d(xg, 6*(i), name="d_add_xg"+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
        mxg = batch_norm(config['batch_size'], name='d_add_xg_bn_'+str(i))(mxg)
        mxg = activation(mxg)

        xgs_conv.append(mxg)
  
        result = tf.concat(3, [result, xg])

      fltr=3
      result = conv2d(result, int(int(result.get_shape()[3])*depth_increase), name='d_expand_layer'+str(i), k_w=fltr, k_h=fltr, d_h=2, d_w=2)

      print('Discriminator pyramid layer:', result)

    filter_size_w = 2
    filter_size_h = 2
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')

    result = batch_norm(config['batch_size'], name='d_expand_bn_end_'+str(i))(result)
    result = activation(result)

    return result


