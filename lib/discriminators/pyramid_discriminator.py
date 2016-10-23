import tensorflow as tf
from lib.util.ops import *
from lib.util.globals import *

def discriminator(config, x, g, xs, gs):
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_densenet_layers']
    depth = 5
    k = config['d_densenet_k']
    result = x
    result = conv2d(result, 64, name='d_expand', k_w=3, k_h=3, d_h=2, d_w=2)

    xgs = []
    for i in range(depth):
      result = batch_norm(config['batch_size'], name='d_expand_bn_'+str(i))(result)
      result = activation(result)
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1):
        xg = tf.concat(0, [xs[i+1], gs[i+1]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['d_noise'], dtype=config['dtype'])
        xgs.append(xg)
  
        mxg = conv2d(xg, 6*(i+1), name="d_add_xg"+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
        mxg = batch_norm(config['batch_size'], name='d_add_xg_bn_'+str(i))(mxg)
        mxg = activation(mxg)
  
        minisx = tf.reduce_mean(xs[i+1], reduction_indices=0, keep_dims=True)
        minisg = tf.reduce_mean(gs[i+1], reduction_indices=0, keep_dims=True)
        minisx = tf.tile(minisx, [config['batch_size'], 1,1,1]) 
        minisg = tf.tile(minisg, [config['batch_size'], 1,1,1]) 
        minis = tf.concat(0, [minisx, minisg])
  
        result = tf.concat(3, [result, mxg, minis])

      result = conv2d(result, int(result.get_shape()[3])*2, name='d_expand_layer'+str(i), k_w=3, k_h=3, d_h=2, d_w=2)
      print('Discriminator pyramid layer:', result)

    set_tensor("xgs", xgs)

    result = batch_norm(config['batch_size'], name='d_expand_bn_end_'+str(i))(result)
    result = activation(result)

    return result


