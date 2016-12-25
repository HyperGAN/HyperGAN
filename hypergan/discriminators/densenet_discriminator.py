import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    layers = config['discriminator.densenet.layers']
    transitions = config['discriminator.densenet.transitions']
    k = config['discriminator.densenet.k']
    activation = config['discriminator.activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['discriminator.pyramid.depth_increase']
    depth = config['discriminator.pyramid.layers']
    batch_norm = config['generator.regularizers.layer']

    result = x
    result = conv2d(result, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)
    xgs = []
    xgs_conv = []
    for i in range(transitions):
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1):
        xg = tf.concat(0, [xs[i], gs[i]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])

        xgs.append(xg)

        mxg = conv2d(xg, 6*(i), name="d_add_xg"+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
        mxg = batch_norm(config['batch_size'], name='d_add_xg_bn_'+str(i))(mxg)
        mxg = activation(mxg)

        xgs_conv.append(mxg)
  
        result = tf.concat(3, [result, xg])
      for j in range(layers):
        result = dense_block(result, k, activation, batch_size, 'layer', 'd_layers_'+str(i)+"_"+str(j))
        print("densenet size", result)
      result = dense_block(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i))


    set_tensor("xgs", xgs)
    set_tensor("xgs_conv", xgs_conv)

    result = batch_norm(config['batch_size'], name='d_expand_bn_end_'+str(i))(result)
    result = activation(result)

    return result



