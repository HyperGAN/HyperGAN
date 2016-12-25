import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def discriminator(config, x, g, xs, gs):
    layers = config['discriminator.painters.layers']
    transitions = config['discriminator.painters.transitions']
    activation = config['discriminator.painters.activation']
    batch_size = int(x.get_shape()[0])
    batch_norm = config['generator.regularizers.layer']

    result = x
    result = conv2d(result, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)
    result = batch_norm(config['batch_size']*2, name='d_expand_bn')(result)
    result = activation(result)
    xgs = []
    xgs_conv = []
    for i in range(transitions):
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1):
        xg = tf.concat(0, [xs[i], gs[i]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])

        xgs.append(xg)

        result = tf.concat(3, [result, xg])
      for j in range(layers):
        result = conv2d(result, 2**(3+i), name="d_add_xg"+str(i)+"-"+str(j), k_w=3, k_h=3, d_h=1, d_w=1)
        result = batch_norm(config['batch_size']*2, name='d_add_xg_bn_'+str(i)+"-"+str(j))(result)
        result = activation(result)
        print("painters size", result)

      result = tf.nn.max_pool(result, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')


    set_tensor("xgs", xgs)
    set_tensor("xgs_conv", xgs_conv)

    result = batch_norm(config['batch_size']*2, name='d_expand_bn_end_'+str(i))(result)
    result = activation(result)
    print("painters size", result)
    result = tf.reshape(result, [config['batch_size']*2, -1])

    #result = tf.nn.dropout(result, 0.9)
    result = linear(result, 1024, scope="d_fc_end1")
    result = batch_norm(config['batch_size']*2, name='d_bn_end1')(result)
    result = activation(result)
    #result = linear(result, 1584, scope="d_fc_end2")
    #result = batch_norm(config['batch_size']*2, name='d_bn_end2')(result)
    #result = activation(result)
    #result = tf.reshape(result, [config['batch_size']*2, 1, 1, 1024])

    return result



