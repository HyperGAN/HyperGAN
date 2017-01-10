import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *
import hypergan.regularizers.minibatch_regularizer as minibatch_regularizer
import hyperchamber as hc

def config(resize=None, layers=None):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set('regularizer', [batch_norm_1]) # Size of fully connected layers

    if layers == None:
        layers = [5]
    selector.set("layers", layers) #Layers in D
    selector.set("dense.layers", 3) #Layers in D
    selector.set("dense.size", 24) #Layers in D

    selector.set('add_noise', [True]) #add noise to input
    selector.set('noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
    selector.set('regularizers', [[minibatch_regularizer.get_features]]) # these regularizers get applied at the end of D
    selector.set('resize', [resize])

    selector.set('create', discriminator)

    return selector.random_config()


def discriminator(root_config, config, x, g, xs, gs, prefix='d_'):
    activation = config['activation']
    batch_size = int(x.get_shape()[0])
    depth = config['layers']
    batch_norm = config['regularizer']
    length = config['dense.layers']
    size_dense = config['dense.size']

    if(config['resize']):
        # shave off layers >= resize 
        def should_ignore_layer(layer, resize):
            return int(layer.get_shape()[1]) > config['resize'][0] or \
                   int(layer.get_shape()[2]) > config['resize'][1]

        xs = [px for px in xs if not should_ignore_layer(px, config['resize'])]
        gs = [pg for pg in gs if not should_ignore_layer(pg, config['resize'])]

        x = tf.image.resize_images(x,config['resize'], 1)
        g = tf.image.resize_images(g,config['resize'], 1)

        print("X XSXS SX", x.get_shape(), g.get_shape(), xs, config['resize'])

    net = tf.concat(0, [x,g])
    if(config['add_noise']):
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise_stddev'], dtype=root_config['dtype'])
    net = conv2d(net, 16, name=prefix+'_expand', k_w=3, k_h=3, d_h=1, d_w=1)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      if batch_norm is not None:
          net = batch_norm(batch_size*2, name=prefix+'_expand_bn_'+str(i))(net)
      net = activation(net)
      # APPEND xs[i] and gs[i]
      #if(i < len(xs) and i > 0):
      #  xg = tf.concat(0, [xs[i], gs[i]])
      #  xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['discriminator.noise_stddev']*(i+1), dtype=config['dtype'])

      #  xgs.append(xg)
  
      #  s = [int(x) for x in xg.get_shape()]

      #  net = tf.concat(3, [net, xg])
      filter_size_w = 2
      filter_size_h = 2
      if i == depth-1:
          filter_size_w = int(net.get_shape()[1])
          filter_size_h = int(net.get_shape()[2])
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      if i == 0:
          length = 1
      for j in range(length):
          net_dense = net
          if i > 0:
              net_dense = activation(net_dense)
              if batch_norm is not None:
                net_dense = batch_norm(batch_size*2, name=prefix+'_expand_bna_'+str(i*10+j))(net_dense)
          newnet = conv2d(net_dense, size_dense, name=prefix+'_expand_layear'+str(i*10+j), k_w=3, k_h=3, d_h=1, d_w=1)
          net = tf.concat(3, [net, newnet])


      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('[discriminator] layer', net)

    k=-1
    if batch_norm is not None:
        net = batch_norm(batch_size*2, name=prefix+'_expand_bn_end_'+str(i))(net)
    net = activation(net)
    net = tf.reshape(net, [batch_size*2, -1])
 
    regularizers = []
    for regularizer in config['regularizers']:
        regs = regularizer(root_config, net, prefix)
        regularizers += regs

    return tf.concat(1, [net]+regularizers)


