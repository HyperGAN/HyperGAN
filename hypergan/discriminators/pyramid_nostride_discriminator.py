import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *
import os

def config(resize=None, layers=5):
    selector = hc.Selector()
    selector.set("final_activation", [tf.nn.tanh])#prelu("d_")])
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set('regularizer', [layer_norm_1]) # Size of fully connected layers

    selector.set("layers", layers) #Layers in D
    selector.set("depth_increase", [2])# Size increase of D's features on each layer

    selector.set('add_noise', [False]) #add noise to input
    selector.set('layer_filter', [None]) #add information to D
    selector.set('layer_filter.progressive_enhancement_enabled', True) #add information to D
    selector.set('noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
    selector.set('resize', [resize])

    selector.set('create', discriminator)
    
    return selector.random_config()

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    activation = config['activation']
    final_activation = config['final_activation']
    depth_increase = config['depth_increase']
    depth = config['layers']
    batch_norm = config['regularizer']

    # TODO: cross-d feature
    if(config['resize']):
        # shave off layers >= resize 
        def should_ignore_layer(layer, resize):
            return int(layer.get_shape()[1]) > config['resize'][0] or \
                   int(layer.get_shape()[2]) > config['resize'][1]

        xs = [px for px in xs if not should_ignore_layer(px, config['resize'])]
        gs = [pg for pg in gs if not should_ignore_layer(pg, config['resize'])]

        x = tf.image.resize_images(x,config['resize'], 1)
        g = tf.image.resize_images(g,config['resize'], 1)

    batch_size = int(x.get_shape()[0])
    # TODO: This is standard optimization from improved GAN, cross-d feature
    if config['layer_filter']:
        g_filter = tf.concat(3, [g, config['layer_filter'](gan, x)])
        x_filter = tf.concat(3, [x, config['layer_filter'](gan, x)])
        net = tf.concat(0, [x_filter,g_filter] )
    else:
        net = tf.concat(0, [x,g])
    if(config['add_noise']):
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise_stddev'], dtype=gan.config.dtype)
        
    net = conv2d(net, 16, name=prefix+'_expand', k_w=3, k_h=3, d_h=1, d_w=1,regularizer=None)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      #TODO better name for `batch_norm`?
      if batch_norm is not None:
          net = batch_norm(batch_size*2, name=prefix+'_expand_bn_'+str(i))(net)
      net = activation(net)
    
      #TODO: cross-d, overwritable
      # APPEND xs[i] and gs[i]
      if(i < len(xs) and i > 0):
        if config['layer_filter']:
            x_filter_i = tf.concat(3, [xs[i], config['layer_filter'](gan, xs[i])])
            g_filter_i = tf.concat(3, [gs[i], config['layer_filter'](gan, xs[i])])
            xg = tf.concat(0, [x_filter_i, g_filter_i])
        else:
            xg = tf.concat(0, [xs[i], gs[i]])

        xgs.append(xg)
  
        if config['layer_filter.progressive_enhancement_enabled']:
            net = tf.concat(3, [net, xg])
    
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name=prefix+'_expand_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1, regularizer=None)
      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('[discriminator] layer', net)

    k=-1
    if batch_norm is not None:
        net = batch_norm(batch_size*2, name=prefix+'_expand_bn_end_'+str(i))(net)
    net = final_activation(net)
    net = tf.reshape(net, [batch_size*2, -1])

    return net


