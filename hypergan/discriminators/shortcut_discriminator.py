import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os

def config(resize=None, layers=7):
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
    selector.set('fc_layers', [0])
    selector.set('fc_layer_size', [1024])

    selector.set('strided', False) #TODO: true does not work

    selector.set('create', discriminator)

    return selector.random_config()

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    activation = config['activation']
    final_activation = config['final_activation']
    depth_increase = config['depth_increase']
    depth = config['layers']
    batch_norm = config['regularizer']
    strided = config.strided
    losses = []

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
    if config['layer_filter']:
        g_filter = tf.concat(axis=3, values=[g, config['layer_filter'](gan, x)])
        x_filter = tf.concat(axis=3, values=[x, config['layer_filter'](gan, x)])
        net = tf.concat(axis=0, values=[x_filter,g_filter] )
    else:
        net = tf.concat(axis=0, values=[x,g])

    for i in range(depth):
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      net = conv2d(net, max(int(int(net.get_shape()[3])*depth_increase*2),16), name=prefix+'_expand_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1, regularizer=None)
      dims = int(net.get_shape()[3])//2
      loss_terms = tf.slice(net, [0,0,0,dims],[-1,-1,-1,dims])
      net = tf.slice(net, [0,0,0,0],[-1,-1,-1,dims])
      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')
      filter_size_w = int(net.get_shape()[1])
      filter_size_h = int(net.get_shape()[2])
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w*2,filter_size_h*2,1]
      loss_terms = tf.nn.avg_pool(loss_terms, ksize=filter, strides=stride, padding='SAME')
      print('loss term', loss_terms)
      loss_terms = tf.reshape(loss_terms, [int(loss_terms.get_shape()[0]),-1])
      losses.append(loss_terms)
      if batch_norm is not None:
          net = batch_norm(batch_size*2, name=prefix+'_expand_bn_'+str(i))(net)
      net = activation(net)
      print('[discriminator shortcut] net is', net)

    print('losses', losses)
    return tf.concat(losses, 1)


