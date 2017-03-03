import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os

def config(
        activation=lrelu,
        depth_increase=2,
        final_activation=None,
        first_conv_size=16,
        first_strided_conv_size=64,
        layer_regularizer=layer_norm_1,
        layers=5,
        resize=None,
        noise=None,
        layer_filter=None,
        progressive_enhancement=True,
        fc_layers=0,
        fc_layer_size=1024,
        strided=False,
        create=None
        ):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set("depth_increase", depth_increase)# Size increase of D's features on each layer
    selector.set("final_activation", final_activation)
    selector.set("first_conv_size", first_conv_size)
    selector.set("first_strided_conv_size", first_conv_size)
    selector.set("layers", layers) #Layers in D
    if create is None:
        selector.set('create', discriminator)
    else:
        selector.set('create', create)

    selector.set('fc_layer_size', fc_layer_size)
    selector.set('fc_layers', fc_layers)
    selector.set('layer_filter', layer_filter) #add information to D
    selector.set('layer_regularizer', layer_regularizer) # Size of fully connected layers
    selector.set('noise', noise) #add noise to input
    selector.set('progressive_enhancement', progressive_enhancement)
    selector.set('resize', resize)
    selector.set('strided', strided) #TODO: true does not work
    return selector.random_config()

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    activation = config['activation']
    final_activation = config['final_activation']
    depth_increase = config['depth_increase']
    depth = config['layers']
    batch_norm = config['layer_regularizer']
    strided = config.strided

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
        g_filter = tf.concat(axis=3, values=[g, config['layer_filter'](gan, x)])
        x_filter = tf.concat(axis=3, values=[x, config['layer_filter'](gan, x)])
        net = tf.concat(axis=0, values=[x_filter,g_filter] )
    else:
        net = tf.concat(axis=0, values=[x,g])
    if(config['noise']):
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)
        

    if strided:
        net = conv2d(net, config.first_strided_conv_size, name=prefix+'_expand', k_w=3, k_h=3, d_h=2, d_w=2,regularizer=None)
    else:
        net = conv2d(net, config.first_conv_size, name=prefix+'_expand', k_w=3, k_h=3, d_h=1, d_w=1,regularizer=None)

    for i in range(depth):
      #TODO better name for `batch_norm`?
      if batch_norm is not None:
          net = batch_norm(batch_size*2, name=prefix+'_expand_bn_'+str(i))(net)
      net = activation(net)
    
      #TODO: cross-d, overwritable
      # APPEND xs[i] and gs[i]
      if(i < len(xs)-1 and i > 0):
        if strided:
            index = i+1
        else:
            index = i
        if config['layer_filter']:
            x_filter_i = tf.concat(axis=3, values=[xs[index], config['layer_filter'](gan, xs[i])])
            g_filter_i = tf.concat(axis=3, values=[gs[index], config['layer_filter'](gan, xs[i])])
            xg = tf.concat(axis=0, values=[x_filter_i, g_filter_i])
        else:
            xg = tf.concat(axis=0, values=[xs[index], gs[index]])

        if(config['noise']):
            xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)
  
        if config['progressive_enhancement']:
            net = tf.concat(axis=3, values=[net, xg])
    
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      if strided:
          net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name=prefix+'_expand_layer'+str(i), k_w=3, k_h=3, d_h=filter_size_h, d_w=filter_size_w, regularizer=None)
      else:
          net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name=prefix+'_expand_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1, regularizer=None)
          net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('[discriminator] layer', net)

    k=-1

    net = tf.reshape(net, [batch_size*2, -1])

    if final_activation or config.fc_layers > 0:
        if batch_norm is not None:
            net = batch_norm(batch_size*2, name=prefix+'_expand_bn_end_'+str(i))(net)

    for i in range(config.fc_layers):
        net = activation(net)
        net = linear(net, config.fc_layer_size, scope=prefix+"_fc_end"+str(i))
        if final_activation or i < config.fc_layers - 1:
            if batch_norm is not None:
                net = batch_norm(batch_size*2, name=prefix+'_fc_bn_end_'+str(i))(net)

    if final_activation:
        net = final_activation(net)


    return net


