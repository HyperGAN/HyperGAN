import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os

def config(
        activation=lrelu,
        depth_increase=2,
        final_activation=tf.nn.tanh,
        layer_regularizer=layer_norm_1,
        layers=7,
        resize=None,
        noise=None,
        layer_filter=None,
        progressive_enhancement=True,
        fc_layers=0,
        fc_layer_size=1024,
        strided=False,
        layer=standard_layer
        ):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set("depth_increase", depth_increase)# Size increase of D's features on each layer
    selector.set("final_activation", final_activation)
    selector.set("layers", layers) #Layers in D
    selector.set('create', discriminator)
    selector.set('fc_layer_size', fc_layer_size)
    selector.set('fc_layers', fc_layers)
    selector.set('layer_filter', layer_filter) #add information to D
    selector.set('layer_regularizer', layer_regularizer) # Size of fully connected layers
    selector.set('noise', noise) #add noise to input
    selector.set('progressive_enhancement', progressive_enhancement) # Adds a resized version of X to each layer
    selector.set('resize', resize) # If set to dimensions, will resize the g/x input to these dimensions
    selector.set('layer', layer) # this function is called to set up each layer of the graph
    return selector.random_config()

def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    activation = config['activation']
    final_activation = config['final_activation']
    depth_increase = config['depth_increase']
    depth = config['layers']
    layer_regularizer = config['layer_regularizer']
    strided = config.strided

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

    if(config['noise']):
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)

    net = config.layer(net, gan, config, i=0, prefix=prefix)

    for i in range(depth):
      if layer_regularizer is not None:
          net = layer_regularizer(batch_size*2, name=prefix+'_expand_bn_'+str(i))(net)
      net = activation(net)

      if(i < len(xs)-1 and i > 0):
        if config['layer_filter']:
            x_filter_i = tf.concat(axis=3, values=[xs[i], config['layer_filter'](gan, xs[i])])
            g_filter_i = tf.concat(axis=3, values=[gs[i], config['layer_filter'](gan, xs[i])])
            xg = tf.concat(axis=0, values=[x_filter_i, g_filter_i])
        else:
            xg = tf.concat(axis=0, values=[xs[i], gs[i]])

        if(config['noise']):
            xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)

        if config['progressive_enhancement']:
            net = tf.concat(axis=3, values=[net, xg])

      net = config.layer(net, gan, config, i=i+1, prefix=prefix)

      print('[discriminator] layer', net)

    net = tf.reshape(net, [batch_size*2, -1])

    if final_activation or config.fc_layers > 0:
        if layer_regularizer is not None:
            net = layer_regularizer(batch_size*2, name=prefix+'_expand_bn_end_'+str(i))(net)

    for i in range(config.fc_layers):
        net = activation(net)
        net = linear(net, config.fc_layer_size, scope=prefix+"_fc_end"+str(i))
        if final_activation or i < config.fc_layers - 1:
            if layer_regularizer is not None:
                net = layer_regularizer(batch_size*2, name=prefix+'_fc_bn_end_'+str(i))(net)

    if final_activation:
        net = final_activation(net)


    return net


