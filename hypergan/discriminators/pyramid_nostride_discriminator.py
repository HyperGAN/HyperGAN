import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *
import hypergan.regularizers.minibatch_regularizer as minibatch_regularizer
import os
import importlib

def load(root_config, x, g, xs, gs):
    print("LOADING")
    return discriminator(root_config, config(root_config), x, g, xs, gs)

#### TODO This belongs in hyperchamber ###
# This looks up a function by name.   Should it be part of hyperchamber?
def get_function(name):
    if not isinstance(name, str):
        return name
    namespaced_method = name.split(":")[1]
    method = namespaced_method.split(".")[-1]
    namespace = ".".join(namespaced_method.split(".")[0:-1])
    return getattr(importlib.import_module(namespace),method)

# Take a config and replace any string starting with 'function:' with a function lookup.
def lookup_functions(config):
    for key, value in config.items():
        if(isinstance(value, str) and value.startswith("function:")):
            config[key]=get_function(value)
        if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
            config[key]=[get_function(v) for v in value]
            
    return config

#############################


def config(root_config):
    hc.reset()
    hc.set("activation", [lrelu])#prelu("d_")])
    hc.set('regularizer', [layer_norm_1, batch_norm_1]) # Size of fully connected layers

    hc.set("layers", [4,5,3]) #Layers in D
    hc.set("depth_increase", [1,2,4])# Size increase of D's features on each layer

    hc.set('add_noise', [True]) #add noise to input
    hc.set('noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
    hc.set('regularizers', [[],[minibatch_regularizer.get_features]]) # these regularizers get applied at the end of D
    
    # TODO loading
    #config_path = os.path.expanduser('~/.hypergan/configs/'+root_config['uuid']+'_pyramid_nostride_discriminator.json')
    #print("Loading "+config_path)
    return lookup_functions(hc.random_config())#hc.load_or_create_config(config_path, None))

def discriminator(root_config, config, x, g, xs, gs):
    activation = config['activation']
    batch_size = int(x.get_shape()[0])
    depth_increase = config['depth_increase']
    depth = config['layers']
    batch_norm = config['regularizer']

    if(config['add_noise']):
        x += tf.random_normal(x.get_shape(), mean=0, stddev=config['noise_stddev'], dtype=root_config['dtype'])

    net = x
    net = conv2d(net, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)

    xgs = []
    xgs_conv = []
    for i in range(depth):
      if batch_norm is not None:
          net = batch_norm(batch_size*2, name='d_expand_bn_'+str(i))(net)
      net = activation(net)
      # APPEND xs[i] and gs[i]
      if(i < len(xs) and i > 0):
        xg = tf.concat(0, [xs[i], gs[i]])
        xg += tf.random_normal(xg.get_shape(), mean=0, stddev=config['noise_stddev']*i, dtype=root_config['dtype'])

        xgs.append(xg)
  
        s = [int(x) for x in xg.get_shape()]

        net = tf.concat(3, [net, xg])
      filter_size_w = 2
      filter_size_h = 2
      filter = [1,filter_size_w,filter_size_h,1]
      stride = [1,filter_size_w,filter_size_h,1]
      net = conv2d(net, int(int(net.get_shape()[3])*depth_increase), name='d_expand_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1)
      net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')

      print('[discriminator] layer', net)

    k=-1
    if batch_norm is not None:
        net = batch_norm(batch_size*2, name='d_expand_bn_end_'+str(i))(net)
    net = activation(net)
    net = tf.reshape(net, [batch_size, -1])

    regularizers = []
    for regularizer in config['regularizers']:
        regs = regularizer(root_config, net)
        regularizers += regs

 
    return tf.concat(1, [net]+regularizers)


