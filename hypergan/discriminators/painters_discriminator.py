import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *

def config(resize=None, layers=None):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set('regularizer', [batch_norm_1]) # Size of fully connected layers

    if layers == None:
        layers = [4,5,3]
    selector.set("layers", layers) #Layers in D
    selector.set("depth_increase", [1,2])# Size increase of D's features on each layer

    selector.set('add_noise', [True]) #add noise to input
    selector.set('noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
    selector.set('regularizers', [[minibatch_regularizer.get_features]]) # these regularizers get applied at the end of D
    selector.set('resize', [resize])

    selector.set('create', discriminator)
    
    return selector.random_config()

def discriminator(root_config, config, x, g, xs, gs, prefix='d_'):
    layers = config['layers']
    transitions = config['transitions']
    activation = config['activation']
    batch_size = int(x.get_shape()[0])
    batch_norm = config['regularizers.layer']

    if(config['resize']):
        # shave off layers >= resize 
        def should_ignore_layer(layer, resize):
            return int(layer.get_shape()[1]) >= config['resize'][0] or \
                   int(layer.get_shape()[2]) >= config['resize'][1]

        xs = [px for px in xs if not should_ignore_layer(px, config['resize'])]
        gs = [pg for pg in gs if not should_ignore_layer(pg, config['resize'])]

        x = tf.image.resize_images(x,config['resize'], 1)
        g = tf.image.resize_images(g,config['resize'], 1)


    if(config['add_noise']):
        x += tf.random_normal(x.get_shape(), mean=0, stddev=config['noise_stddev'], dtype=root_config['dtype'])

 

    result = x
    result = conv2d(result, 16, name=prefix+'expand', k_w=3, k_h=3, d_h=1, d_w=1)
    result = batch_norm(config['batch_size']*2, name=prefix+'expand_bn')(result)
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
        result = conv2d(result, 2**(3+i), name=prefix+"add_xg"+str(i)+"-"+str(j), k_w=3, k_h=3, d_h=1, d_w=1)
        result = batch_norm(config['batch_size']*2, name=prefix+'add_xg_bn_'+str(i)+"-"+str(j))(result)
        result = activation(result)
        print("painters size", result)

      result = tf.nn.max_pool(result, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')


    set_tensor("xgs", xgs)
    set_tensor("xgs_conv", xgs_conv)

    result = batch_norm(config['batch_size']*2, name=prefix+'expand_bn_end_'+str(i))(result)
    result = activation(result)
    print("painters size", result)
    result = tf.reshape(result, [config['batch_size']*2, -1])

    result = linear(result, 1024, scope=prefix+"fc_end1")
    result = batch_norm(config['batch_size']*2, name='d_bn_end1')(result)
    result = activation(result)

    regularizers = []
    for regularizer in config['regularizers']:
        regs = regularizer(root_config, net, prefix)
        regularizers += regs
    return result



