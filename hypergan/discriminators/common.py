import tensorflow as tf
import hyperchamber as hc

def repeating_block(ops, net, config, depth):
   layer_regularizer = config.layer_regularizer
   activation = config.activation
   filter_size_w = 2
   filter_size_h = 2
   filter = [1,filter_size_w,filter_size_h,1]
   stride = [1,filter_size_w,filter_size_h,1]
   for i in range(config.block_repeat_count-1):
     if layer_regularizer is not None:
        net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)
     net = ops.conv2d(net, 3, 3, 1, 1, depth)
     net = activation(net)
     print("[discriminator] hidden layer", net)

   net = ops.conv2d(net, 3, 3, 1, 1, depth)
   net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')
   print('[discriminator] layer', net)
   return net

def standard_block(ops, net, config, depth):
   filter_size_w = 2
   filter_size_h = 2
   filter = [1,filter_size_w,filter_size_h,1]
   stride = [1,filter_size_w,filter_size_h,1]

   net = ops.conv2d(net, 3, 3, 1, 1, depth)
   #TODO
   net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')
   print('[discriminator] layer', net)
   return net

def strided_block(ops, net, config, depth):
   net = ops.conv2d(net, 3, 3, 2, 2, depth)
   print('[discriminator] layer', net)
   return net
