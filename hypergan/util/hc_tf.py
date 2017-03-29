#This is like ops.py, but for larger compositions of graph nodes.
#TODO: could use a better name
from hypergan.util.ops import *

#TODO Remove
def build_reshape(output_size, nodes, method, batch_size, dtype):
    node_size = sum([int(x.get_shape()[1]) for x in nodes])
    dims = output_size-node_size
    if(method == 'noise'):
        noise = tf.random_uniform([batch_size, dims],-1, 1, dtype=dtype)
        result = tf.concat(axis=1, values=nodes+[noise])
    elif(method == 'tiled'):
        t_nodes = tf.concat(axis=1, values=nodes)
        dims =  int(t_nodes.get_shape()[1])
        result= tf.tile(t_nodes,[1, output_size//dims])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            #zeros = tf.zeros([batch_size, width])
            slice = tf.slice(result, [0,0],[batch_size,width])
            result = tf.concat(axis=1, values=[result, slice])


    elif(method == 'zeros'):
        result = tf.concat(axis=1, values=nodes)
        result = tf.pad(result, [[0, 0],[dims//2, dims//2]])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            zeros = tf.zeros([batch_size, width],dtype=dtype)
            result = tf.concat(axis=1, values=[result, zeros])
    elif(method == 'linear'):
        result = tf.concat(axis=1, values=[y, z])
        result = linear(result, dims, 'g_input_proj')

    else:
        assert 1 == 0
    return result

#TODO can live elsewhere
def find_smallest_prime(x, y):
    for i in range(3,x-1):
        for j in range(1, y-1):
            if(x % (i) == 0 and y % (j) == 0 and x // i == y // j):
                #if(i==j):
                #    return 2,2
                return i,j
    return None,None

#TODO can live elsewhere
def build_categories_config(num):
    return [np.random.randint(2,30) for i in range(np.random.randint(2,15))]

#TODO: This is not used
def build_atrous_layer(result, layer, filter, name='g_atrous'):
    padding="SAME"
    rate=2
    #Warning only float32
    filters = tf.get_variable(name+'_w', [filter, filter, result.get_shape()[-1], layer],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    result = tf.nn.atrous_conv2d(result, filters, rate, padding)
    return result

#TODO This is broken
def residual_block(result, activation, batch_size,id,name):
    size = int(result.get_shape()[-1])
    if(id=='widen'):
        left = conv2d(result, size*2, name=name+'l', k_w=3, k_h=3, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = conv2d(left, size*2, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = conv2d(result, size*2, name=name+'r', k_w=3, k_h=3, d_h=1, d_w=1)
    elif(id=='identity'):
        left = result
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = conv2d(left, size, name=name+'l', k_w=3, k_h=3, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn2')(left)
        left = activation(left)
        left = conv2d(left, size, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = result
    elif(id=='conv'):
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        left = result
        right = result
        left = conv2d(left, size*2, name=name+'l', k_w=3, k_h=3, d_h=2, d_w=2)
        left = batch_norm(batch_size, name=name+'lbn')(left)
        left = activation(left)
        left = conv2d(left, size*2, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = conv2d(right, size*2, name=name+'r', k_w=3, k_h=3, d_h=2, d_w=2)
    return left+right

#TODO move this somewhere?  Used by graph creation
def block_conv(result, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    size = int(result.get_shape()[-1])
    result = activation(result)
    if(batch_norm is not None):
        result = batch_norm(batch_size, name=name+'bn')(result)
    s = result.get_shape()
    if(sigmoid_gate is not None):
        mask = linear(sigmoid_gate, s[1]*s[2]*s[3], scope=name+"lin_proj_mask")
        mask = tf.reshape(mask, result.get_shape())
        result *= tf.nn.sigmoid(mask)

    if(resize):
        result = tf.image.resize_images(result, resize, 1)

    if reshaped_z_proj is not None:
        result = tf.concat(axis=3,values=[result, reshaped_z_proj])

    if(noise_shape):
      noise = tf.random_uniform(noise_shape,-1, 1,dtype=dtype)
      result = tf.concat(axis=3, values=[result, noise])
    if(id=='conv'):
        result = conv2d(result, int(result.get_shape()[3]), name=name, k_w=filter, k_h=filter, d_h=stride, d_w=stride)
    elif(id=='identity'):
        result = conv2d(result, output_channels, name=name, k_w=filter, k_h=filter, d_h=1, d_w=1)

    return result

#TODO this is not used
def dense_block(result, k, activation, batch_size, id, name):
    size = int(result.get_shape()[-1])
    if(id=='layer'):
        identity = tf.identity(result)
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        fltr = min(3, int(result.get_shape()[1]))
        fltr = min(fltr, int(result.get_shape()[2]))
        result = conv2d(result, k, name=name+'conv', k_w=fltr, k_h=fltr, d_h=1, d_w=1)

        return tf.concat(axis=3,values=[identity, result])
    elif(id=='transition'):
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = conv2d(result, size, name=name+'id', k_w=1, k_h=1, d_h=1, d_w=1)
        filter = [1,2,2,1] #todo verify
        stride = [1,2,2,1]
        result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
        return result
