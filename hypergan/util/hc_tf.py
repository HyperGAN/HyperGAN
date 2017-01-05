#This is like ops.py, but for larger compositions of graph nodes.
#TODO: could use a better name
from hypergan.util.ops import *
from hypergan.util.globals import *

def build_reshape(output_size, nodes, method, batch_size, dtype):
    node_size = sum([int(x.get_shape()[1]) for x in nodes])
    dims = output_size-node_size
    if(method == 'noise'):
        noise = tf.random_uniform([batch_size, dims],-1, 1, dtype=dtype)
        result = tf.concat(1, nodes+[noise])
    elif(method == 'tiled'):
        t_nodes = tf.concat(1, nodes)
        dims =  int(t_nodes.get_shape()[1])
        result= tf.tile(t_nodes,[1, output_size//dims])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            #zeros = tf.zeros([batch_size, width])
            slice = tf.slice(result, [0,0],[batch_size,width])
            result = tf.concat(1, [result, slice])


    elif(method == 'zeros'):
        result = tf.concat(1, nodes)
        result = tf.pad(result, [[0, 0],[dims//2, dims//2]])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            zeros = tf.zeros([batch_size, width],dtype=dtype)
            result = tf.concat(1, [result, zeros])
    elif(method == 'linear'):
        result = tf.concat(1, [y, z])
        result = linear(result, dims, 'g_input_proj')
    elif(method == 'atrous'):
        pass 
    elif(method == 'conv'):
        result = nodes[1]
        result = tf.reshape(result,[batch_size, 1, 1, 64])
        result = conv2d(result, 256*48, name='g_input_conv', k_w=1, k_h=1, d_h=1, d_w=1)

    else:
        assert 1 == 0
    return result

def pad_input(primes, output_size, nodes):
    node_size = sum([int(x.get_shape()[1]) for x in nodes])
    dims = output_size
    prime = primes[0]*primes[1]
    while(dims-node_size < 0):
        dims += prime
    if(dims % (prime) != 0):
        dims += (prime-(dims % (prime)))
    return dims

def find_smallest_prime(x, y):
    for i in range(3,x-1):
        for j in range(1, y-1):
            if(x % (i) == 0 and y % (j) == 0 and x // i == y // j):
                #if(i==j):
                #    return 2,2
                return i,j
    return None,None

def build_conv_tower(result, layers, filter, batch_size, batch_norm_enabled, batch_norm_last_layer, name, activation, stride=2):
    last_layer_size = layers[0]
    for i, layer in enumerate(layers):
        istride = stride
        if filter > result.get_shape()[1]:
            filter = int(result.get_shape()[1])
            istride = 1
        if filter > result.get_shape()[2]:
            filter = int(result.get_shape()[2])
            istride = 1

        result = conv2d(result, layer, name=name+str(i), k_w=filter, k_h=filter, d_h=istride, d_w=istride)
        if(len(layers) == i+1):
            if(batch_norm_last_layer):
                result = batch_norm(batch_size, name=name+'_bn_'+str(i))(result)
        else:
            if(batch_norm_enabled):
                result = batch_norm(batch_size, name=name+'_bn_'+str(i))(result)
            #result += conv2d(result, layer, name=name+'id'+str(i), k_w=1, k_h=1, d_h=1, d_w=1)
            result = activation(result)
    return result


def build_resnet(result, depth, filter, name, activation, batch_size, batch_norm_enabled, conv=False):
    root=result
    for i in range(depth):
        if(conv):
            result = conv2d(result, int(result.get_shape()[-1]), name=name+str(i), k_w=filter, k_h=filter, d_h=1, d_w=1)
        else:
            result = deconv2d(result, result.get_shape(), name=name+str(i), k_w=filter, k_h=filter, d_h=1, d_w=1)
        if(batch_norm_enabled):
            result = batch_norm(batch_size, name=name+'_bn_'+str(i))(result)
        if i % 2 == 1:
            result += root
            root = result
        #if i % 3 == 2 and filter != 1:
        #    filter2=1
        #    result += conv2d(result, int(result.get_shape()[-1]), name='1x1-'+str(i), k_w=filter2, k_h=filter2, d_h=1, d_w=1)
        result = activation(result)
    return result


def build_deconv_tower(result, layers, dims, conv_size, name, activation, batch_norm_enabled, batch_norm_last_layer, batch_size,last_layer_stddev,stride=2):
    for i, layer in enumerate(layers):
        istride=stride
        j=int(result.get_shape()[1])*istride
        k=int(result.get_shape()[2])*istride

        output = [batch_size, j,k,int(layer)]
        stddev = 0.002
        if(len(layers) == i+1):
            if(batch_norm_last_layer):
                stddev = 0.15
        result = deconv2d(result, output, name=name+str(i), k_w=conv_size, k_h=conv_size, d_h=istride, d_w=istride, stddev=stddev)
        if(len(layers) == i+1):
            if(batch_norm_last_layer):
                result = batch_norm(batch_size, name=name+'_bn_'+str(i))(result)
        else:
            if(batch_norm_enabled):
                result = batch_norm(batch_size, name=name+'_bn_'+str(i))(result)
            result = activation(result)
    return result


def build_categories_config(num):
    return [np.random.randint(2,30) for i in range(np.random.randint(2,15))]



def build_conv_config(layers, start, end):
    def get_layer(layer, i):
        reverse = 2**(layer+3)*i
        noise = int(np.random.uniform(-1,1)*10)*i

        result = reverse
        if(result < 3): 
            result = 3
        if(reverse+noise > 3):
            result = reverse+noise
        return result
    def get_option(i):
        return [get_layer(layer, i) for layer in range(layers)]
    #return [sorted(get_option(i)) for i in np.arange(start, end)]
    return [[64, 128, 256,512, 1024]]


def build_deconv_config(layers,start, end):
    def get_layer(layer, i):
        reverse = 2**(layers-layer+3)*i
        noise = int(np.random.uniform(-1,1)*10)*i

        result = reverse
        if(result < 3): 
            result = 3
        if(reverse+noise > 3):
            result = reverse+noise
        return result
    def get_option(i):
        return [get_layer(layer, i) for layer in range(layers)]
    #return [list(reversed(sorted(get_option(i)))) for i in np.arange(start, end)]
    return [[16*3*2*2*2*2, 128, 3]]


def build_atrous_layer(result, layer, filter, name='g_atrous'):
    padding="SAME"
    rate=2
    #Warning only float32
    filters = tf.get_variable(name+'_w', [filter, filter, result.get_shape()[-1], layer],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    result = tf.nn.atrous_conv2d(result, filters, rate, padding)
    return result


def get_graph_vars(sess, graph):
   return {}
   # summary = get_tensor("hc_summary")
   # all_vars = sess.run([s[3] for s in summary])
   # i=0
   # retv = {'weights':{}}
   # for shape, name, dtype, _ in summary:
   #     data=all_vars[i]
   #     retv['weights'][name]={
   #             'shape':[int(s) for s in shape],
   #             'name':name,
   #             'dtype':str(dtype),
   #             'value':str(data)
   #             }
   #     i+=1
   #     
   # return retv
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
        print("IDENTITY FROM", result)
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
    print("residual block", id, left+right)
    return left+right

def residual_block_deconv(result, activation, batch_size,id,name, output_channels=None, stride=2, channels=None):
    size = int(result.get_shape()[-1])
    s = result.get_shape()
    if(id=='widen'):
        output_shape = [s[0], s[1], s[2],s[3]*2]
        output_shape = [int(o) for o in output_shape]
        left = deconv2d(result, output_shape, name=name+'l', k_w=3, k_h=3, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = deconv2d(result, output_shape, name=name+'r', k_w=3, k_h=3, d_h=1, d_w=1)
    elif(id=='bottleneck'):
        output_shape = [s[0], s[1], s[2],channels]
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn_pre')(result)
        result = activation(result)
        left = deconv2d(result, output_shape, name=name+'l', k_w=3, k_h=3, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = deconv2d(result, output_shape, name=name+'r', k_w=3, k_h=3, d_h=1, d_w=1)
    elif(id=='identity'):
        output_shape = s
        output_shape = [int(o) for o in output_shape]
        left = result
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l', k_w=3, k_h=3, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn2')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = result
    elif(id=='deconv'):
        output_shape = [s[0], s[1]*stride, s[2]*stride,s[3]//stride]
        if(output_channels):
            output_shape[-1] = output_channels
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        left = result
        right = result
        left = deconv2d(left, output_shape, name=name+'l', k_w=stride+1, k_h=stride+1, d_h=stride, d_w=stride)
        left = batch_norm(batch_size, name=name+'lbn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_w=3, k_h=3, d_h=1, d_w=1)
        right = deconv2d(right, output_shape, name=name+'r', k_w=stride+1, k_h=stride+1, d_h=stride, d_w=stride)
    return left+right

def block_deconv(result, activation, batch_size,id,name, output_channels=None, stride=2, noise_shape=None,filter=3):

    size = int(result.get_shape()[-1])
    s = result.get_shape()
    if(id=='deconv'):
        output_shape = [s[0], s[1]*stride, s[2]*stride,s[3]//stride]
        if(output_channels):
            output_shape[-1] = output_channels
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = deconv2d(result, output_shape, name=name+'l', k_w=stride+1, k_h=stride+1, d_h=stride, d_w=stride)
    elif(id=='identity'):
        output_shape = [s[0], s[1], s[2],output_channels]
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)

        if(noise_shape):
          noise = tf.random_uniform(noise_shape,-1, 1,dtype=config['dtype'])
          result = tf.concat(3, [result, noise])
        result = deconv2d(result, output_shape, name=name+'l', k_w=filter, k_h=filter, d_h=1, d_w=1)
    return result

def block_conv(result, activation, batch_size,id,name, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, dropout=None):
    size = int(result.get_shape()[-1])
    s = result.get_shape()
    if(batch_norm is not None):
        result = batch_norm(batch_size, name=name+'bn')(result)
    print("DROPOUT IS", dropout)
    result = activation(result)
    if(dropout):
        z = get_tensor('original_z')
        mask = linear(z, s[1]*s[2]*s[3], scope="g_lin_proj_mask")
        mask = tf.reshape(mask, result.get_shape())
        result *= tf.nn.sigmoid(mask)
        #HACKS
        #mask_noise = tf.random_uniform(result.get_shape(), 0, 1)
        #set_tensor('mask_noise', mask_noise)
        #mask = tf.greater(mask_noise, tf.zeros_like(mask_noise))
        #result = result * mask_noise#* tf.cast(mask, tf.float32)
        #result = tf.nn.dropout(result, dropout, seed=1)

    if(noise_shape):
      noise = tf.random_uniform(noise_shape,-1, 1,dtype=dtype)
      noise = tf.zeros_like(noise)
      result = tf.concat(3, [result, noise])
    if(id=='conv'):
        result = conv2d(result, int(result.get_shape()[3]), name=name, k_w=filter, k_h=filter, d_h=stride, d_w=stride)
    elif(id=='identity'):
        result = conv2d(result, output_channels, name=name, k_w=filter, k_h=filter, d_h=1, d_w=1)

    return result


def block_conv_dts(result, activation, batch_size,id,name, output_channels=None, stride=2):
    size = int(result.get_shape()[-1])
    s = result.get_shape()
    result = batch_norm(batch_size, name=name+'bn')(result)
    result = activation(result)
    if(id=='conv'):
        result = tf.depth_to_space(result, 2)
        result = conv2d(result, int(result.get_shape()[3])*2, name=name, k_w=3, k_h=3, d_h=1, d_w=1)
    elif(id=='end'):
        result = tf.depth_to_space(result, 2)

        result = conv2d(result, output_channels, name=name, k_w=3, k_h=3, d_h=1, d_w=1)
    return result



def dense_block(result, k, activation, batch_size, id, name):
    size = int(result.get_shape()[-1])
    if(id=='layer'):
        identity = tf.identity(result)
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        fltr = min(3, int(result.get_shape()[1]))
        fltr = min(fltr, int(result.get_shape()[2]))
        result = conv2d(result, k, name=name+'conv', k_w=fltr, k_h=fltr, d_h=1, d_w=1)

        return tf.concat(3,[identity, result])
    elif(id=='transition'):
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = conv2d(result, size, name=name+'id', k_w=1, k_h=1, d_h=1, d_w=1)
        filter = [1,2,2,1] #todo verify
        stride = [1,2,2,1]
        result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
        return result


