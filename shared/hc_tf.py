from shared.ops import *

#hc_tf.config.optimizer(["adam"], lr=[1e-3,1e-5])
#hc_tf.config.deconv
#hc_tf.config.conv(filter=3, )
#hc_tf.config.reshape(method)

#hc_tf.build.optimizer()
#hc_tf.build.deconv
#hc_tf.build.conv
#hc_tf.build.reshape()

def build_reshape(output_size, nodes, method, batch_size):
    node_size = sum([int(x.get_shape()[1]) for x in nodes])
    dims = output_size-node_size
    if(method == 'noise'):
        noise = tf.random_uniform([batch_size, noise_dims],-1, 1)
        result = tf.concat(1, nodes+[noise])
    elif(method == 'zeros'):
        result = tf.concat(1, nodes)
        result = tf.pad(result, [[0, 0],[dims//2, dims//2]])
        width = output_size - int(result.get_shape()[1])
        if(width > 0):
            zeros = tf.zeros([batch_size, width])
            result = tf.concat(1, [result, zeros])
    elif(method == 'linear'):
        result = tf.concat(1, [y, z])
        result = linear(result, dims, 'g_input_proj')
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
    print('dims', dims % (prime))
    return dims

def find_smallest_prime(x, y):
    for i in range(3,x-1):
        for j in range(3, y-1):
            print(i,j,x,y)
            if(x % (i) == 0 and y % (j) == 0 and x // i == y // j):
                return i,j
    return None,None
