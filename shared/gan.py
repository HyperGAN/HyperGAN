
from shared.ops import *
from shared.util import *
from shared.hc_tf import *
import shared.vggnet_loader as vggnet_loader
import tensorflow as tf
TINY = 1e-12

def generator(config, inputs, reuse=False):
    x_dims = config['x_dims']
    output_channels = config['channels']
    activation = config['g_activation']
    batch_size = config['batch_size']
    print("CREATE", reuse)
    if(config['include_f_in_d'] == True):
        output_channels += 1
    with(tf.variable_scope("generator", reuse=reuse)):
        output_shape = x_dims[0]*x_dims[1]*config['channels']
        primes = find_smallest_prime(x_dims[0], x_dims[1])
        z_proj_dims = int(config['conv_g_layers'][0])
        print("PRIMES ARE", primes, z_proj_dims*primes[0]*primes[1])
        if(int(config['z_dim_random_uniform']) > 0):
            z_dim_random_uniform = tf.random_uniform([config['batch_size'], int(config['z_dim_random_uniform'])],-1, 1,dtype=config['dtype'])
            #z_dim_random_uniform = tf.zeros_like(z_dim_random_uniform)
            z_dim_random_uniform = tf.identity(z_dim_random_uniform)
            inputs.append(z_dim_random_uniform)
        else:
            z_dim_random_uniform = None

        if(config['g_project'] == 'linear'):
            original_z = tf.concat(1, inputs)
            result = linear(original_z, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj")
        elif(config['g_project']=='tiled'):
            result = build_reshape(z_proj_dims*primes[0]*primes[1], inputs, 'tiled', config['batch_size'], config['dtype'])
        elif(config['g_project']=='noise'):
            result = build_reshape(z_proj_dims*primes[0]*primes[1], inputs, 'noise', config['batch_size'], config['dtype'])
        elif(config['g_project']=='atrous'):
            result = build_reshape(z_proj_dims*primes[0]*primes[1], inputs, 'atrous', config['batch_size'], config['dtype'])


        result = tf.reshape(result,[config['batch_size'], primes[0], primes[1], z_proj_dims])

        if config['conv_g_layers']:
            if(config['g_strategy'] == 'small-skip'):
                widenings = 6
                stride = 2
                zs = [None]
                h = int(result.get_shape()[1])
                w = int(result.get_shape()[2])
                sc_layers = config['g_skip_connections_layers']
                for i in range(widenings-1):
                    w*=stride
                    h*=stride
                    size = w*h*int(sc_layers[i])
                    print("original z", original_z, size)
                    if(size != 0):
                        new_z = tf.random_uniform([config['batch_size'], size],-1, 1,dtype=config['dtype'])
                        print('new_z', new_z)
                        #new_z = linear(new_z, size, scope='g_skip_z_'+str(i))
                        new_z = tf.reshape(new_z, [config['batch_size'], h,w, sc_layers[i]])

                        zs.append(new_z)
                for i in range(widenings):
                    print("BEFORE SIZE IS" ,result)
                    if(config['g_skip_connections'] and i!=0 and i < len(zs)):
                        result = tf.concat(3, [result, zs[i]])
                    if(i==widenings-1):
                        result = block_deconv(result, activation, batch_size, 'deconv', 'g_layers_'+str(i), output_channels=config['channels'], stride=stride)
                    else:
                        result = block_deconv(result, activation, batch_size, 'deconv', 'g_layers_'+str(i), stride=stride)
                    print("SIZE IS" ,result)
  
            elif(config['g_strategy'] == 'wide-resnet'):
                #result = residual_block_deconv(result, activation, batch_size, 'widen', 'g_layers_p')
                #result = residual_block_deconv(result, activation, batch_size, 'identity', 'g_layers_i1')
                widenings = 6
                stride = 2
                zs = [None]
                h = int(result.get_shape()[1])
                w = int(result.get_shape()[2])
                sc_layers = config['g_skip_connections_layers']
                for i in range(widenings-1):
                    w*=stride
                    h*=stride
                    size = w*h*int(sc_layers[i])
                    print("original z", original_z, size)
                    if(size != 0):
                        new_z = tf.random_uniform([config['batch_size'], size],-1, 1,dtype=config['dtype'])
                        print('new_z', new_z)
                        #new_z = linear(new_z, size, scope='g_skip_z_'+str(i))
                        new_z = tf.reshape(new_z, [config['batch_size'], h,w, sc_layers[i]])

                        zs.append(new_z)
                for i in range(widenings):
                    print("BEFORE SIZE IS" ,result)
                    if(config['g_skip_connections'] and i!=0 and i < len(zs)):
                        result = tf.concat(3, [result, zs[i]])
                    if(i==widenings-1):
                        result = residual_block_deconv(result, activation, batch_size, 'deconv', 'g_layers_'+str(i), output_channels=config['channels']+13, stride=stride)
                        result = residual_block_deconv(result, activation, batch_size, 'bottleneck', 'g_layers_bottleneck_'+str(i), channels=config['channels'])
                        #result = residual_block_deconv(result, activation, batch_size, 'identity', 'g_layers_i_'+str(i))
                    else:
                        result = residual_block_deconv(result, activation, batch_size, 'deconv', 'g_layers_'+str(i), stride=stride)
                        result = residual_block_deconv(result, activation, batch_size, 'identity', 'g_layers_i_'+str(i))
                    print("SIZE IS" ,result)
                #result = tf.reshape(result,[config['batch_size'],x_dims[0],x_dims[1],-1])
                #result = batch_norm(batch_size, name='g_rescap_bn')(result)
                #result = activation(result)
                #result = conv2d(result, config['channels'], name='g_grow', k_w=3,k_h=3, d_h=1, d_w=1)
                #result = tf.slice(result, [0,0,0,0],[config['batch_size'], x_dims[0],x_dims[1],config['channels']])
                #print("END SIZE IS", result)
                #stride = x_dims[0]//int(result.get_shape()[1])
                #result = build_deconv_tower(result, [output_channels], x_dims, stride+1, 'g_conv_2', config['g_activation'], config['g_batch_norm'], config['g_batch_norm_last_layer'], config['batch_size'], config['g_last_layer_stddev'], stride=stride)
            elif(config['g_strategy'] == 'huge_deconv'):
                result = batch_norm(config['batch_size'], name='g_bn_lin_proj')(result)
                result = config['g_activation'](result)
                result = build_resnet(result, config['g_resnet_depth'], config['g_resnet_filter'], 'g_conv_res_', config['g_activation'], config['batch_size'], config['g_batch_norm'])
                result = build_deconv_tower(result, config['conv_g_layers'][1:2]+[output_channels], x_dims, config['g_huge_filter'], 'g_conv_2', config['g_activation'], config['g_batch_norm'], config['g_batch_norm_last_layer'], config['batch_size'], config['g_last_layer_stddev'], stride=config['g_huge_stride'])
            elif(config['g_strategy'] == 'deep_deconv'):
                result = batch_norm(config['batch_size'], name='g_bn_lin_proj')(result)
                result = config['g_activation'](result)
                result = build_deconv_tower(result, config['conv_g_layers'][1:2], x_dims, config['conv_size'], 'g_conv_', config['g_activation'], config['g_batch_norm'], True, config['batch_size'], config['g_last_layer_stddev'])
                result = config['g_activation'](result)
                result = build_resnet(result, config['g_resnet_depth'], config['g_resnet_filter'], 'g_conv_res_', config['g_activation'], config['batch_size'], config['g_batch_norm'])
                result = build_deconv_tower(result, config['conv_g_layers'][2:-1]+[output_channels], x_dims, config['g_post_res_filter'], 'g_conv_2', config['g_activation'], config['g_batch_norm'], config['g_batch_norm_last_layer'], config['batch_size'], config['g_last_layer_stddev'])

        if(config['include_f_in_d']):
            rs = [int(s) for s in result.get_shape()]
            result1 = tf.slice(result,[0,0,0,0],[config['batch_size'], rs[1],rs[2],3])
            result2 = tf.slice(result,[0,0,0,3],[config['batch_size'], rs[1],rs[2],1])
            result1 = config['g_last_layer'](result1)
            result2 = batch_norm(config['batch_size'], name='g_bn_relu_f')(result2)
            result2 = tf.nn.relu(result2)
            result = tf.concat(3, [result1, result2])
        elif(config['g_last_layer']):
            result = config['g_last_layer'](result)

        print("RETURN")
        return result,z_dim_random_uniform

def discriminator(config, x, f,z,g,gz):
    x_dims = config['x_dims']
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    channels = (config['channels'])
    # combine to one batch, per Ian's "Improved GAN"
    x = tf.concat(0, [x,g])

    # careful on order.  See https://arxiv.org/pdf/1606.00704v1.pdf
    z = tf.concat(0, [z, gz])
    x = tf.reshape(x, [batch_size, -1, channels])
    if(config['d_add_noise']):
        x += tf.random_normal(x.get_shape(), mean=0, stddev=config['d_noise'], dtype=config['dtype'])

    if(config['include_f_in_d']):
        channels+=1
        x_tmp = tf.reshape(x, [single_batch_size, -1, channels-1])
        f = build_reshape(int(x_tmp.get_shape()[1]), [f], config['d_project'], single_batch_size, config['dtype'])
        f = tf.reshape(f, [single_batch_size, -1, 1])
        x = tf.concat(2, [x_tmp, f])
        x = tf.reshape(x, g.get_shape())


    if(config['latent_loss']):
        orig_x = x
        x = build_reshape(int(x.get_shape()[1]), [z], config['d_project'], batch_size, config['dtype'])
        x = tf.reshape(x, [batch_size, -1, 1])
        x = tf.concat(2, [x, tf.reshape(orig_x, [batch_size, -1, channels])])
        x = tf.reshape(x,[batch_size, x_dims[0], x_dims[1], channels+1])
    else:
        x = tf.reshape(x,[batch_size, x_dims[0], x_dims[1], channels])



    if(config['d_architecture']=='wide_resnet'):
        result = discriminator_wide_resnet(config,x)
    elif(config['d_architecture']=='densenet'):
        result = discriminator_densenet(config,x)
    else:
        result = discriminator_vanilla(config,x)

    minis = get_minibatch_features(config, result, batch_size,config['dtype'])
    result = tf.concat(1, [result]+minis)

    #result = tf.nn.dropout(result, 0.7)
    print('before linear layer', result)
    if(config['d_linear_layer']):
        result = linear(result, config['d_linear_layers'], scope="d_linear_layer")
        #TODO batch norm?
        if(config['d_batch_norm']):
            result = batch_norm(config['batch_size'], name='d_bn_lin_proj')(result)
        result = config['d_activation'](result)

    last_layer = result
    last_layer = tf.reshape(last_layer, [batch_size, -1])
    last_layer = tf.slice(last_layer, [single_batch_size, 0], [single_batch_size, -1])

    print('last layer size', result)
    result = linear(result, config['y_dims']+1, scope="d_proj")

    def build_logits(class_logits, num_classes):

        generated_class_logits = tf.squeeze(tf.slice(class_logits, [0, num_classes - 1], [batch_size, 1]))
        positive_class_logits = tf.slice(class_logits, [0, 0], [batch_size, num_classes - 1])

        """
        # make these a separate matmul with weights initialized to 0, attached only to generated_class_logits, or things explode
        generated_class_logits = tf.squeeze(generated_class_logits) + tf.squeeze(linear(diff_feat, 1, stddev=0., scope="d_indivi_logits_from_diff_feat"))
        assert len(generated_class_logits.get_shape()) == 1
        # re-assemble the logits after incrementing the generated class logits
        class_logits = tf.concat(1, [positive_class_logits, tf.expand_dims(generated_class_logits, 1)])
        """

        mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
        safe_pos_class_logits = positive_class_logits - mx

        gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
        assert len(gan_logits.get_shape()) == 1

        return class_logits, gan_logits
    num_classes = config['y_dims']+1
    class_logits, gan_logits = build_logits(result, num_classes)
    return [tf.slice(class_logits, [0, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [0], [single_batch_size]),
                tf.slice(class_logits, [single_batch_size, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [single_batch_size], [single_batch_size]), 
                last_layer]


def discriminator_densenet(config, x):
    activation = config['d_activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_densenet_layers']
    depth = config['d_densenet_block_depth']
    k = config['d_densenet_k']
    result = x
    result = conv2d(result, 16, name='d_expand', k_w=3, k_h=3, d_h=1, d_w=1)
    for i in range(layers):
        if i != layers-1:
            print("transition")
            result = dense_block(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i))
        else:
            print("no transition")
        for j in range(depth):
            result = dense_block(result, k, activation, batch_size, 'layer', 'd_layers_'+str(i)+"_"+str(j))
            print("resnet size", result)


    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
    print("RESULT SIZE IS", result)
    result = tf.reshape(result, [batch_size, -1])

    return result

def discriminator_wide_resnet(config, x):
    activation = config['d_activation']
    batch_size = int(x.get_shape()[0])
    layers = config['d_wide_resnet_depth']
    result = x
    #result = build_conv_tower(result, config['conv_d_layers'][:1], config['d_pre_res_filter'], config['batch_size'], config['d_batch_norm'], True, 'd_', config['d_activation'], stride=config['d_pre_res_stride'])

    #result = activation(result)
    result = conv2d(result, layers[0], name='d_expand1a', k_w=3, k_h=3, d_h=1, d_w=1)
    result = batch_norm(config['batch_size'], name='d_expand_bn1a')(result)
    result = activation(result)
    result = conv2d(result, layers[0], name='d_expand1b', k_w=1, k_h=1, d_h=1, d_w=1)
    #result = residual_block(result, activation, batch_size, 'widen', 'd_layers_0')
    #result = residual_block(result, activation, batch_size, 'identity', 'd_layers_1')
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_2')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_3')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_4')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_5')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_6')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_7')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_8')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_9')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'conv', 'd_layers_10')
    print("DRESULT", result)
    result = residual_block(result, activation, batch_size, 'identity', 'd_layers_11')
    print("DRESULT", result)
    #result = residual_block(result, stride=1, 'conv')
    #result = residual_block(result, stride=1, 'identity')
    #result = residual_block(result, stride=1,  'conv')
    #result = residual_block(result, stride=1,  'identity')
    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
    print("RESULT SIZE IS", result)
    result = tf.reshape(result, [batch_size, -1])

    return result


def discriminator_vanilla(config, x):
    x_dims = config['x_dims']
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    channels = (config['channels'])

    result = x
    if config['conv_d_layers']:
        result = build_conv_tower(result, config['conv_d_layers'][:1], config['d_pre_res_filter'], config['batch_size'], config['d_batch_norm'], True, 'd_', config['d_activation'], stride=config['d_pre_res_stride'])
        if(config['d_pool']):
            result = tf.nn.max_pool(result, [1, 3, 3, 1], [1, 2,2,1],padding='SAME')
        result = config['d_activation'](result)
        result = build_resnet(result, config['d_resnet_depth'], config['d_resnet_filter'], 'd_conv_res_', config['d_activation'], config['batch_size'], config['d_batch_norm'], conv=True)
        result = build_conv_tower(result, config['conv_d_layers'][1:], config['d_conv_size'], config['batch_size'], config['d_batch_norm'], config['d_batch_norm_last_layer'], 'd_2_', config['d_activation'])
        result = tf.reshape(result, [batch_size, -1])

    return result

def z_from_f(config, f, categories):
    batch_size = config["batch_size"]
    transfer_fct = config['transfer_fct']
    n_z = int(config['z_dim'])
    n_c = sum(config['categories'])

    result = f
    print("RESULT IS", result)
    if(config['f_skip_fc']):
        pass
    else:
        result = tf.reshape(result, [config['batch_size'], 2048])
        result = linear(result, config['f_hidden_1'], scope="v_f_hidden")
        result = batch_norm(config['batch_size'], name='v_f_hidden_bn')(result)
        result = transfer_fct(result)
        result = linear(result, config['f_hidden_2'], scope="v_f_hidden2")
        result = batch_norm(config['batch_size'], name='v_f_hidden_bn2')(result)
        result = transfer_fct(result)
    last_layer = result
    result = linear(result, n_z, scope="v_f_hidden3")
    result = batch_norm(config['batch_size'], name='v_f_hidden_bn3')(result)
    result = transfer_fct(result)

    result = tf.reshape(result, [config['batch_size'], -1])

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=config['dtype']))
    out_mean= tf.get_variable('v_out_mean', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    mu = tf.add(tf.matmul(result, out_mean),b_out_mean)

    out_log_sigma=tf.get_variable('v_out_logsigma', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    b_out_log_sigma= tf.get_variable('v_b_out_logsigma', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    sigma = tf.add(tf.matmul(result, out_log_sigma),b_out_log_sigma)

    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=config['dtype'])

    set_tensor('eps', eps)
    z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(sigma)), eps))
    e_z = tf.random_normal([config['batch_size'], n_z], mu, tf.exp(sigma), dtype=config['dtype'])

    if config['category_loss']:
        e_c = linear(e_z,n_c, 'v_ez_lin')
        e_c = [tf.nn.softmax(x) for x in split_categories(e_c, config['batch_size'], categories)]
    else:
        e_c = []




    if(config['e_last_layer']):
        z = config['e_last_layer'](z)
        e_z = config['e_last_layer'](e_z)
    return e_z, e_c, z, mu, sigma

def approximate_z(config, x, y):
    y = tf.concat(1, y)
    x_dims = config['x_dims']
    batch_size = config["batch_size"]
    transfer_fct = config['transfer_fct']
    x = tf.reshape(x, [config["batch_size"], -1,config['channels']])
    noise_dims = int(x.get_shape()[1])-int(y.get_shape()[1])
    n_z = int(config['z_dim'])
    channels = (config['channels']+1)

    result = build_reshape(int(x.get_shape()[1]), [y], config['d_project'], batch_size, config['dtype'])
    result = tf.reshape(result, [batch_size, -1, 1])
    result = tf.concat(2, [result, x])

    result = tf.reshape(result, [config["batch_size"], x_dims[0],x_dims[1],channels])

    if config['g_encode_layers']:
        result = build_conv_tower(result, 
                    config['g_encode_layers'], 
                    config['e_conv_size'], 
                    config['batch_size'], 
                    config['e_batch_norm'], 
                    config['e_batch_norm_last_layer'], 
                    'v_', 
                    transfer_fct
                    )

    result = transfer_fct(result)
    last_layer = result
    result = tf.reshape(result, [config['batch_size'], -1])

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    out_mean= tf.get_variable('v_out_mean', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    mu = tf.add(tf.matmul(result, out_mean),b_out_mean)

    out_log_sigma=tf.get_variable('v_out_logsigma', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer(dtype=config['dtype']), dtype=config['dtype'])
    b_out_log_sigma= tf.get_variable('v_b_out_logsigma', initializer=tf.zeros([n_z], dtype=config['dtype']), dtype=config['dtype'])
    sigma = tf.add(tf.matmul(result, out_log_sigma),b_out_log_sigma)

    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=config['dtype'])
    set_tensor('eps', eps)

    z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(sigma)), eps))

    e_z = tf.random_normal([config['batch_size'], n_z], mu, tf.exp(sigma), dtype=config['dtype'])

    if(config['e_last_layer']):
        z = config['e_last_layer'](z)
        e_z = config['e_last_layer'](e_z)
    return e_z, z, mu, sigma
def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy


def split_categories(layer, batch_size, categories):
    start = 0
    ret = []
    for category in categories:
        count = int(category.get_shape()[1])
        ret.append(tf.slice(layer, [0, start], [batch_size, count]))
        start += count
    return ret


def categories_loss(categories, layer, batch_size):
    loss = 0
    def split(layer):
        start = 0
        ret = []
        for category in categories:
            count = int(category.get_shape()[1])
            ret.append(tf.slice(layer, [0, start], [batch_size, count]))
            start += count
        return ret
            
    for category,layer_s in zip(categories, split(layer)):
        size = int(category.get_shape()[1])
        #TOdO compute loss
        category_prior = tf.ones([batch_size, size])*np.float32(1./size)
        logli_prior = tf.reduce_sum(tf.log(category_prior + TINY) * category, reduction_indices=1)
        layer_softmax = tf.nn.softmax(layer_s)
        logli = tf.reduce_sum(tf.log(layer_softmax+TINY)*category, reduction_indices=1)
        disc_ent = tf.reduce_mean(-logli_prior)
        disc_cross_ent =  tf.reduce_mean(-logli)

        loss += disc_ent - disc_cross_ent
    return loss

def random_category(batch_size, size, dtype):
    prior = tf.ones([batch_size, size])*1./size
    dist = tf.log(prior + TINY)
    with tf.device('/cpu:0'):
        sample=tf.multinomial(dist, num_samples=1)[:, 0]
        return tf.one_hot(sample, size, dtype=dtype)

def create(config, x,y,f):
    set_ops_dtype(config['dtype'])
    batch_size = config["batch_size"]
    z_dim = int(config['z_dim'])

    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = tf.concat(1, categories)
        #categories_t = [tf.tile(categories_t, [config['batch_size'], 1])]
    else:
        categories_t = []

    if(config['pretrained_model'] == 'preprocess'):
        #img = tf.reshape(x, [config['batch_size'], -1])
        #img = build_reshape(224*224*config['channels'], [img], 'zeros', config['batch_size'])
        #img = tf.reshape(img, [config['batch_size'],224,224,config['channels']])
        #print("IMG:", img)

        #f = vggnet_loader.create_graph(img, 'pool4:0')[0]
        #f = tf.reshape(f, [config['batch_size'], -1])
        if(config['latent_loss']):
            encoded_z, encoded_c, z, z_mu, z_sigma = z_from_f(config, f, categories)
        else:
            encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
            z_mu = None
            z_sigma = None
            z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])

    elif(config['latent_loss']):
        encoded_z, z, z_mu, z_sigma = approximate_z(config, x, [y])
    else:
        encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
        z_mu = None
        z_sigma = None
        z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])


    print("Z IS ", z)
    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = [tf.concat(1, categories)]
    else:
        categories_t = []

    g,z_dim_random_uniform = generator(config, [y, z]+categories_t)
    #g = generator(config, [y, z]+categories_t)
    set_tensor('z_dim_random_uniform', z_dim_random_uniform)
    with tf.device('/cpu:0'):
    #    print_z = tf.Print(z, [tf.reduce_mean(z), y, get_tensor("z_dim_random_uniform")], message="z is")
        print_z = tf.Print(z, [tf.reduce_mean(z), y], message="z is")
    print('-+',y,encoded_z,categories_t, [y, encoded_z]+categories_t)

    encoded,_ = generator(config, [y, encoded_z]+categories_t, reuse=True)
    #encoded = generator(config, [y, encoded_z]+categories_t, reuse=True)

    def discard_layer(sample):
        sample = tf.reshape(sample, [config['batch_size'],-1,config['channels']+1])
        sample = tf.slice(sample, [0,0,0],[int(sample.get_shape()[0]),int(sample.get_shape()[1]),config['channels']])
        sample = tf.reshape(sample, [config['batch_size'], int(x.get_shape()[1]), int(x.get_shape()[2]), config['channels']])
        return sample
    if(config['include_f_in_d']):
        encoded = discard_layer(encoded)
        g_sample = discard_layer(g)
    else:
        g_sample = g
    #print("shape of z,encoded_z", z.get_shape(), encoded_z.get_shape())
    d_real, d_real_sig, d_fake, d_fake_sig, d_last_layer = discriminator(config,x, f, encoded_z, g, z)

    if(config['latent_loss']):
        latent_loss = -config['latent_lambda'] * tf.reduce_mean(1 + z_sigma
                                       - tf.square(z_mu)
                                       - tf.exp(z_sigma), 1)

    else:
        latent_loss = None
    np_fake = np.array([0]*config['y_dims']+[1])
    fake_symbol = tf.tile(tf.constant(np_fake, dtype=config['dtype']), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],config['y_dims']+1])

    #real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])
    real_symbols = y


    zeros = tf.zeros_like(d_fake_sig, dtype=config['dtype'])
    ones = tf.zeros_like(d_real_sig, dtype=config['dtype'])

    #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real, ones)

    generator_target_prob = config['g_target_prob']
    d_label_smooth = config['d_label_smooth']
    d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake_sig, zeros)
    #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real_sig, ones)
    d_real_loss = sigmoid_kl_with_logits(d_real_sig, 1.-d_label_smooth)
    #if(config['adv_loss']):
    #    d_real_loss +=  sigmoid_kl_with_logits(d_fake_sig, d_label_smooth)

    d_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_real,real_symbols)

    g_loss= sigmoid_kl_with_logits(d_fake_sig, generator_target_prob)
    simple_g_loss = g_loss
    if(config['adv_loss']):
        g_loss+= sigmoid_kl_with_logits(d_real_sig, d_label_smooth)

    g_loss_fake= tf.nn.sigmoid_cross_entropy_with_logits(d_real_sig, zeros)
    g_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)

    #g_loss_encoder = tf.nn.sigmoid_cross_entropy_with_logits(d_real, zeros)
    #d_real = tf.nn.sigmoid(d_real)
    #d_fake = tf.nn.sigmoid(d_fake)
    #d_fake_loss = -tf.log(1-d_fake+TINY)
    #d_real_loss = -tf.log(d_real+TINY)
    #g_loss_softmax = -tf.log(1-d_real+TINY)
    #g_loss_encoder = -tf.log(d_fake+TINY)
    g_loss = tf.reduce_mean(g_loss)

    if(config['g_class_loss']):
        g_loss+=config['g_class_lambda']*tf.reduce_mean(g_class_loss)

    d_loss = tf.reduce_mean(d_fake_loss) + \
            tf.reduce_mean(d_real_loss)

    if(int(y.get_shape()[1]) > 1):
        print("ADDING D_CLASS_LOSS")
        d_loss += tf.reduce_mean(d_class_loss)
    else:
        print("REMOVING D_CLASS_LOSS")

    if(config['latent_loss']):
        g_loss += tf.reduce_mean(latent_loss)
        #d_loss += tf.reduce_mean(latent_loss)

    if(config['d_fake_class_loss']):
        d_fake_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake,fake_symbol)
        d_loss += config['g_class_lambda']*tf.reduce_mean(d_fake_class_loss)


    categories_l = None
    if config['category_loss']:

        category_layer = linear(d_last_layer, sum(config['categories']), 'v_categories',stddev=0.15)
        category_layer = batch_norm(config['batch_size'], name='v_cat_loss')(category_layer)
        category_layer = config['g_activation'](category_layer)
        categories_l = categories_loss(categories, category_layer, config['batch_size'])
        g_loss -= config['categories_lambda']*categories_l
        d_loss -= config['categories_lambda']*categories_l

    if config['regularize']:
        ws = None
        with tf.variable_scope("generator"):
            with tf.variable_scope("g_lin_proj"):
                tf.get_variable_scope().reuse_variables()
                ws = tf.get_variable('Matrix',dtype=config['dtype'])
                tf.get_variable_scope().reuse_variables()
            lam = config['regularize_lambda']
            print("ADDING REG", lam, ws)
            g_loss += lam*tf.nn.l2_loss(ws)


    if(config['latent_loss']):
        mse_loss = tf.reduce_max(tf.square(x-encoded))
    else:
        mse_loss = None
    if config['mse_loss']:
        mse_lam = config['mse_lambda']
        g_loss += mse_lam * mse_loss

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    v_vars = [var for var in tf.trainable_variables() if 'v_' in var.name]
    if(config['v_train'] == 'generator'):
        g_vars += v_vars
    elif(config['v_train'] == 'discriminator'):
        d_vars += v_vars
    elif(config['v_train'] == 'both'):
        g_vars += v_vars
        d_vars += v_vars
    else:
        print("ERROR: No variables training z space")

    if(config['optimizer'] == 'simple'):
        d_lr = np.float32(config['simple_lr'])
        g_mul = np.float32(config['simple_lr_g'])
        g_lr = d_lr*g_mul
        g_optimizer = tf.train.GradientDescentOptimizer(g_lr).minimize(g_loss, var_list=g_vars)
        d_optimizer = tf.train.GradientDescentOptimizer(d_lr).minimize(d_loss, var_list=d_vars)
    elif(config['optimizer'] == 'adam'):
        g_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(g_loss, var_list=g_vars)
        lr = np.float32(config['d_learning_rate'])
        set_tensor("lr_value", lr)
        lr = tf.get_variable('lr', [], trainable=False, initializer=tf.constant_initializer(lr,dtype=config['dtype']),dtype=config['dtype'])
        set_tensor('lr', lr)
        d_optimizer = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)
        
    elif(config['optimizer'] == 'momentum'):
        d_lr = np.float32(config['momentum_lr'])
        g_mul = np.float32(config['momentum_lr_g'])
        g_lr = d_lr*g_mul
        moment = config['momentum']
        g_optimizer = tf.train.MomentumOptimizer(g_lr, moment).minimize(g_loss, var_list=g_vars)
        d_optimizer = tf.train.MomentumOptimizer(d_lr, moment).minimize(d_loss, var_list=d_vars)
    elif(config['optimizer'] == 'rmsprop'):
        lr = np.float32(config['rmsprop_lr'])
        set_tensor("lr_value", lr)
        lr = tf.placeholder(config['dtype'], shape=[])
        set_tensor('lr', lr)

        d_optimizer = tf.train.RMSPropOptimizer(lr).minimize(d_loss, var_list=d_vars)

    if(config['d_optim_strategy'] == 'adam'):
        d_optimizer = tf.train.AdamOptimizer(np.float32(config['d_learning_rate'])).minimize(d_loss, var_list=d_vars)
    elif(config['d_optim_strategy'] == 'g_adam'):
        g_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate']))

        gvs = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        g_optimizer = g_optimizer.apply_gradients(capped_gvs)

    elif(config['d_optim_strategy'] == 'g_rmsprop'):
        lr = np.float32(config['rmsprop_lr']) * np.float32(config['rmsprop_lr_g'])
        g_optimizer = tf.train.RMSPropOptimizer(lr).minimize(g_loss, var_list=g_vars)

    elif(config['d_optim_strategy'] == 'g_momentum'):
        d_lr = np.float32(config['momentum_lr'])
        g_mul = np.float32(config['momentum_lr_g'])
        g_lr = d_lr*g_mul
        moment = config['momentum']
        g_optimizer = tf.train.MomentumOptimizer(g_lr, moment).minimize(g_loss, var_list=g_vars)

    if(config['mse_loss']):
        mse_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(mse_loss, var_list=tf.trainable_variables())
    else:
        mse_optimizer = None

    summary = tf.all_variables()
    def summary_reduce(s):
        if(len(s.get_shape())==0):
            return s
        while(len(s.get_shape())>1):
            s=tf.reduce_mean(s,1)
            s=tf.squeeze(s)
        return tf.reduce_mean(s,0)

    summary = [(s.get_shape(), s.name, s.dtype, summary_reduce(s)) for s in summary]
    set_tensor("hc_summary",summary)

    set_tensor('categories', categories_t)
    if(config['category_loss']):
        set_tensor('categories_loss', config['categories_lambda']*categories_l)
    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor("z", z)
    set_tensor("f", f)
    set_tensor("print_z", print_z)
    set_tensor("g_loss", g_loss)
    set_tensor("d_loss", d_loss)
    set_tensor("g_optimizer", g_optimizer)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("mse_optimizer", mse_optimizer)
    set_tensor("g", g_sample)
    set_tensor("encoded", encoded)
    set_tensor('encoded_z', encoded_z)
    set_tensor("encoder_mse", mse_loss)
    set_tensor("d_real", tf.reduce_mean(d_real))
    set_tensor("d_fake", tf.reduce_mean(d_fake))
    set_tensor("d_fake_loss", tf.reduce_mean(d_fake_loss))
    set_tensor("d_real_loss", tf.reduce_mean(d_real_loss))
    set_tensor("d_class_loss", tf.reduce_mean(d_class_loss))
    set_tensor("g_class_loss", tf.reduce_mean(g_class_loss))
    set_tensor("d_fake_sigmoid", tf.sigmoid(d_fake_sig))
    set_tensor("d_fake_sig", tf.reduce_mean(tf.sigmoid(d_fake_sig)))
    set_tensor("d_real_sig", tf.reduce_mean(tf.sigmoid(d_real_sig)))
    set_tensor("g_loss_sig", tf.reduce_mean(simple_g_loss))
    if(config['latent_loss']):
        set_tensor('latent_loss', tf.reduce_mean(latent_loss))

iteration = 0
def train(sess, config):
    x_t = get_tensor('x')
    g_t = get_tensor('g')
    g_loss = get_tensor("g_loss_sig")
    d_loss = get_tensor("d_loss")
    d_fake_loss = get_tensor('d_fake_loss')
    d_real_loss = get_tensor('d_real_loss')
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")
    d_class_loss = get_tensor("d_class_loss")
    g_class_loss = get_tensor("g_class_loss")
    mse_optimizer = get_tensor("mse_optimizer")
    lr = get_tensor('lr')
    lr_value = get_tensor('lr_value')#todo: not actually a tensor
    #encoder_mse = get_tensor("encoder_mse")
    #categories_l = get_tensor("categories_loss")
    #latent_l = get_tensor("latent_loss")
    _, d_cost = sess.run([d_optimizer, d_loss], feed_dict={lr:lr_value})
    _, g_cost,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, d_fake_loss, d_real_loss, d_class_loss])
    print("%2d: d_lr %.1e g cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (iteration, lr_value, g_cost,d_fake, d_real, d_class ))

    slowdown = 1
    bounds_max = config['bounds_d_fake_max']
    bounds_min = config['bounds_d_fake_min']
    bounds_slow = config['bounds_d_fake_slowdown']
    max_lr = config['rmsprop_lr']
    if(d_fake < bounds_min):
        slowdown = 1/(bounds_slow*config['bounds_step'])
    elif(d_fake > bounds_max):
        slowdown = 1
    else:
        percent = 1 - (d_fake - bounds_min)/(bounds_max-bounds_min)
        slowdown = 1/(percent * bounds_slow + TINY)
        if(slowdown > 1):
            slowdown=1
    new_lr = max_lr*slowdown
    set_tensor("lr_value", new_lr)

    global iteration
    iteration+=1
    #print("X mean %.2f max %.2f min %.2f" % (np.mean(x), np.max(x), np.min(x)))
    #print("G mean %.2f max %.2f min %.2f" % (np.mean(g), np.max(g), np.min(g)))
    #print("Categories loss %.6f" % categories_r)

    return d_cost, g_cost

def test(sess, config):
    x = get_tensor("x")
    y = get_tensor("y")
    d_fake = get_tensor("d_fake")
    d_real = get_tensor("d_real")
    g_loss = get_tensor("g_loss")

    g_cost, d_fake_cost, d_real_cost = sess.run([g_loss, d_fake, d_real])


    #hc.event(costs, sample_image = sample[0])

    #print("test g_loss %.2f d_fake %.2f d_loss %.2f" % (g_cost, d_fake_cost, d_real_cost))
    return g_cost,d_fake_cost, d_real_cost,0


