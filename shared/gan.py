
from shared.ops import *
from shared.util import *
from shared.hc_tf import *
import tensorflow as tf
TINY = 1e-12

def generator(config, inputs, reuse=False):
    x_dims = config['x_dims']
    with(tf.variable_scope("generator", reuse=reuse)):
        output_shape = x_dims[0]*x_dims[1]*config['channels']
        primes = find_smallest_prime(x_dims[0], x_dims[1])
        z_proj_dims = int(config['conv_g_layers'][0])

        result = tf.concat(1, inputs)
        result = linear(result, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj")
        result = batch_norm(config['batch_size'], name='g_bn_lin_proj')(result)
        result = config['g_activation'](result)
        #result = build_reshape(z_proj_dims*primes[0]*primes[1], [y,z], 'tiled', config['batch_size'])

        result = tf.reshape(result,[config['batch_size'], primes[0], primes[1], z_proj_dims])

        if config['conv_g_layers']:
            result = build_deconv_tower(result, config['conv_g_layers'][1:2], x_dims, config['conv_size'], 'g_conv_', config['g_activation'], config['g_batch_norm'], True, config['batch_size'], config['g_last_layer_stddev'])
            result = build_resnet(result, config['g_resnet_depth'], config['g_resnet_filter'], 'g_conv_res_', config['g_activation'], config['batch_size'], config['g_batch_norm'])
            result = build_deconv_tower(result, config['conv_g_layers'][2:], x_dims, config['conv_size'], 'g_conv_2', config['g_activation'], config['g_batch_norm'], config['g_batch_norm_last_layer'], config['batch_size'], config['g_last_layer_stddev'])

        if(config['g_last_layer']):
            result = config['g_last_layer'](result)
        return result

def discriminator(config, x, z,g,gz, reuse=False):
    x_dims = config['x_dims']
    if(reuse):
        tf.get_variable_scope().reuse_variables()
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    x = tf.concat(0, [x,g])
    z = tf.concat(0, [z,gz])
    x = tf.reshape(x, [batch_size, -1, config['channels']])
    if(config['d_add_noise']):
        x += tf.random_normal(x.get_shape(), mean=0, stddev=0.1)

    channels = (config['channels']+1)
    #result = x

    result = build_reshape(int(x.get_shape()[1]), [z], config['d_project'], batch_size)
    result = tf.reshape(result, [batch_size, -1, 1])
    result = tf.concat(2, [result, tf.reshape(x, [batch_size, -1, channels-1])])
    result = tf.reshape(result,[batch_size, x_dims[0], x_dims[1], channels])

    if config['conv_d_layers']:
        result = build_conv_tower(result, config['conv_d_layers'][:2], config['d_conv_size'], config['batch_size'], config['d_batch_norm'], True, 'd_', config['d_activation'])
        print("RESULT",result)
        result = build_resnet(result, config['d_resnet_depth'], config['d_resnet_filter'], 'd_conv_res_', config['d_activation'], config['batch_size'], config['d_batch_norm'], conv=True)
        result = build_conv_tower(result, config['conv_d_layers'][2:], config['d_conv_size'], config['batch_size'], config['d_batch_norm'], config['d_batch_norm_last_layer'], 'd_2_', config['d_activation'])
        result = tf.reshape(x, [batch_size, -1])

    def get_minibatch_features(h):
        n_kernels = int(config['d_kernels'])
        dim_per_kernel = int(config['d_kernel_dims'])
        x = linear(h, n_kernels * dim_per_kernel, scope="d_h")
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation,3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask
        def half(tens, second):
            m, n, _ = tens.get_shape()
            m = int(m)
            n = int(n)
            return tf.slice(tens, [0, 0, second * single_batch_size], [m, n, single_batch_size])
        # TODO: speedup by allocating the denominator directly instead of constructing it by sum
        #       (current version makes it easier to play with the mask and not need to rederive
        #        the denominator)
        f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
        f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

        return [f1, f2]
    minis = get_minibatch_features(result)
    g_proj = tf.concat(1, [result]+minis)

    #result = tf.nn.dropout(result, 0.7)
    if(config['d_linear_layer']):
        result = linear(result, config['d_linear_layers'], scope="d_linear_layer")
        #TODO batch norm?
        if(config['d_batch_norm']):
            result = batch_norm(config['batch_size'], name='d_bn_lin_proj')(result)
        result = config['d_activation'](result)

    last_layer = result
    last_layer = tf.reshape(last_layer, [batch_size, -1])
    last_layer = tf.slice(last_layer, [single_batch_size, 0], [single_batch_size, -1])

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
    num_classes = config['y_dims'] +1
    class_logits, gan_logits = build_logits(result, num_classes)
    return [tf.slice(class_logits, [0, 0], [single_batch_size, num_classes]),
                tf.slice(gan_logits, [0], [single_batch_size]),
                tf.slice(class_logits, [single_batch_size, 0], [single_batch_size, num_classes]),
                tf.slice(gan_logits, [single_batch_size], [single_batch_size]), 
                last_layer]


def approximate_z(config, x, y):
    y = tf.concat(1, y)
    x_dims = config['x_dims']
    batch_size = config["batch_size"]
    transfer_fct = config['transfer_fct']
    x = tf.reshape(x, [config["batch_size"], -1,config['channels']])
    noise_dims = int(x.get_shape()[1])-int(y.get_shape()[1])
    n_z = int(config['z_dim'])
    channels = (config['channels']+1)

    result = build_reshape(int(x.get_shape()[1]), [y], config['d_project'], batch_size)
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

    b_out_mean= tf.get_variable('v_b_out_mean', initializer=tf.zeros([n_z], dtype=tf.float32))
    out_mean= tf.get_variable('v_out_mean', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer())
    mu = tf.add(tf.matmul(result, out_mean),b_out_mean)

    out_log_sigma=tf.get_variable('v_out_logsigma', [result.get_shape()[1], n_z], initializer=tf.contrib.layers.xavier_initializer())
    b_out_log_sigma= tf.get_variable('v_b_out_logsigma', initializer=tf.zeros([n_z], dtype=tf.float32))
    sigma = tf.add(tf.matmul(result, out_log_sigma),b_out_log_sigma)

    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=tf.float32)

    z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(sigma)), eps))

    e_z = tf.random_normal([config['batch_size'], n_z], mu, tf.exp(sigma), dtype=tf.float32)

    if(config['e_last_layer']):
        z = config['e_last_layer'](z)
        e_z = config['e_last_layer'](e_z)
    return e_z, z, mu, sigma, last_layer
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

def random_category(batch_size, size):
    prior = tf.ones([batch_size, size])*1./size
    dist = tf.log(prior + TINY)
    sample=tf.multinomial(dist, num_samples=1)[:, 0]
    return tf.one_hot(sample, size)

def create(config, x,y):
    batch_size = config["batch_size"]
    z_dim = int(config['z_dim'])

    #categories = [random_category(config['batch_size'], size) for size in config['categories']]
    #categories_t = tf.concat(1, categories)
    if(config['latent_loss']):
        encoded_z, z, z_mu, z_sigma, e_last_layer = approximate_z(config, x, [y])
    else:
        encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1)
        z_mu = None
        z_sigma = None
        e_last_layer = None
        z = tf.random_uniform([config['batch_size'], z_dim],-1, 1)


    categories = [random_category(1, size) for size in config['categories']]
    categories_t = tf.concat(1, categories)
    categories_t = tf.tile(categories_t, [config['batch_size'], 1])

    print("Build generator")
    g = generator(config, [y, z]+[categories_t])
    encoded = None
    if(config['latent_loss']):
        encoded = generator(config, [y, encoded_z]+[categories_t], reuse=True)
    print("shape of g,x", g.get_shape(), x.get_shape())
    #print("shape of z,encoded_z", z.get_shape(), encoded_z.get_shape())
    d_real, d_real_sig, d_fake, d_fake_sig, d_last_layer = discriminator(config,x, encoded_z, g, z, reuse=False)

    if(config['latent_loss']):
        latent_loss = -config['latent_lambda'] * tf.reduce_mean(1 + z_sigma
                                       - tf.square(z_mu)
                                       - tf.exp(z_sigma), 1)

    else:
        latent_loss = None
    np_fake = np.array([0]*config['y_dims']+[1])
    fake_symbol = tf.tile(tf.constant(np_fake, dtype=tf.float32), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],config['y_dims']+1])

    real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])
    #real_symbols = y


    if(config['loss'] == 'softmax'):
        d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
        d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

        g_loss= tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)

    else:
        zeros = tf.zeros_like(d_fake_sig)
        ones = tf.zeros_like(d_real_sig)

        #d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, zeros)
        #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real, ones)

        generator_target_prob = config['g_target_prob']
        d_label_smooth = config['d_label_smooth']
        d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake_sig, zeros)
        #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real_sig, ones)
        d_real_loss = sigmoid_kl_with_logits(d_real_sig, 1.-d_label_smooth)
        if(config['adv_loss']):
            d_real_loss +=  sigmoid_kl_with_logits(d_fake_sig, d_label_smooth)

        d_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_real,real_symbols)
        d_fake_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake,fake_symbol)

        g_loss= sigmoid_kl_with_logits(d_fake_sig, generator_target_prob)
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
            tf.reduce_mean(d_real_loss) + \
            tf.reduce_mean(d_class_loss)

    if(config['latent_loss']):
        g_loss += tf.reduce_mean(latent_loss)
        d_loss += tf.reduce_mean(latent_loss)

    if(config['d_fake_class_loss']):
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
            with tf.variable_scope("g_conv_0"):
                tf.get_variable_scope().reuse_variables()
                ws = tf.get_variable('w')
                tf.get_variable_scope().reuse_variables()
                b = tf.get_variable('biases')
            lam = config['regularize_lambda']
            g_loss += lam*tf.nn.l2_loss(ws)+lam*tf.nn.l2_loss(b)


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
    g_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.AdamOptimizer(np.float32(config['d_learning_rate'])).minimize(d_loss, var_list=d_vars)

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
    set_tensor("g_loss", g_loss)
    set_tensor("d_loss", d_loss)
    set_tensor("g_optimizer", g_optimizer)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("mse_optimizer", mse_optimizer)
    set_tensor("g", g)
    set_tensor("encoded", encoded)
    set_tensor("encoder_mse", mse_loss)
    set_tensor("d_fake", tf.reduce_mean(d_fake))
    set_tensor("d_real", tf.reduce_mean(d_real))
    set_tensor("d_fake_loss", tf.reduce_mean(d_fake_loss))
    set_tensor("d_real_loss", tf.reduce_mean(d_real_loss))
    set_tensor("d_class_loss", tf.reduce_mean(d_class_loss))
    set_tensor("g_class_loss", tf.reduce_mean(g_class_loss))
    set_tensor("d_fake_sigmoid", tf.sigmoid(d_fake_sig))
    set_tensor("d_loss", tf.reduce_mean(d_real_loss))
    if(config['latent_loss']):
        set_tensor('latent_loss', tf.reduce_mean(latent_loss))

def train(sess, config):
    x = get_tensor('x')
    g = get_tensor('g')
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    d_fake_loss = get_tensor('d_fake_loss')
    d_real_loss = get_tensor('d_real_loss')
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")
    d_class_loss = get_tensor("d_class_loss")
    g_class_loss = get_tensor("g_class_loss")
    mse_optimizer = get_tensor("mse_optimizer")
    #encoder_mse = get_tensor("encoder_mse")
    #categories_l = get_tensor("categories_loss")
    #latent_l = get_tensor("latent_loss")
    _, d_cost = sess.run([d_optimizer, d_loss])
    _, g_cost, x, g,d_fake,d_real,d_class = sess.run([g_optimizer, g_loss, x, g, d_fake_loss, d_real_loss, d_class_loss])
    #_ = sess.run([mse_optimizer])

    print("g cost %.2f d cost %.2f d_fake %.2f d_real %.2f d_class %.2f" % (g_cost, d_cost,d_fake, d_real, d_class ))
    print("X mean %.2f max %.2f min %.2f" % (np.mean(x), np.max(x), np.min(x)))
    print("G mean %.2f max %.2f min %.2f" % (np.mean(g), np.max(g), np.min(g)))
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


