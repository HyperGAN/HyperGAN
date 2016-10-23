
from lib.util.ops import *
from lib.util.globals import *
from lib.util.hc_tf import *
import tensorflow as tf
import lib.util.wavegan as wavegan
TINY = 1e-12

def generator(config, inputs, reuse=False):
    x_dims = config['x_dims']
    output_channels = config['channels']
    activation = config['generator.activation']
    batch_size = config['batch_size']
    z_proj_dims = int(config['generator.z_projection_depth'])

    if(config['include_f_in_d'] == True):
        output_channels += 1
    with(tf.variable_scope("generator", reuse=reuse)):
        output_shape = x_dims[0]*x_dims[1]*config['channels']
        primes = find_smallest_prime(x_dims[0], x_dims[1])
        if(int(config['z_dim_random_uniform']) > 0):
            z_dim_random_uniform = tf.random_uniform([config['batch_size'], int(config['z_dim_random_uniform'])],-1, 1,dtype=config['dtype'])
            #z_dim_random_uniform = tf.zeros_like(z_dim_random_uniform)
            z_dim_random_uniform = tf.identity(z_dim_random_uniform)
            inputs.append(z_dim_random_uniform)
        else:
            z_dim_random_uniform = None

        original_z = tf.concat(1, inputs)
        layers = config['generator.fully_connected_layers']
        net = original_z
        for i in range(layers):
            net = linear(net, net.get_shape()[-1], scope="g_fc_"+str(i))
            net = batch_norm(batch_size, name='g_rp_bn'+str(i))(net)
            net = activation(net)

            noise = tf.random_uniform([config['batch_size'],32],-1, 1,dtype=config['dtype'])
            net = tf.concat(1, [net, noise])
        print("Generator creating linear layer from", int(net.get_shape()[1]), "z to ", list(primes)+[z_proj_dims])
        net = linear(net, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj")

        net = tf.reshape(net,[config['batch_size'], primes[0], primes[1], z_proj_dims])

        nets = config['generator'](config, net)

        return nets,z_dim_random_uniform

def discriminator(config, x, f,z,g,gz):
    x_dims = config['x_dims']
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    activation = config['discriminator.activation']
    channels = (config['channels'])
    # combine to one batch, per Ian's "Improved GAN"
    xs = [x]
    gs = g
    set_tensor("xs", xs)
    set_tensor("gs", gs)
    g = g[-1]
    for i in gs:
        resized = tf.image.resize_images(xs[-1],xs[-1].get_shape()[1]//2,xs[-1].get_shape()[2]//2, 1)
        xs.append(resized)
    xs.pop()
    gs.reverse()
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


    if(config['format']!= 'mp3'):
        if(config['latent_loss']):
            orig_x = x
            x = build_reshape(int(x.get_shape()[1]), [z], config['d_project'], batch_size, config['dtype'])
            x = tf.reshape(x, [batch_size, -1, 1])
            x = tf.concat(2, [x, tf.reshape(orig_x, [batch_size, -1, channels])])
            x = tf.reshape(x,[batch_size, x_dims[0], x_dims[1], channels+1])
        else:
            x = tf.reshape(x,[batch_size, x_dims[0], x_dims[1], channels])


    net = config['discriminator'](config, x, g, xs, gs)
    net = tf.reshape(net, [batch_size, -1])


    if(config['minibatch']=='openai'):
      minis = get_minibatch_features(config, net, batch_size,config['dtype'])
      net = tf.concat(1, [net]+minis)

    if(config['minibatch']=='openai-smallest-image'):
      smallest_xg = get_tensor("xgs")[-1]
      print("Discriminator minibatch input size:", smallest_xg)
      minis = smallest_xg
      #todo this could be a different network?
      minis = conv2d(minis, 16, name='d_expand_minibatch', k_w=3, k_h=3, d_h=2, d_w=2)
      if(config['d_batch_norm']):
        minis = batch_norm(config['batch_size'], name='d_bn_minibatch')(minis)
      minis = activation(minis)
      minis = conv2d(minis, 32, name='d_expand_minibatch2', k_w=3, k_h=3, d_h=2, d_w=2)
      if(config['d_batch_norm']):
        minis = batch_norm(config['batch_size'], name='d_bn_minibatch2')(minis)
      minis = activation(minis)
      minis = tf.reshape(minis, [config['batch_size']*2, -1])
      minis = get_minibatch_features(config, minis, batch_size,config['dtype'])
      net = tf.concat(1, [net]+minis)

    if(config['discriminator.fc_layer']):
        print('Discriminator before linear layer', net, config['discriminator.fc_layer'])

        for layer in range(config['discriminator.fc_layers']):
            net = linear(net, config['discriminator.fc_layer.size'], scope="d_linear_layer"+str(layer))
            if(config['d_batch_norm']):
              net = batch_norm(config['batch_size'], name='d_bn_lin_proj'+str(layer))(net)


            net = activation(net)

    last_layer = net
    last_layer = tf.reshape(last_layer, [batch_size, -1])
    last_layer = tf.slice(last_layer, [single_batch_size, 0], [single_batch_size, -1])

    print('Discriminator last layer size:', net)
    net = linear(net, config['y_dims']+1, scope="d_proj")

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
    class_logits, gan_logits = build_logits(net, num_classes)
    return [tf.slice(class_logits, [0, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [0], [single_batch_size]),
                tf.slice(class_logits, [single_batch_size, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [single_batch_size], [single_batch_size]), 
                last_layer]


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


# Used for building the tensorflow graph with only G
def create_generator(config, x,y,f):
    set_ops_dtype(config['dtype'])
    #TODO fix copy/paste job here
    z_dim = int(config['z_dim'])
    z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = [tf.concat(1, categories)]
    else:
        categories_t = []


    args = [y, z]+categories_t
    g,z_dim_random_uniform = generator(config, args)
    set_tensor("g", g)
    set_tensor("y", y)
    set_tensor("z", z)
    return g

def create(config, x,y,f):
    # This is a hack to set dtype across ops.py, since each tensorflow instruction needs a dtype argument
    set_ops_dtype(config['dtype'])

    batch_size = config["batch_size"]
    z_dim = int(config['z_dim'])

    g_losses = []
    d_losses = []

    #initialize with random categories
    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = [tf.concat(1, categories)]
    else:
        categories_t = []

    #pretrained vectors is experimental and not recommended
    if(config['pretrained_model'] == 'preprocess'):
        if(config['latent_loss']):
            encoded_z, encoded_c, z, z_mu, z_sigma = z_from_f(config, f, categories)
        else:
            encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
            z_mu = None
            z_sigma = None
            z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])

    elif(config['latent_loss']):
        #draw z from the variational encoder
        encoded_z, z, z_mu, z_sigma = approximate_z(config, x, [y])
    else:
        #draw z from random uniform noise
        encoded_z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])
        z_mu = None
        z_sigma = None
        z = tf.random_uniform([config['batch_size'], z_dim],-1, 1,dtype=config['dtype'])

    # create generator
    g,z_dim_random_uniform = generator(config, [y, z]+categories_t)

    encoded,_ = generator(config, [y, encoded_z]+categories_t, reuse=True)

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

    #g loss from improved gan paper
    simple_g_loss = sigmoid_kl_with_logits(d_fake_sig, generator_target_prob)
    g_losses.append(simple_g_loss)
    if(config['adv_loss']):
        g_losses.append(sigmoid_kl_with_logits(d_real_sig, d_label_smooth))

    g_loss_fake= tf.nn.sigmoid_cross_entropy_with_logits(d_real_sig, zeros)
    g_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)


    if(config['g_class_loss']):
        g_losses.append(config['g_class_lambda']*tf.reduce_mean(g_class_loss))

    d_losses.append(d_fake_loss)
    d_losses.append(d_real_loss)

    if(int(y.get_shape()[1]) > 1):
        print("Discriminator class loss is on.  Semi-supervised learning mode activated.")
        d_losses.append(d_class_loss)
    else:
        print("Discriminator class loss is off.  Unsupervised learning mode activated.")

    if(config['latent_loss']):
        g_losses.append(latent_loss)

    if(config['d_fake_class_loss']):
        d_fake_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake,fake_symbol)
        d_losses.append(config['g_class_lambda']*tf.reduce_mean(d_fake_class_loss))


    categories_l = None
    if config['category_loss']:

        category_layer = linear(d_last_layer, sum(config['categories']), 'v_categories',stddev=0.15)
        category_layer = batch_norm(config['batch_size'], name='v_cat_loss')(category_layer)
        category_layer = config['generator.activation'](category_layer)
        categories_l = categories_loss(categories, category_layer, config['batch_size'])
        g_losses.append(-1*config['categories_lambda']*categories_l)
        d_losses.append(-1*config['categories_lambda']*categories_l)

    if config['regularize']:
        ws = None
        with tf.variable_scope("generator"):
            with tf.variable_scope("g_lin_proj"):
                tf.get_variable_scope().reuse_variables()
                ws = tf.get_variable('Matrix',dtype=config['dtype'])
                tf.get_variable_scope().reuse_variables()
            lam = config['regularize_lambda']
            print("ADDING REG", lam, ws)
            g_losses.append(lam*tf.nn.l2_loss(ws))
            if config['g_fc_layers'] > 0:
                with tf.variable_scope("g_fc_0"):
                    tf.get_variable_scope().reuse_variables()
                    ws = tf.get_variable('Matrix',dtype=config['dtype'])
                    tf.get_variable_scope().reuse_variables()
                lam = config['regularize_lambda']
                print("ADDING REG", lam, ws)
                g_losses.append(lam*tf.nn.l2_loss(ws))



    if(config['latent_loss']):
        mse_loss = tf.reduce_max(tf.square(x-encoded))
    else:
        mse_loss = None

    print("adding", g_losses)
    print("and", d_losses)
    g_loss = tf.reduce_mean(tf.add_n(g_losses))
    d_loss = tf.reduce_mean(tf.add_n(d_losses))
    joint_loss = tf.reduce_mean(tf.add_n(g_losses + d_losses))

    summary = tf.all_variables()
    def summary_reduce(s):
        if(len(s.get_shape())==0):
            return s
        while(len(s.get_shape())>1):
            s=tf.reduce_mean(s,1)
            #s=tf.squeeze(s)
        return tf.reduce_mean(s,0)

    summary = [(s.get_shape(), s.name, s.dtype, summary_reduce(s)) for s in summary]

    set_tensor("d_class_loss", tf.reduce_mean(d_class_loss))
    set_tensor("d_fake", tf.reduce_mean(d_fake))
    set_tensor("d_fake_loss", tf.reduce_mean(d_fake_loss))
    set_tensor("d_fake_sig", tf.reduce_mean(tf.sigmoid(d_fake_sig)))
    set_tensor("d_fake_sigmoid", tf.sigmoid(d_fake_sig))
    set_tensor("d_loss", d_loss)
    set_tensor("d_real", tf.reduce_mean(d_real))
    set_tensor("d_real_loss", tf.reduce_mean(d_real_loss))
    set_tensor("d_real_sig", tf.reduce_mean(tf.sigmoid(d_real_sig)))
    set_tensor("encoded", encoded)
    set_tensor("encoder_mse", mse_loss)
    set_tensor("f", f)
    set_tensor("g", g_sample)
    set_tensor("g_class_loss", tf.reduce_mean(g_class_loss))
    set_tensor("g_loss", g_loss)
    set_tensor("g_loss_sig", tf.reduce_mean(simple_g_loss))
    set_tensor("hc_summary",summary)
    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor("z", z)
    set_tensor('categories', categories_t)
    set_tensor('encoded_z', encoded_z)
    set_tensor('joint_loss', joint_loss)
    set_tensor('z_dim_random_uniform', z_dim_random_uniform)
    if(config['latent_loss']):
        set_tensor('latent_loss', tf.reduce_mean(latent_loss))

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    v_vars = [var for var in tf.trainable_variables() if 'v_' in var.name]
    g_vars += v_vars
    g_optimizer, d_optimizer = config['trainer.initializer'](config, d_vars, g_vars)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("g_optimizer", g_optimizer)

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


