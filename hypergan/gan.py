from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.util.hc_tf import *
import tensorflow as tf
import hypergan.util.wavegan as wavegan
TINY = 1e-12

def generator(config, inputs, reuse=False):
    x_dims = config['x_dims']
    output_channels = config['channels']
    activation = config['generator.activation']
    batch_size = config['batch_size']
    z_proj_dims = int(config['generator.z_projection_depth'])
    batch_norm = config['generator.regularizers.layer']

    with(tf.variable_scope("generator", reuse=reuse)):
        output_shape = x_dims[0]*x_dims[1]*config['channels']
        primes = find_smallest_prime(x_dims[0], x_dims[1])
        dropout = tf.Variable(0.5)
        set_tensor("dropout", dropout)

        original_z = tf.concat(1, inputs)
        layers = config['generator.fully_connected_layers']
        net = original_z
        for i in range(layers):
            net = linear(net, net.get_shape()[-1], scope="g_fc_"+str(i))
            net = batch_norm(batch_size, name='g_rp_bn'+str(i))(net)
            net = activation(net)

        set_tensor('original_z', net)
        net = linear(net, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj")
        new_shape = [config['batch_size'], primes[0],primes[1],z_proj_dims]
        net = tf.reshape(net, new_shape)

        nets = config['generator'](config, net)

        return nets

def discriminator(config, x, f,z,g,gz):
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    activation = config['discriminator.activation']
    channels = (config['channels'])
    batch_norm = config['discriminator.regularizers.layer']
    # combine to one batch, per Ian's "Improved GAN"
    xs = [x]
    gs = g
    set_tensor("xs", xs)
    set_tensor("gs", gs)
    g = g[-1]
    for i in gs:
        resized = tf.image.resize_images(xs[-1],[int(xs[-1].get_shape()[1]//2),int(xs[-1].get_shape()[2]//2)], 1)
        xs.append(resized)
    xs.pop()
    gs.reverse()
    x = tf.concat(0, [x,g])

    # careful on order.  See https://arxiv.org/pdf/1606.00704v1.pdf
    z = tf.concat(0, [z, gz])
    if(config['discriminator.add_noise']):
        x += tf.random_normal(x.get_shape(), mean=0, stddev=config['discriminator.noise_stddev'], dtype=config['dtype'])

    net = config['discriminator'](config, x, g, xs, gs)

    if(config['discriminator.fc_layer']):
        for layer in range(config['discriminator.fc_layers']):
            net = linear(net, config['discriminator.fc_layer.size'], scope="d_linear_layer"+str(layer))
            net = batch_norm(config['batch_size'], name='d_bn_lin_proj'+str(layer))(net)
            net = activation(net)

    last_layer = net
    last_layer = tf.reshape(last_layer, [batch_size, -1])
    last_layer = tf.slice(last_layer, [single_batch_size, 0], [single_batch_size, -1])

    regularizers = []
    for regularizer in config['discriminator.regularizers']:
        regs = regularizer(config, net)
        regularizers += regs

    net = tf.concat(1, [net]+regularizers)

    num_classes = config['y_dims']+1
    if config['y_dims'] == 1:
        net = linear(net, 1, scope="d_fc_end", stddev=0.003)
        class_logits = net
        gan_logits = tf.squeeze(net)

    else:
        net = linear(net, num_classes, scope="d_fc_end", stddev=0.003)
        class_logits = tf.slice(net, [0,1], [single_batch_size*2,num_classes-1])
        gan_logits = tf.squeeze(tf.slice(net, [0,0], [single_batch_size*2,1]))

    return [tf.slice(class_logits, [0, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [0], [single_batch_size]),
                tf.slice(class_logits, [single_batch_size, 0], [single_batch_size, num_classes-1]),
                tf.slice(gan_logits, [single_batch_size], [single_batch_size]), 
                last_layer]


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
    set_ops_globals(config['dtype'], config['batch_size'])
    z_dim = int(config['generator.z'])
    z, encoded_z, z_mu, z_sigma = config['encoder'](config, x, y)
    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = [tf.concat(1, categories)]
    else:
        categories_t = []


    args = [y, z]+categories_t
    g = generator(config, args)
    set_tensor("g", g)
    set_tensor("y", y)
    set_tensor("z", z)
    return g

def create(config, x,y,f):
    # This is a hack to set dtype across ops.py, since each tensorflow instruction needs a dtype argument
    set_ops_globals(config['dtype'], config['batch_size'])

    batch_size = config["batch_size"]
    z_dim = int(config['generator.z'])
    batch_norm = config['generator.regularizers.layer']

    g_losses = []
    extra_g_loss = []
    d_losses = []

    #initialize with random categories
    categories = [random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
    if(len(categories) > 0):
        categories_t = [tf.concat(1, categories)]
    else:
        categories_t = []

    z, encoded_z, z_mu, z_sigma = config['encoder'](config, x, y)

    # create generator
    g = generator(config, [y, z]+categories_t)

    #encoded = generator(config, [y, encoded_z]+categories_t, reuse=True)

    g_sample = g

    d_real, d_real_sig, d_fake, d_fake_sig, d_last_layer = discriminator(config,x, f, encoded_z, g, z)

    if(config['latent_loss']):
        latent_loss = -config['latent_lambda'] * tf.reduce_mean(1 + z_sigma
                                       - tf.square(z_mu)
                                       - tf.exp(z_sigma), 1)

        latent_loss = tf.reshape(latent_loss, [int(latent_loss.get_shape()[0]), 1])
    else:
        latent_loss = None
    np_fake = np.array([0]*config['y_dims']+[1])
    fake_symbol = tf.tile(tf.constant(np_fake, dtype=config['dtype']), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],config['y_dims']+1])

    #real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])
    real_symbols = y


    zeros = tf.zeros_like(d_fake_sig, dtype=config['dtype'])
    ones = tf.zeros_like(d_real_sig, dtype=config['dtype'])

    generator_target_prob = config['g_target_prob']
    d_label_smooth = config['d_label_smooth']
    d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake_sig, zeros)
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
        print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
        d_losses.append(d_class_loss)
    else:
        print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

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

    for reg in config['generator.regularizers']:
        extra_g_loss += reg(config)

    if(config['latent_loss']):
        #mse_loss = tf.reduce_max(tf.square(x-encoded))
        mse_loss = None
    else:
        mse_loss = None

    g_loss = tf.reduce_mean(tf.add_n(g_losses))
    for extra in extra_g_loss:
        g_loss += extra
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
    #set_tensor("encoded", encoded)
    set_tensor("encoder_mse", mse_loss)
    set_tensor("f", f)
    set_tensor("g", g_sample)
    set_tensor("g_class_loss", tf.reduce_mean(g_class_loss))
    set_tensor("g_loss", g_loss)
    set_tensor("g_loss_sig", tf.reduce_mean(simple_g_loss))
    set_tensor("hc_summary",summary)
    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor('categories', categories_t)
    set_tensor('encoded_z', encoded_z)
    set_tensor('joint_loss', joint_loss)
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

    return g_cost,d_fake_cost, d_real_cost,0
