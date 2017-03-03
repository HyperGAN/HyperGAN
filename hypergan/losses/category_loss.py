def config():
    selector = hc.Selector()
    selector.set('discriminator', None)

    selector.set('create', create)

    return selector.random_config()

def create(config, gan):
    category_layer = linear(d_last_layer, sum(config['categories']), 'v_categories',stddev=0.15)
    category_layer = batch_norm(config['batch_size'], name='v_cat_loss')(category_layer)
    category_layer = config['generator.activation'](category_layer)
    categories_l = categories_loss(categories, category_layer, config['batch_size'])
    g_losses.append(-1*config['categories_lambda']*categories_l)
    d_losses.append(-1*config['categories_lambda']*categories_l)

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
        logli_prior = tf.reduce_sum(tf.log(category_prior + TINY) * category, axis=1)
        layer_softmax = tf.nn.softmax(layer_s)
        logli = tf.reduce_sum(tf.log(layer_softmax+TINY)*category, axis=1)
        disc_ent = tf.reduce_mean(-logli_prior)
        disc_cross_ent =  tf.reduce_mean(-logli)

        loss += disc_ent - disc_cross_ent
    return loss


