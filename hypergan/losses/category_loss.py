from hypergan.losses.base_loss import BaseLoss

class CategoryLoss(BaseLoss):

    def required(self):
        return "categories category_lambda activation".split()

    def _create(self, d_real, d_fake):
        gan = self.gan
        ops = self.ops
        config = self.config
        categories = config.categories
        activation = ops.lookup(config.activation)
        #TODO broken.
        # TODO get the d_last_layer
        category_layer = gan.discriminator.ops.linear(d_real, sum(config.categories))
        category_layer= ops.layer_regularizer(d_real, config.layer_regularizer, config.batch_norm_epsilon)
        category_layer = activation(category_layer)

        loss = self.categories_loss(categories, category_layer)

        loss = -1*config.category_lambda*loss
        d_loss = loss
        g_loss = loss

        return d_loss, g_loss

    def split_categories(layer, batch_size, categories):
        start = 0
        ret = []
        for category in categories:
            count = int(category.get_shape()[1])
            ret.append(tf.slice(layer, [0, start], [batch_size, count]))
            start += count
        return ret

    def categories_loss(self, categories, layer):
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


