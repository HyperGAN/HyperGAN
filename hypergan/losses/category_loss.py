from hypergan.losses.base_loss import BaseLoss

import numpy as np
import tensorflow as tf

TINY = 1e-12
class CategoryLoss(BaseLoss):

    def required(self):
        return "category_lambda activation".split()

    def _create(self, d_real, d_fake):
        gan = self.gan
        ops = self.ops
        config = self.config
        categories = gan.encoder.categories
        size = sum([ops.shape(x)[1] for x in categories])
        activation = ops.lookup(config.activation)

        category_layer = gan.discriminator.ops.linear(gan.discriminator.sample, size)
        category_layer= ops.layer_regularizer(d_real, config.layer_regularizer, config.batch_norm_epsilon)
        category_layer = activation(category_layer)

        loss = self.categories_loss(categories, category_layer)

        loss = -1*config.category_lambda*loss
        d_loss = loss
        g_loss = loss

        return d_loss, g_loss

    def categories_loss(self, categories, layer):
        gan = self.gan
        loss = 0
        batch_size = gan.batch_size()
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


