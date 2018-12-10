import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss
from hypergan.multi_component import MultiComponent

TINY=1e-8
class MultiLoss(BaseLoss):
    """Takes multiple distributions and does an additional approximator"""
    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config
        losses = []
        split = self.split

        for d in gan.discriminator.children:
            if config.swapped:
                d_swap = d_real
                d_real = d_fake
                d_fake = d_swap
            ds = self.split_batch(d.sample, split)
            d_real = ds[0]
            d_fake = tf.add_n(ds[1:])/(len(ds)-1)

            loss_object = self.config['loss_class'](gan, self.config, d_real=d_real, d_fake=d_fake)

            losses.append(loss_object)

        #relational layer?
        combine = MultiComponent(combine='concat', components=losses)

        g_loss = combine.g_loss_features
        d_loss = combine.d_loss_features

        self.d_loss = d_loss
        self.g_loss = g_loss

        self.losses = losses

        return [d_loss, g_loss]


