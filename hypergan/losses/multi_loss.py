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
        d_real_partitions = tf.split(d_real, len(config.losses), len(gan.ops.shape(d_real))-1)
        d_fake_partitions = tf.split(d_fake, len(config.losses), len(gan.ops.shape(d_real))-1)
        for i, loss in enumerate(config.losses):
            
            loss_object = loss['class'](gan, loss, discriminator=self.discriminator, generator=self.generator)
            # TODO: Split d_fake?
            if config.partition:
                loss_object.create(d_real=d_real_partitions[i], d_fake=d_fake_partitions[i])
            else:
                loss_object.create(d_real=d_real, d_fake=d_fake)
            print("CRreating from ", loss_object.g_loss_features, loss_object.d_loss_features)

            losses.append(loss_object)

            print(loss_object)
        #relational layer?
        combine = MultiComponent(combine='concat', components=losses)

        print("LOSSES", combine.g_loss_features, "2",combine.lookup("g_loss_features"))
        if config.combine == 'concat':
            g_loss = combine.g_loss_features
            d_loss = combine.d_loss_features
        elif config.combine == 'linear':
            g_loss = self.F(combine.g_loss_features, "g_loss").sample
            d_loss = self.F(combine.d_loss_features, "d_loss").sample

        return [d_loss, g_loss]

    def F(self, loss, name):
        f_discriminator = self.gan.create_component(self.config.discriminator)
        f_discriminator.ops.describe(name)
        if self.discriminator.ops._reuse:
            f_discriminator.ops.reuse()

        result = f_discriminator.build(net=loss)
        self.discriminator.ops.add_weights(f_discriminator.variables())
        if self.discriminator.ops._reuse:
            f_discriminator.ops.stop_reuse()

        return f_discriminator


