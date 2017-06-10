import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class SupervisedLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        batch_size = gan.batch_size()
        net = d_real

        num_classes = gan.ops.shape(gan.inputs.y)[1]
        net = gan.discriminator.ops.linear(net, num_classes)
        net = self.layer_regularizer(net)

        d_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net,labels=gan.inputs.y)
        d_class_loss = gan.ops.squash(d_class_loss)

        self.metrics = {
            'd_class_loss': d_class_loss
        }

        return [d_class_loss, None]
