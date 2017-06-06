import tensorflow as tf
import hyperchamber as hc


class SupervisedLoss:

    def create(self):
        batch_norm = config.batch_norm
        batch_size = gan.config.batch_size

        num_classes = gan.config.y_dims
        net = gan.graph.d_real
        net = linear(net, num_classes, scope="d_fc_end", stddev=0.003)
        net = batch_norm(batch_size, name='d_bn_end')(net)

        d_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net,labels=gan.graph.y)

        gan.graph.d_class_loss=tf.reduce_mean(d_class_loss)

        return [tf.reduce_mean(d_class_loss), None]

