import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
from hypergan.trainers.base_trainer import BaseTrainer

class ProportionalControlTrainer(BaseTrainer):

    def _create(self):
        d_loss = gan.graph.d_loss
        g_loss = gan.graph.g_loss
        g_lr = np.float32(config.g_learn_rate)
        d_lr = np.float32(config.d_learn_rate)

        gan.graph.d_vars = d_vars
        g_defk = {k[2:]: v for k, v in config.items() if k[2:] in inspect.getargspec(config.g_trainer).args and k.startswith("d_")}
        d_defk = {k[2:]: v for k, v in config.items() if k[2:] in inspect.getargspec(config.d_trainer).args and k.startswith("g_")}
        g_optimizer = config.g_trainer(g_lr, **g_defk)
        d_optimizer = config.d_trainer(d_lr, **d_defk)
        if(config.clipped_gradients):
            g_optimizer = capped_optimizer(g_optimizer, config.clipped_gradients, g_loss, g_vars)
            d_optimizer = capped_optimizer(d_optimizer, config.clipped_gradients, d_loss, d_vars)
        else:
            g_optimizer = g_optimizer.minimize(g_loss, var_list=g_vars)
            d_optimizer = d_optimizer.minimize(d_loss, var_list=d_vars)

        gan.graph.clip = [tf.assign(d,tf.clip_by_value(d, -config.d_clipped_weights, config.d_clipped_weights))  for d in d_vars]

        return g_optimizer, d_optimizer


    def _step(gan, feed_dict):
        sess = gan.sess
        config = gan.config
        x_t = gan.graph.x
        g_t = gan.graph.g
        d_log_t = gan.graph.d_log
        g_loss = gan.graph.g_loss
        d_loss = gan.graph.d_loss
        g_optimizer = gan.graph.g_optimizer
        d_optimizer = gan.graph.d_optimizer
        d_class_loss = gan.graph.d_class_loss
        d_vars = gan.graph.d_vars

        # in WGAN paper, values are clipped.  This might not work, and is slow.
        if(config.d_clipped_weights):
            sess.run(gan.graph.clip)

        _, d_cost, d_log = sess.run([d_optimizer, d_loss, d_log_t], feed_dict)

        _, g_cost, g_k, measure= sess.run([g_optimizer, g_loss, gan.graph.update_k, gan.graph.measure], feed_dict)
        if self.step % 100 == 0:
            print("%2d: g cost %.2f d_loss %.2f k %.2f m %.2f gamma %.2f" % (self.step, g_cost , d_cost, g_k, measure, gan.graph.gamma))

        self.step+=1

        return d_cost, g_cost
