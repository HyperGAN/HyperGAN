import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
from .common import *

class ProportionalControlTrainer:

    def __init__(self,
            d_learn_rate=1e-3,
            d_epsilon=1e-8,
            d_beta1=0.9,
            d_beta2=0.999,
            g_learn_rate=1e-3,
            g_epsilon=1e-8,
            g_beta1=0.9,
            g_beta2=0.999,
            g_momentum=0.01, 
            d_momentum=0.00001, 
            g_decay=0.999, 
            d_decay=0.995, 
            d_rho=0.95,
            g_rho=0.95,
            d_initial_accumulator_value=0.1,
            g_initial_accumulator_value=0.1,
            d_trainer=tf.train.AdamOptimizer,
            g_trainer=tf.train.AdamOptimizer,
            d_clipped_weights=False,
            clipped_gradients=False
        ):
        selector = hc.Selector()

        selector.set('d_learn_rate', d_learn_rate)
        selector.set('d_epsilon', d_epsilon)
        selector.set('d_beta1', d_beta1)
        selector.set('d_beta2', d_beta2)

        selector.set('g_learn_rate', g_learn_rate)
        selector.set('g_epsilon', g_epsilon)
        selector.set('g_beta1', g_beta1)
        selector.set('g_beta2', g_beta2)

        selector.set('clipped_gradients', clipped_gradients)
        selector.set('d_clipped_weights', d_clipped_weights)

        selector.set('d_decay', d_decay)
        selector.set('g_decay', g_decay)

        selector.set('d_momentum', d_momentum)
        selector.set('g_momentum', g_momentum)

        selector.set('d_trainer', d_trainer)
        selector.set('g_trainer', g_trainer)

        selector.set('d_rho', d_rho)
        selector.set('g_rho', g_rho)

        selector.set('d_initial_accumulator_value', d_initial_accumulator_value)
        selector.set('g_initial_accumulator_value', g_initial_accumulator_value)

        self.config = selector.random_config()
        self.iteration = 0

    def create(config, gan, d_vars, g_vars):
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


    def run(gan, feed_dict):
        global iteration
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

        _, d_cost, d_log = sess.run([d_optimizer, d_loss, d_log_t])

        _, g_cost, g_k, measure= sess.run([g_optimizer, g_loss, gan.graph.update_k, gan.graph.measure], feed_dict)
        if iteration % 100 == 0:
            print("%2d: g cost %.2f d_loss %.2f k %.2f m %.2f gamma %.2f" % (iteration,g_cost , d_cost,g_k, measure, gan.graph.gamma))

        iteration+=1

        return d_cost, g_cost


