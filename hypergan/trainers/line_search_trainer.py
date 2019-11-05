import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer
from hypergan.ops.tensorflow.probls.tensorflow_interface.interface_sgd import ProbLSOptimizerSGDInterface
from hypergan.ops.tensorflow.probls.line_search import ProbLSOptimizer

TINY = 1e-12

class LineSearchTrainer(BaseTrainer):
    def _create(self):
        self._variables = []
        gan = self.gan
        config = self.config

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.step_ops = None

        #self.d_opt = ProbLSOptimizerSGDInterface()
        #self.d_min = self.d_opt.minimize(self.gan.loss.unsquashed_d_loss, self.gan.trainable_d_vars())
        #self.g_opt = ProbLSOptimizerSGDInterface()
        #self.g_min = self.g_opt.minimize(self.gan.loss.unsquashed_g_loss, self.gan.trainable_g_vars())

        d_vars = self.gan.discriminator.ops.weights + self.gan.discriminator.ops.biases
        g_vars = self.gan.generator.ops.weights + self.gan.generator.ops.biases
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)


        g_alpha_var = tf.Variable(self.config.initial_alpha, trainable=False, name="g_alpha_dontsave")
        d_alpha_var = tf.Variable(self.config.initial_alpha, trainable=False, name="d_alpha_dontsave")
        self.reset_d_alpha = d_alpha_var.assign(self.config.initial_alpha)
        self.reset_g_alpha = g_alpha_var.assign(self.config.initial_alpha)
        self._variables += [g_alpha_var, d_alpha_var]

        g_alpha_variables = [ _x + _g * g_alpha_var for _x, _g in zip(g_vars, g_grads) ]
        d_alpha_variables = [ _x + _g * d_alpha_var for _x, _g in zip(d_vars, d_grads) ]

        g_alpha_biases = g_alpha_variables[len(self.gan.generator.ops.weights):]
        g_alpha_weights = g_alpha_variables[:len(self.gan.generator.ops.weights)]
        d_alpha_biases = d_alpha_variables[len(self.gan.discriminator.ops.weights):]
        d_alpha_weights = d_alpha_variables[:len(self.gan.discriminator.ops.weights)]
        print("________________________________________________", g_alpha_weights)

        self.g_alpha = self.gan.create_component(gan.config.generator, name="generator", input=self.gan.latent.sample, weights=g_alpha_weights, biases=g_alpha_biases)
        x, g = self.gan.inputs.x, self.g_alpha.sample
        self.d_alpha = self.gan.create_component(gan.config.discriminator, name="discriminator", input=tf.concat([x,g],axis=0), weights=d_alpha_weights, biases=d_alpha_biases)

        self.loss = self.gan.create_component(gan.config.loss, discriminator=self.d_alpha)
        self.alpha_d_loss, self.alpha_g_loss = self.loss.sample

        self.d_optimizer = self.gan.create_optimizer(self.config.alpha_optimizer)
        self.d_train = self.d_optimizer.minimize(self.alpha_d_loss, var_list=[d_alpha_var])
        self.g_optimizer = self.gan.create_optimizer(self.config.alpha_optimizer)
        self.g_train = self.g_optimizer.minimize(self.alpha_g_loss, var_list=[g_alpha_var])
        self.reset_d_alpha_optimizer = tf.variables_initializer(self.d_optimizer.variables())
        self.reset_g_alpha_optimizer = tf.variables_initializer(self.g_optimizer.variables())

        self.d_alpha_var = d_alpha_var
        self.g_alpha_var = g_alpha_var

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.gan.trainer = self

        d_alpha_grads = [ _g * tf.nn.relu(d_alpha_var) for _g in d_grads ]
        g_alpha_grads = [ _g * tf.nn.relu(g_alpha_var) for _g in g_grads ]

        apply_vec = list(zip((d_alpha_grads + g_alpha_grads), (d_vars + g_vars))).copy()
        self.optimizer = self.gan.create_optimizer(config.optimizer)
        self.optimize_t = self.optimizer.apply_gradients(apply_vec)
    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        metrics = gan.metrics()

        self.before_step(self.current_step, feed_dict)

        sess.run([self.reset_d_alpha, self.reset_g_alpha, self.reset_d_alpha_optimizer, self.reset_g_alpha_optimizer])

        for i in range(self.config.nsteps):
            alpha_d, alpha_g = sess.run([self.d_alpha_var, self.g_alpha_var], feed_dict)
            if self.config.verbose:
                print(i, "Alpha", alpha_d, alpha_g)
            sess.run([self.d_train, self.g_train], feed_dict)
        alpha_d, alpha_g = sess.run([self.d_alpha_var, self.g_alpha_var], feed_dict)
        if self.config.verbose:
            print("Final Alpha", alpha_d, alpha_g)
        sess.run(self.optimize_t, feed_dict)
        d_loss, g_loss = loss.sample

        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            metric_values = self.gan.session.run(self.output_variables(metrics))
            self.print_metrics(self.current_step)

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.gan.session.run(self.output_variables(metrics))
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))

    def variables(self):
        return super().variables() + self._variables
