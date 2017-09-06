from hypergan.gan_component import GANComponent
import tensorflow as tf
import inspect

class BaseTrainer(GANComponent):

    def __init__(self, gan, config, d_vars=None, g_vars=None, loss=None):
        GANComponent.__init__(self, gan, config)
        self.create_called = False
        self.current_step = 0
        self.g_vars = g_vars
        self.d_vars = d_vars
        self.loss = loss
        self.d_shake = None
        self.g_shake = None

    def _create(self):
        raise Exception('BaseTrainer _create called directly.  Please override.')

    def _step(self, feed_dict):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def create(self):
        self.create_called = True

        return self._create()

    def step(self, feed_dict={}):
        if not self.create_called:
            self.create()

        self.shake_weights()
        step = self._step(feed_dict)
        self.current_step += 1
        return step

    def shake_weight_d_assigns(self, weights):
        if(self.config.shake_weights_d is None):
            return []
        if(self.d_shake is None):
            self.d_shake = []
            for weight in weights:
                dimensions = 1
                for dim in self.ops.shape(weight):
                    dimensions *= dim 
                gaussian = tf.random_normal([dimensions])
                uniform = tf.random_uniform(shape=[dimensions],minval=0.,maxval=1.)
                unit_ball = gaussian / tf.nn.l2_normalize(gaussian, dim=0)
                unit_ball = uniform*self.config.shake_weights_d
                unit_ball = tf.reshape(unit_ball, self.ops.shape(weight))
                op = tf.assign(weight, weight+unit_ball)
                self.d_shake.append(op)
        return self.d_shake

    def shake_weight_g_assigns(self, weights):
        if(self.config.shake_weights_g is None):
            return []
        if(self.g_shake is None):
            self.g_shake = []
            for weight in weights:
                dimensions = 1
                for dim in self.ops.shape(weight):
                    dimensions *= dim
                gaussian = tf.random_normal([dimensions])
                uniform = tf.random_uniform(shape=[dimensions],minval=0.,maxval=1.)
                unit_ball = gaussian / tf.nn.l2_normalize(gaussian, dim=0)
                unit_ball = uniform*self.config.shake_weights_g
                unit_ball = tf.reshape(unit_ball, self.ops.shape(weight))
                print(unit_ball)
                op = tf.assign(weight, weight+unit_ball)
                self.g_shake.append(op)
        return self.g_shake

    def shake_weights(self):
        gan = self.gan
        if self.config.shake_weights_d:
            d_vars = self.d_vars or gan.discriminator.variables()
            d_ops = self.shake_weight_d_assigns(d_vars)
            if d_ops:
                self.gan.session.run(d_ops)
        if self.config.shake_weights_g:
            g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())
            g_ops = self.shake_weight_g_assigns(g_vars)
            if g_ops:
                self.gan.session.run(g_ops)


    def required(self):
        return "d_trainer g_trainer d_learn_rate g_learn_rate".split()

    def output_string(self, metrics):
        name = self.gan.name or ""
        output = name + " %2d: " 
        for name in sorted(metrics.keys()):
            output += " " + name
            output += " %.2f"
        return output

    def output_variables(self, metrics):
        gan = self.gan
        sess = gan.session
        return [metrics[k] for k in sorted(metrics.keys())]

    def capped_optimizer(optimizer, cap, loss, var_list):
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        def create_cap(grad,var):
            if(grad == None) :
                print("Warning: No gradient for variable ",var.name)
                return None
            return (tf.clip_by_value(grad, -cap, cap), var)

        capped_gvs = [create_cap(grad,var) for grad, var in gvs]
        capped_gvs = [x for x in capped_gvs if x != None]
        return optimizer.apply_gradients(capped_gvs)


    def build_optimizer(self, config, prefix, trainer_config, learning_rate, var_list, loss):
        with tf.variable_scope(prefix):
            defn = {k[2:]: v for k, v in config.items() if k[2:] in inspect.getargspec(trainer_config).args and k.startswith(prefix)}
            optimizer = trainer_config(learning_rate, **defn)
            if(config.clipped_gradients):
                apply_gradients = self.capped_optimizer(optimizer, config.clipped_gradients, loss, var_list)
            else:
                apply_gradients = optimizer.minimize(loss, var_list=var_list)

        return apply_gradients



