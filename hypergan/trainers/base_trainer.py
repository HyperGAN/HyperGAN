from hypergan.gan_component import GANComponent
import hyperchamber as hc
import tensorflow as tf
import inspect

class BaseTrainer(GANComponent):
    def __init__(self, gan, config, d_vars=None, g_vars=None, name="BaseTrainer"):
        self.current_step = 0
        self.g_vars = g_vars
        self.d_vars = d_vars
        self.d_shake = None
        self.g_shake = None
        self.train_hooks = []
        for hook_config in (config.hooks or []):
            hook_config = hc.lookup_functions(hook_config.copy())
            defn = {k: v for k, v in hook_config.items() if k in inspect.getargspec(hook_config['class']).args}
            defn['gan']=gan
            defn['config']=hook_config
            defn['trainer']=self
            hook = hook_config["class"](**defn)
            losses = hook.losses()
            if losses[0] is not None:
                gan.loss.sample[0] += losses[0]
            if losses[1] is not None:
                gan.loss.sample[1] += losses[1]
            self.train_hooks.append(hook)
 
        GANComponent.__init__(self, gan, config, name=name)

    def _step(self, feed_dict):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def variables(self):
        return self.ops.variables() + self.optimizer.variables()

    def create(self):
        config = self.config
        g_lr = config.g_learn_rate
        d_lr = config.d_learn_rate
        self.create_called = True
        self.global_step = tf.train.get_global_step()
        decay_function = config.decay_function
        if decay_function:
            print("using decay function", decay_function)
            decay_steps = config.decay_steps or 50000
            decay_rate = config.decay_rate or 0.9
            decay_staircase = config.decay_staircase or False
            self.d_lr = decay_function(d_lr, self.global_step, decay_steps, decay_rate, decay_staircase)
            self.g_lr = decay_function(g_lr, self.global_step, decay_steps, decay_rate, decay_staircase)
        else:
            self.d_lr = d_lr
            self.g_lr = g_lr

        return self._create()

    def step(self, feed_dict={}):
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
        return "".split()

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


    def before_step(self, step, feed_dict):
        for component in self.train_hooks:
            component.before_step(step, feed_dict)

    def after_step(self, step, feed_dict):
        for component in self.train_hooks:
            component.after_step(step, feed_dict)
