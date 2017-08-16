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

    def _create(self):
        raise Exception('BaseTrainer _create called directly.  Please override.')

    def _step(self, feed_dict):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def create(self):
        config = self.config
        g_lr = config.g_learn_rate
        d_lr = config.d_learn_rate
        self.create_called = True
        self.global_step = tf.Variable(0, trainable=False)
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
        if not self.create_called:
            self.create()

        step = self._step(feed_dict)
        self.current_step += 1
        return step

    def required(self):
        return "d_trainer g_trainer d_learn_rate g_learn_rate".split()

    def output_string(self, metrics):
        output = "\%2d: " 
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
        return optimizer.apply_gradients(capped_gvs, global_step=self.global_step)


    def build_optimizer(self, config, prefix, trainer_config, learning_rate, var_list, loss):
        with tf.variable_scope(prefix):
            defn = {k[2:]: v for k, v in config.items() if k[2:] in inspect.getargspec(trainer_config).args and k.startswith(prefix)}
            optimizer = trainer_config(learning_rate, **defn)
            if(config.clipped_gradients):
                apply_gradients = self.capped_optimizer(optimizer, config.clipped_gradients, loss, var_list)
            else:
                apply_gradients = optimizer.minimize(loss, var_list=var_list, global_step=self.global_step)

        return apply_gradients



