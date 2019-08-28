from hypergan.gan_component import GANComponent
import hyperchamber as hc
import tensorflow as tf
import inspect

class BaseTrainer(GANComponent):
    def __init__(self, gan, config, d_vars=None, g_vars=None, name="BaseTrainer"):
        self.current_step = 0
        self.g_vars = g_vars
        self.d_vars = d_vars
        self.train_hooks = []
        
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
        self.d_lr = d_lr
        self.g_lr = g_lr
        for hook_config in (config.hooks or []):
            hook_config = hc.lookup_functions(hook_config.copy())
            defn = {k: v for k, v in hook_config.items() if k in inspect.getargspec(hook_config['class']).args}
            defn['gan']=self.gan
            defn['config']=hook_config
            defn['trainer']=self
            hook = hook_config["class"](**defn)
            self.gan.components += [hook]
            losses = hook.losses()
            if losses[0] is not None:
                self.gan.loss.sample[0] += losses[0]
            if losses[1] is not None:
                self.gan.loss.sample[1] += losses[1]
            self.train_hooks.append(hook)
 
        result = self._create()

        for hook in self.train_hooks:
            hook.after_create()

    def step(self, feed_dict={}):
        with self.gan.graph.as_default():
            step = self._step(feed_dict)
        self.current_step += 1
        return step

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
