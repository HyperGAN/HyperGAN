from hypergan.gan_component import GANComponent
import hyperchamber as hc
import inspect

class BaseTrainer(GANComponent):
    def __init__(self, gan, config):
        self.current_step = 0
        self.train_hooks = []
        
        GANComponent.__init__(self, gan, config)

    def _step(self, feed_dict):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def create(self):
        config = self.config
        g_lr = config.g_learn_rate
        d_lr = config.d_learn_rate
        self.create_called = True
        self.d_lr = d_lr
        self.g_lr = g_lr
        for hook_config in (config.hooks or []):
            hook_config = hc.lookup_functions(hook_config.copy())
            defn = {k: v for k, v in hook_config.items() if k in inspect.getargspec(hook_config['class']).args}
            defn['gan']=self.gan
            defn['config']=hook_config
            defn['trainer']=self
            hook = hook_config["class"](**defn)
            self.gan.add_component("hook", hook)
            losses = hook.losses()
            if losses[0] is not None:
                self.gan.loss.sample[0] += losses[0]
            if losses[1] is not None:
                self.gan.loss.sample[1] += losses[1]
            if "only" in hook_config and hook_config["only"]:
                self.gan.loss = hc.Config({"sample": losses})
            self.train_hooks.append(hook)

        result = self._create()

        for hook in self.train_hooks:
            hook.after_create()

    def calculate_gradients(self):
        raise ValidationException("BaseTrainer#calculate_gradients called directly, please override")

    def step(self, feed_dict={}):
        step = self._step(feed_dict)
        self.gan.add_metric('d_loss', self.d_loss)
        self.gan.add_metric('g_loss', self.g_loss)
        self.current_step += 1
        return step

    def required(self):
        return "".split()

    def output_string(self, metrics):
        output = " %2d: " 
        for name in sorted(metrics.keys()):
            output += " " + name
            output += " %.2f"
        return output

    def output_variables(self, metrics):
        gan = self.gan
        return [metrics[k] for k in sorted(metrics.keys())]


    def before_step(self, step, feed_dict):
        for component in self.train_hooks:
            component.before_step(step, feed_dict)

    def after_step(self, step, feed_dict):
        for component in self.train_hooks:
            component.after_step(step, feed_dict)
