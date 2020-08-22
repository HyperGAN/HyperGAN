from hypergan.gan_component import GANComponent
import hyperchamber as hc
import inspect

class BaseTrainer(GANComponent):
    def __init__(self, gan, config, trainable_gan):
        self.current_step = 0
        self.train_hooks = gan.hooks
        self.trainable_gan = trainable_gan

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
        for hook in self.gan.hooks:
            losses = hook.losses()
            if losses[0] is not None:
                self.trainable_gan.loss.sample[0] += losses[0]
            if losses[1] is not None:
                self.trainable_gan.loss.sample[1] += losses[1]

        result = self._create()

        for i, hook in enumerate(self.train_hooks):
            hook.after_create()
            setattr(self, 'train_hook'+str(i), hook)

    def create_optimizer(self, name="optimizer"):
        defn = getattr(self.config, name) or self.config.optimizer
        defn = defn.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]

        print("S", self.trainable_gan)
        optimizer = self.trainable_gan.create_optimizer(klass, defn)
        return optimizer

    def calculate_gradients(self):
        raise ValidationException("BaseTrainer#calculate_gradients called directly, please override")

    def step(self, feed_dict={}):
        step = self._step(feed_dict)
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
