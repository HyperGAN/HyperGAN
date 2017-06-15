from hypergan.gan_component import GANComponent

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
        self.create_called = True

        return self._create()

    def step(self, feed_dict={}):
        if not self.create_called:
            self.g_optimizer, self.d_optimizer = self.create()

        step = self._step(feed_dict)
        self.current_step += 1
        return step

    def required(self):
        return "d_trainer g_trainer d_learn_rate g_learn_rate".split()

