from hypergan.gan_component import GANComponent

class BaseTrainer(GANComponent):

    def __init__(self, gan, config):
        GANComponent.__init__(self, gan, config)
        self.create_called = False

    def _create(self):
        raise Exception('BaseTrainer _create called directly.  Please override.')

    def _step(self):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def create(self):
        self.create_called = True
        return self._create()

    def step(self):
        if not self.create_called:
            self.g_optimizer, self.d_optimizer = self.create()
        step = self._step()
        self.step += 1
        return step

    def required(self):
        return "d_trainer g_trainer d_learn_rate g_learn_rate".split()

