class BaseTrainer:
    def __init__(self):
        self.setup_called = False

    def _create(self):
        raise Exception('BaseTrainer _create called directly.  Please override.')

    def _step(self):
        raise Exception('BaseTrainer _step called directly.  Please override.')

    def step(self):
        if not self.setup_called:
            self.setup_called = True
            self.g_optimizer, self.d_optimizer = self._create()
        return self._step()
    
