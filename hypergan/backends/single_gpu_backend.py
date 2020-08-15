from .backend import Backend

class SingleGPUBackend(Backend):
    def __init__(self, gan, cli, devices):
        self.gan = gan
        self.cli = cli

    def step(self):
        self.gan.step()
