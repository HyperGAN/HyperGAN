from .backend import Backend

class SingleGPUBackend(Backend):
    def __init__(self, trainable_gan, devices):
        self.trainable_gan = trainable_gan

    def save(self):
        self.trainable_gan.save_locally()

    def step(self):
        self.trainable_gan.trainer.step()
