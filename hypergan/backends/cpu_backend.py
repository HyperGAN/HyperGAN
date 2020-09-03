from .backend import Backend

class CPUBackend(Backend):
    def __init__(self, trainable_gan, devices):
        self.trainable_gan = trainable_gan
        self.trainable_gan.to('cpu')

    def save(self):
        self.trainable_gan.save_locally()

    def step(self):
        self.trainable_gan.trainer.step()
