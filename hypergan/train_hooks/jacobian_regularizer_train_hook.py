from jacobian import JacobianReg
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class JacobianRegularizerTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.reg = JacobianReg()

    def forward(self, d_loss, g_loss):
        lam = self.config.jacobian_lambda or 0.01
        self.gan.augmented_x.requires_grad = True

        reg_loss = lam * self.reg(self.gan.augmented_x, self.gan.discriminator(self.gan.augmented_x))
        return [reg_loss, reg_loss]
