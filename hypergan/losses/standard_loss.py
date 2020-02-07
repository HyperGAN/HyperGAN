import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class StandardLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def _create(self, d_real, d_fake):
        config = self.config
        gan = self.gan

        d_loss = 0
        g_loss = 0 #TODO
        return [d_loss, g_loss]

