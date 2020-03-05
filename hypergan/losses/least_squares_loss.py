import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LeastSquaresLoss(BaseLoss):

    def required(self):
        return "".split()

    def _forward(self, d_real, d_fake):
        config = self.config

        a,b,c = (config.labels or [-1,1,1])
        d_loss = 0.5*((d_real - b)**2) + 0.5*((d_fake - a)**2)
        g_loss = 0.5*((d_fake - c)**2)

        return [d_loss, g_loss]
