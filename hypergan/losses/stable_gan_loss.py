import hypergan
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class StableGANLoss:
    """
    Stablized gan loss. Subject to change.
    ```python
        x = gan.inputs.next()
        g = gan.generator(gan.latent.next())
        discriminator = gan.create_component("discriminator")
        d_loss, g_loss = StableDiscriminatorLoss(discriminator, [x], [g]).stable_loss()
    ```
    """
    def __init__(self, discriminator):
        self.gamma1 = torch.tensor(10.0, device= discriminator.gan.device)
        self.g_gamma1 = torch.tensor(10.0, device= discriminator.gan.device)
        self.gamma2 = torch.tensor(100.0, device= discriminator.gan.device)
        self.g_gamma2 = torch.tensor(0.1, device= discriminator.gan.device)
        self.target1_x = None
        self.target1_g = None
        self.target2_x = None
        self.target2_g = None
        self.discriminator = discriminator

    def ragan(self, d_real, d_fake):
        cr = torch.mean(d_real,0)
        cf = torch.mean(d_fake,0)
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
        d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        return d_loss, g_loss

    def stable_loss(self, x, g):
        self.x = x
        self.g = g
        self.d_real = self.discriminator(self.x)
        self.d_fake = self.discriminator(self.g)
        d_losses = []
        g_losses = []
        d_loss, g_loss = self.ragan(self.d_real, self.d_fake)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        if self.target1_x == None:
            self.target1_g = Parameter(self.g, requires_grad=True)
            self.target1_x = Parameter(self.x, requires_grad=True)
            self.target2_g = Parameter(self.g, requires_grad=True)
            self.target2_x = Parameter(self.x, requires_grad=True)

        self.target1_g.data = self.g.clone()
        self.target2_g.data = self.g.clone()
        self.target1_x.data = self.x.clone()
        self.target2_x.data = self.x.clone()

        neg_inverse_fake = self.inverse(self.discriminator(self.target2_g), self.d_real, self.target2_g)[0]
        neg_inverse_real = self.inverse(self.d_fake, self.discriminator(self.target2_x), self.target2_x)[0]

        reg_fake, _ = self.ragan(self.discriminator(self.x), self.discriminator(neg_inverse_fake))
        reg_real, g_ = self.ragan(self.discriminator(neg_inverse_real), self.discriminator(self.g))

        d_losses.append(self.gamma2*(reg_fake+reg_real))
        g_losses.append(self.g_gamma2 * g_)

        inverse_fake = self.inverse(self.d_real, self.discriminator(self.target1_g), self.target1_g)[0]
        inverse_real = self.inverse(self.discriminator(self.target1_x), self.d_fake, self.target1_x)[0]

        reg_fake, g_ = self.ragan(self.discriminator(inverse_fake), self.discriminator(self.g))
        reg_real = self.ragan(self.discriminator(self.x), self.discriminator(inverse_real))[0]

        d_losses.append(self.gamma1*(reg_fake+reg_real))
        g_losses.append(self.g_gamma1 * g_)

        return sum(d_losses)/len(d_losses), sum(g_losses)/len(g_losses)

    def inverse(self, d_real, d_fake, target):
        loss = self.ragan(d_fake, d_real)[0]
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
