import hypergan
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class StableGANLoss:
    """
    Stablized gan loss. Subject to change.
    ```python
        loss = StableGANLoss()
        loss.stable_loss(discriminator, [x], [g])
    ```
    """
    def __init__(self, device='cuda:0'):
        self.gamma1 = torch.tensor(10.0, device=device)
        self.g_gamma1 = torch.tensor(10.0, device=device)
        self.gamma2 = torch.tensor(10.0, device=device)
        self.g_gamma2 = torch.tensor(0.1, device=device)
        self.inverse_gamma = torch.tensor(1e3, device=device)
        self.target1_x = None
        self.target1_g = None
        self.target2_x = None
        self.target2_g = None

    def ragan(self, d_real, d_fake):
        cr = torch.mean(d_real,0)
        cf = torch.mean(d_fake,0)
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
        d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        return d_loss, g_loss

    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None):
        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_losses = []
        g_losses = []
        d_loss, g_loss = self.ragan(d_real, d_fake)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        if self.target1_x == None:
            self.target1_g = [Parameter(g, requires_grad=True) for g in gs]
            self.target1_x = [Parameter(x, requires_grad=True) for x in xs]
            self.target2_g = [Parameter(g, requires_grad=True) for g in gs]
            self.target2_x = [Parameter(x, requires_grad=True) for x in xs]
        for target, g in zip(self.target1_g + self.target2_g, gs + gs):
            target.data = g.clone()
        for target, x in zip(self.target1_x + self.target2_x, xs + xs):
            target.data = x.clone()

        neg_inverse_fake = self.inverse(discriminator(*self.target2_g), d_real, self.target2_g)
        neg_inverse_real = self.inverse(d_fake, discriminator(*self.target2_x), self.target2_x)

        reg_fake, _ = self.ragan(d_real, discriminator(*neg_inverse_fake))
        reg_real, g_ = self.ragan(discriminator(*neg_inverse_real), d_fake)

        d_losses.append(self.gamma2*(reg_fake+reg_real))
        g_losses.append(self.g_gamma2 * g_)

        inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
        inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

        reg_fake, g_ = self.ragan(discriminator(*inverse_fake), d_fake)
        reg_real = self.ragan(d_real, discriminator(*inverse_real))[0]

        d_losses.append(self.gamma1*(reg_fake+reg_real))
        g_losses.append(self.g_gamma1 * g_)

        return sum(d_losses), sum(g_losses)

    def inverse(self, d_real, d_fake, target):
        #loss = (d_fake - d_real) * self.inverse_gamma
        loss = self.ragan(d_fake, d_real)[0]
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
