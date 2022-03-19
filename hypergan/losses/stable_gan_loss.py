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
    def __init__(self, device='cuda:0', gammas=[1.0, 10.0, 1.0, 0, 100.0], gan=None):
        self.gan = gan
        if gammas[0] == 0:
            self.gamma1 = None
        else:
            self.gamma1 = torch.tensor(gammas[0], device=device)
        if gammas[1] == 0:
            self.g_gamma1 = None
        else:
            self.g_gamma1 = torch.tensor(gammas[1], device=device)
        if gammas[2] == 0:
            self.gamma2 = None
        else:
            self.gamma2 = torch.tensor(gammas[2], device=device)
        if gammas[3] == 0:
            self.g_gamma2 = None
        else:
            self.g_gamma2 = torch.tensor(gammas[3], device=device)
        self.inverse_gamma = torch.tensor(gammas[4], device=device)
        self.softplus = torch.nn.Softplus(1, 20)
        self.target1_x = None
        self.target1_g = None
        self.target2_x = None
        self.target2_g = None

    def loss_fn(self, d_real, d_fake):
        #cr = torch.mean(d_real,0)
        #cf = torch.mean(d_fake,0)
        #criterion = torch.nn.BCEWithLogitsLoss()
        #g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
        #d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        #g_loss = criterion(d_fake, torch.ones_like(d_fake))
        #d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        #g_loss = torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        #d_loss = torch.log(torch.sigmoid(d_real) + 1e-13) +torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        #d_loss = (d_real - d_fake).mean()
        #g_loss = d_fake.mean()
        d_loss = (self.softplus(-d_real) + self.softplus(d_fake)).mean()
        g_loss = self.softplus(-d_fake).mean()
        return d_loss, g_loss

    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None):
        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_losses = []
        g_losses = []
        d_loss, g_loss = self.loss_fn(d_real, d_fake)
        #d_losses.append(d_loss)
        #g_losses.append(g_loss)
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
        #if self.gan is not None:
        #    self.gan.add_metric('nif_diff', (neg_inverse_fake[0] - self.target2_g[0]).abs().sum())
        #    self.gan.add_metric('nir_diff', (neg_inverse_real[0] - self.target2_x[0]).abs().sum())

        reg_fake, _ = self.loss_fn(d_real, discriminator(*neg_inverse_fake))
        reg_real, g_ = self.loss_fn(discriminator(*neg_inverse_real), d_fake)

        if self.gamma2 is not None:
            d_losses.append(self.gamma2*(reg_fake+reg_real))
        if self.g_gamma2 is not None:
            g_losses.append(self.g_gamma2 * g_)

        #if self.gan is not None:
        #    self.gan.add_metric('dl_s2', self.gamma2*(reg_fake+reg_real))
        #    self.gan.add_metric('gl_s2', self.g_gamma2*(g_))

        inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
        inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

        #if self.gan is not None:
        #    self.gan.add_metric('if_diff', (inverse_fake[0] - self.target1_g[0]).abs().sum())
        #    self.gan.add_metric('ir_diff', (inverse_real[0] - self.target1_x[0]).abs().sum())

        reg_fake, g_ = self.loss_fn(discriminator(*inverse_fake), d_fake)
        reg_real = self.loss_fn(d_real, discriminator(*inverse_real))[0]

        if self.gamma1 is not None:
            d_losses.append(self.gamma1*(reg_fake+reg_real))
        if self.g_gamma1 is not None:
            g_losses.append(self.g_gamma1 * g_)

        #if self.gan is not None:
        #    self.gan.add_metric('dl_s1', self.gamma1*(reg_fake+reg_real))
        #    self.gan.add_metric('gl_s1', self.g_gamma1*(g_))

        return sum(d_losses), sum(g_losses)

    def inverse(self, d_real, d_fake, target):
        #loss = (d_fake - d_real) * self.inverse_gamma
        loss = self.loss_fn(d_fake, d_real)[0]
        #d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        return [_t + _d1/_d1.abs().sum()*self.inverse_gamma for _d1, _t in zip(d1_grads, target)]
        #return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
