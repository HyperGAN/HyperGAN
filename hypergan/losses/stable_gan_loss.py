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
    def __init__(self, device='cuda:0', gammas=None, offsets=None, gan=None, metric_name=''):
        self.gan = gan
        self.metric_name = metric_name
        if gammas is None:
            gammas = [0, 10.0, 10.0, 0, 100.0]
        if offsets is None:
            offsets = [10,10,10,10]
        if(len(gammas) <= 5):
            gammas += [0,0,1e3,0]
        self.offsets = []
        for offset in offsets:
            self.offsets.append(torch.tensor(offset, device=device))
        self.gammas = []
        for gamma in gammas:
            if gamma is None or gamma == 0:
                self.gammas.append(None)
            else:
                self.gammas.append(torch.tensor(gamma, device=device))
        self.gamma1 = self.gammas[0]
        self.g_gamma1 = self.gammas[1]
        self.gamma2 = self.gammas[2]
        self.g_gamma2 = self.gammas[3]
        self.inverse_gamma = torch.tensor(gammas[4], device=device)

        print(self.gammas)
        self.l_gammas = self.gammas[5:]

        self.l_offsets = offsets

        self.softplus = torch.nn.Softplus(1, 20)
        self.target1_x = None
        self.target1_g = None
        self.target2_x = None
        self.target2_g = None

    def loss_fn(self, d_real, d_fake):
        if self.gan is None:
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
        else:
            return self.gan.trainable_gan.loss.forward(d_real, d_fake)

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

        g1, neg_inverse_fake = self.inverse(discriminator(*self.target2_g), d_real, self.target2_g)
        g2, neg_inverse_real = self.inverse(d_fake, discriminator(*self.target2_x), self.target2_x)
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

        g3, inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
        g4, inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

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
        #g_losses.append(-g4[0].abs().mean()*self.l_gammas[3])
        #g_losses.append(-g1[0].abs().mean()*self.l_gammas[0])

        def add_l_loss_for(index, name, g):
            if self.l_gammas[index] is not None:
                result = torch.relu(g[0].abs().mean()-self.offsets[index])*self.l_gammas[index]
                d_losses.append(result)
                if self.gan is not None:
                    self.gan.add_metric(self.metric_name+name, g[0].abs().mean())
                    self.gan.add_metric(self.metric_name+name+"-", result)
            else:
                return None
            return result
        add_l_loss_for(0, 'l0', g1)
        add_l_loss_for(1, 'l1', g2)
        add_l_loss_for(2, 'l2', g3)
        add_l_loss_for(3, 'l3', g4)
        return sum(d_losses), sum(g_losses)

    def inverse(self, d_real, d_fake, target):
        #loss = (d_fake - d_real) * self.inverse_gamma
        loss = self.loss_fn(d_fake, d_real)[0]
        #loss = (d_real - d_fake)
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        return d1_grads, [_t + _d1/_d1.abs().sum()*self.inverse_gamma for _d1, _t in zip(d1_grads, target)]
        #return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
