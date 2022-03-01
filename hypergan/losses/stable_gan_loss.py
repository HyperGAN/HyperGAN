
import hypergan
import torch
import torch.nn.functional as F
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
        self.i = 0
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
        d_loss = F.softplus(d_fake) + F.softplus(-d_real)
        g_loss = F.softplus(-d_fake)
        #g_loss = torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        #d_loss = torch.log(torch.sigmoid(d_real) + 1e-13) +torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        return d_loss, g_loss

    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None, autoencode=False, form=None):
        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_losses = []
        g_losses = []

        if form == "c_transform":
            def c_transform(x, y):
                xorig = x
                bs = x.shape[0]
                xshape = x.shape
                x = x.reshape([1, -1])
                x = torch.tile(x, [bs, 1])
                #d_real = d_real.reshape([d_real.shape[0], 1])
                #d_real = torch.tile(d_real, [1, d_real.shape[0]])
                #d_fake = d_fake.reshape([1, d_fake.shape[0]])
                #d_fake = torch.tile(d_fake, [d_fake.shape[1], 1])
                y = y.reshape([-1, 1])
                y = torch.tile(y, [1, bs])
                #print(len(xshape))
                x = x.reshape([bs, bs, xshape[1], xshape[2], xshape[3]])
                y = y.reshape([bs, bs, xshape[1], xshape[2], xshape[3]])
                def c(_x, _y):
                    return (_x - _y).abs().mean()
                c_transform = torch.min(c(x,y)-discriminator(xorig), dim=1)
                return c_transform.values
            J1 = d_real - d_fake
            J2 = d_real + c_transform(gs[0], xs[0])#c_transform(d_real, d_fake)
            J3 = - d_fake - c_transform(xs[0], gs[0])#c_transform(d_fake, d_real)
            #J4 = c_transform(d_real, d_fake) - c_transform(d_fake, d_real)
            J4 = c_transform(gs[0], xs[0]) - c_transform(xs[0], gs[0])

            d_loss = J1
            j2_gate = J2 < J1
            j3_gate = J3 < J1
            j2_mask = torch.relu(torch.sign(j2_gate.float()))
            j3_mask = torch.relu(torch.sign(j3_gate.float())) * (1-j2_mask)
            d_loss = J2 * j2_mask + J3 * j3_mask + J1 *(1-j3_mask)
            g_loss = -J1
            self.i += 1
            if self.i % 10 == 0:
                print("J1 %.2e J2 %.2e J3 %.2e J4 %.2e" % ((1.-j3_mask).sum(), j2_mask.sum(), j3_mask.sum(), J4.mean()))
            return d_loss, g_loss

        d_loss, g_loss = self.loss_fn(d_real, d_fake)
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

        reg_fake2, _ = self.loss_fn(d_real, discriminator(*neg_inverse_fake))
        reg_real2, g2_ = self.loss_fn(discriminator(*neg_inverse_real), d_fake)

        inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
        inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

        reg_fake, g_ = self.loss_fn(discriminator(*inverse_fake), d_fake)
        reg_real = self.loss_fn(d_real, discriminator(*inverse_real))[0]

        if form == 99100:
            uncertainty = self.gamma1 * reg_fake.mean() + self.gamma2 * reg_fake2.mean() + self.g_gamma1 * g_ + self.g_gamma2 * g2_.mean()
            d_loss = d_loss + uncertainty
            g_loss = g_loss + reg_real.mean() + reg_real2.mean()

            d_loss = d_loss.mean()
            g_loss = g_loss.mean()

            d_losses.append(d_loss)
            g_losses.append(g_loss)
        else:
            d_losses.append(self.g_gamma1 * torch.max(torch.zeros(1, device=reg_fake.device), reg_fake - reg_fake2).mean())
            g_losses.append(self.g_gamma1 * reg_fake.mean())
            d_losses.append(self.g_gamma2 * torch.max(torch.zeros(1, device=reg_real.device), reg_real - reg_real2).mean())
            g_losses.append(self.g_gamma2 * reg_real.mean())


        g_loss = sum(g_losses) / len(g_losses)
        d_loss = sum(d_losses) / len(d_losses)
        return d_loss, g_loss

    def inverse(self, d_real, d_fake, target):
        #loss = (d_fake - d_real) * self.inverse_gamma
        loss = self.loss_fn(d_fake, d_real)[0]
        loss = loss.mean()
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        #return [_t/_t.norm() * self.inverse_gamma + _d1 for _d1, _t in zip(d1_grads, target)]
        return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
