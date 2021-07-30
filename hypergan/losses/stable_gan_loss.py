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

    def loss_fn(self, d_real, d_fake):
        #cr = torch.mean(d_real,0)
        #cf = torch.mean(d_fake,0)
        criterion = torch.nn.BCEWithLogitsLoss()
        #g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
        #d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        #g_loss = torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        #d_loss = torch.log(torch.sigmoid(d_real) + 1e-13) +torch.log(1-torch.sigmoid(d_fake) + 1e-13)
        return d_loss, g_loss

    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None, DATGAN=True):
        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_losses = []
        g_losses = []
        d_loss, g_loss = self.loss_fn(d_real, d_fake)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        if DATGAN:
            # From https://github.com/iceli1007/DAT-GAN
            t=1
            real_value = d_real.mean()
            fake_value = d_fake.mean()
            fake_imgs_adv=gs[0].clone()
            real_imgs_adv=xs[0].clone()
            real_imgs_adv=Variable(real_imgs_adv,requires_grad=True)
            fake_imgs_adv=Variable(fake_imgs_adv,requires_grad=True)

            fake_output= discriminator(fake_imgs_adv)
            fake_output=fake_output.mean()
            fake_adv_loss = torch.abs(fake_output-real_value)
            fake_grad=torch.autograd.grad(fake_adv_loss,fake_imgs_adv)
            fake_imgs_adv=fake_imgs_adv-fake_grad[0].clamp(-1*t,t)
            fake_imgs_adv=fake_imgs_adv.clamp(-1,1)
            real_output= discriminator(real_imgs_adv)
            real_output=real_output.mean()
            real_adv_loss = torch.abs(real_output-fake_value)
            real_grad=torch.autograd.grad(real_adv_loss,real_imgs_adv)
            real_imgs_adv=real_imgs_adv-real_grad[0].clamp(-1*t,t)
            fake_adv_validity= discriminator(fake_imgs_adv.detach())
            real_adv_validity = discriminator(real_imgs_adv)
            real_imgs_adv=real_imgs_adv.clamp(-1,1) 

            #return self.loss_fn(real_adv_validity, fake_adv_validity)
            d_loss, g_loss = self.loss_fn(real_adv_validity, fake_adv_validity)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        else:
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

            reg_fake, _ = self.loss_fn(d_real, discriminator(*neg_inverse_fake))
            reg_real, g_ = self.loss_fn(discriminator(*neg_inverse_real), d_fake)

            d_losses.append(self.gamma2*(reg_fake+reg_real))
            g_losses.append(self.g_gamma2 * g_)

            inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
            inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

            reg_fake, g_ = self.loss_fn(discriminator(*inverse_fake), d_fake)
            reg_real = self.loss_fn(d_real, discriminator(*inverse_real))[0]

            d_losses.append(self.gamma1*(reg_fake+reg_real))
            g_losses.append(self.g_gamma1 * g_)




        #else:
        #    if self.target1_x == None:
        #        self.target1_x = [Parameter(x, requires_grad=True) for x in xs]
        #        self.target2_x = [Parameter(x, requires_grad=True) for x in xs]
        #    for target, x in zip( self.target2_x, xs + xs):
        #        target.data = x.clone()

        #    #neg_inverse_fake = self.inverse(discriminator(*self.target2_g), d_real, self.target2_g)
        #    neg_inverse_real = self.inverse(d_fake, discriminator(*self.target2_x), self.target2_x)

        #    #reg_fake, _ = self.loss_fn(d_real, discriminator(*neg_inverse_fake))
        #    neg_reg_real, g_ = self.loss_fn(discriminator(*neg_inverse_real), d_fake)

        #    #d_losses.append(self.gamma2*(reg_fake+reg_real))
        #    #g_losses.append(self.g_gamma2 * g_)

        #    #inverse_fake = self.inverse(d_real, discriminator(*self.target1_g), self.target1_g)
        #    inverse_real = self.inverse(discriminator(*self.target1_x), d_fake, self.target1_x)

        #    #reg_fake, g_ = self.loss_fn(discriminator(*inverse_fake), d_fake)
        #    reg_real = self.loss_fn(d_real, discriminator(*inverse_real))[0]

        #    d_losses.append(self.gamma1*(neg_reg_real+reg_real))
        #    g_losses.append(self.g_gamma1 * g_)

        return sum(d_losses), sum(g_losses)

    def inverse(self, d_real, d_fake, target):
        #loss = (d_fake - d_real) * self.inverse_gamma
        #loss = self.ragan(d_fake, d_real)[0]
        loss = self.loss_fn(d_fake, d_real)[0]
        #loss = loss.mean()
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        #return [_t + _d1/_d1.norm() for _d1, _t in zip(d1_grads, target)]
        return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
