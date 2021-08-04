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
        self.one = torch.tensor(1.0, device=device)

    def loss_fn(self, d_real, d_fake):
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        return d_loss, g_loss

    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None):
        if d_fake is None:
            d_fake = discriminator(*gs)
        if d_real is None:
            d_real = discriminator(*xs)
        d_fake_prime = d_fake
        d_real_prime = d_real
        d_losses = []
        g_losses = []
        fake_grad_sum = 10000
        real_grad_sum = 10000
        real_adv_loss = 1e8
        fake_adv_loss = 1e8
        # https://github.com/iceli1007/DAT-GAN (modified)
        i = 0
        while(real_adv_loss > self.one or fake_adv_loss > self.one):
            t=1
            real_value = d_real.mean()
            fake_value = d_fake.mean()
            fake_imgs_adv=[_g.clone() for _g in gs]
            real_imgs_adv=[_x.clone() for _x in xs]
            real_imgs_adv=[Variable(a, requires_grad=True) for a in real_imgs_adv]
            fake_imgs_adv=[Variable(a,requires_grad=True) for a in fake_imgs_adv]

            fake_output= discriminator(*fake_imgs_adv)
            fake_output=fake_output.mean()
            fake_adv_loss = torch.abs(fake_output-real_value)
            fake_grad=torch.autograd.grad(fake_adv_loss,fake_imgs_adv)
            fake_imgs_adv=[(a - _g) for a, _g in zip(fake_imgs_adv,fake_grad)]
            fake_grad_sum = sum([f.abs().sum() for f in fake_grad])
            #print('i', i, 'fake',fake_grad_sum, 'loss', fake_adv_loss)
            real_output= discriminator(*real_imgs_adv)
            real_output=real_output.mean()
            real_adv_loss = torch.abs(real_output-fake_value)
            real_grad=torch.autograd.grad(real_adv_loss,real_imgs_adv)
            real_imgs_adv=[(a - _g) for a, _g in zip(real_imgs_adv,real_grad)]
            real_grad_sum = sum([f.abs().sum() for f in real_grad])
            #print('i', i, 'real',real_grad_sum, 'loss', real_adv_loss)

            # if autoencode:
                #fake_adv_validity= discriminator(gs[0].detach(), fake_imgs_adv[1].detach())
                #real_adv_validity= discriminator(real_imgs_adv[0].detach(), fake_imgs_adv[0].detach())
            #else:
            fake_adv_validity= discriminator(*[a.detach() for a in fake_imgs_adv])

            real_adv_validity = discriminator(*real_imgs_adv)

            d_loss, g_loss = self.loss_fn(real_adv_validity, fake_adv_validity)
            gs = [x.detach() for x in fake_imgs_adv]
            xs = [x.detach() for x in real_imgs_adv]

            d_fake = discriminator(*gs)
            d_real = discriminator(*xs)
            i += 1
            if i > 100:
                print("Stable gan loss training step", i)
        d_losses.append(d_loss)
        d_loss, g_loss = self.loss_fn(d_real_prime, d_real)
        d_losses.append(d_loss)
        d_loss, g_loss = self.loss_fn(d_fake, d_fake_prime)
        g_losses.append(-d_loss)
        d_losses.append(d_loss)

        return sum(d_losses), sum(g_losses)
