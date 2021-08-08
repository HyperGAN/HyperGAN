import hypergan
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch import optim
import collections

class StableGANLoss:
    """
    Stablized gan loss. Subject to change.
    ```python
        loss = StableGANLoss()
        loss.stable_loss(discriminator, [x], [g])
    ```
    """
    def __init__(self, device='cuda:0'):
        self.one = torch.tensor(0.01, device=device)
        self.optimizer = None

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
        d_loss, g_loss = self.loss_fn(d_real, d_fake)
        g_losses.append(g_loss)
        fake_grad_sum = 10000
        real_grad_sum = 10000
        real_adv_loss = 1e8
        fake_adv_loss = 1e8
        # https://github.com/iceli1007/DAT-GAN (modified)
        i = 0
        if self.optimizer is None:
            self.fake_imgs_adv= [Variable(g.clone().detach(), requires_grad=True) for g in gs]
            self.real_imgs_adv= [Variable(x.clone().detach(), requires_grad=True) for x in xs]
            #self.optimizer = optim.SGD(self.fake_imgs_adv + self.real_imgs_adv, lr=1.0)
            #self.optimizer = optim.Adam(self.fake_imgs_adv + self.real_imgs_adv, lr=0.001)
            self.optimizer = torch.optim.RMSprop(self.fake_imgs_adv + self.real_imgs_adv, lr=0.01)
        else:
            for target, data in zip(self.fake_imgs_adv, gs):
                target.data = data.clone()
            for target, data in zip(self.real_imgs_adv, xs):
                target.data = data.clone()
        fake_imgs_adv = self.fake_imgs_adv
        real_imgs_adv = self.real_imgs_adv
        self.optimizer.state = collections.defaultdict(dict)
        while(True):
            real_value = d_real.mean()
            fake_value = d_fake.mean()

            fake_output= discriminator(*fake_imgs_adv)
            fake_output=fake_output.mean()
            fake_adv_loss = torch.abs(fake_output-real_value)
            fake_grad=torch.autograd.grad(fake_adv_loss,fake_imgs_adv)
            for grad, fake in zip(fake_grad, fake_imgs_adv):
                fake.grad = grad
            #fake_grad_sum = sum([f.abs().sum() for f in fake_grad])
            #print('i', i, 'fake',fake_grad_sum, 'loss', fake_adv_loss)
            real_output= discriminator(*real_imgs_adv)
            real_output=real_output.mean()
            real_adv_loss = torch.abs(real_output-fake_value)
            real_grad=torch.autograd.grad(real_adv_loss,real_imgs_adv)
            for grad, real in zip(real_grad, real_imgs_adv):
                real.grad = grad

            real_grad_sum = sum([f.abs().sum() for f in real_grad])
            #print('i', i, 'real',real_grad_sum, 'loss', real_adv_loss)
            self.optimizer.step()

            fake_adv_validity= discriminator(*[a.clone().detach() for a in fake_imgs_adv])
            real_adv_validity = discriminator(*[a.clone().detach() for a in real_imgs_adv])
            next_adv_loss = torch.abs(real_adv_validity.mean() - fake_adv_validity.mean())
            if(next_adv_loss > fake_adv_loss):# redundant or fake_adv_loss > self.one):
                break

            d_loss, g_loss = self.loss_fn(real_adv_validity, fake_adv_validity)

            d_fake = fake_adv_validity
            d_real = real_adv_validity
            gs = fake_imgs_adv
            xs = real_imgs_adv
            i += 1
            if i%10 == 0 and i > 100:
                print("Stable gan loss training step", i, "fake loss", float(fake_adv_loss.cpu()), "real loss", float(real_adv_loss.cpu()))
        print('si', i, fake_adv_loss)
        d_losses.append(d_loss)
        d_loss, g_loss = self.loss_fn(d_real_prime, d_real)
        d_losses.append(d_loss)
        d_loss, g_loss = self.loss_fn(d_fake, d_fake_prime)
        d_losses.append(d_loss)

        return sum(d_losses), sum(g_losses)
