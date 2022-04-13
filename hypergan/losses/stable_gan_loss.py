import hypergan
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
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
        self.target_x = None
        self.target_g = None
        self.fake_target_x = None
        self.fake_target_g = None
        self.reg_fake_target = None
        self.reg_target = None
        self.loss = self.gan.create_component("loss")
        if gammas is None:
            d_gammas = [10.0, 10.0, 1e6, 1e6]
        else:
            d_gammas = gammas
        self.d_gammas = [torch.tensor(gamma,device=device) for gamma in d_gammas]
        self.g_gamma = torch.tensor(10.0,device=device)
        self.fake_g_gamma = torch.tensor(0.1,device=device)
        self.inverse_gamma = torch.tensor(1e3,device=device)

    def stable_loss(self, discriminator, xs, gs, d_fake = None, d_real = None):
        d_losses = []
        g_losses = []
        for form in ["teacher", "teacher_fake", "regularize_fake", "regularize_real"]:
            adv_d_loss, adv_g_loss = self.adversarial_norm(form, xs[0], gs[0], discriminator, d_real, d_fake) # TODO multiple
            self.gan.add_metric(form+'d', adv_d_loss.mean())
            d_losses.append(adv_d_loss)
            if adv_g_loss is not None:
                self.gan.add_metric(form+'g', adv_g_loss.mean())
                g_losses.append(adv_g_loss)
        return sum(d_losses), sum(g_losses)

    def adversarial_norm(self, form, x, g, discriminator, d_real, d_fake):
        if form == "teacher":
            if self.target_x is None:
                self.target_g = Parameter(g, requires_grad=True)
                self.target_x = Parameter(x, requires_grad=True)
            self.target_x.data = x.data.clone()
            self.target_g.data = g.data.clone()
            teacher_fake = self.inverse(d_real, discriminator(self.target_g), self.target_g)
            teacher_real = self.inverse(discriminator(self.target_x), d_fake, self.target_x)

            reg_fake, g_ = self.loss.forward(discriminator(teacher_fake), d_fake)
            reg_real, _ = self.loss.forward(d_real, discriminator(teacher_real))

            return self.d_gammas[0]*(reg_fake+reg_real), self.g_gamma* g_

        if form == "teacher_fake":
            if self.fake_target_x is None:
                self.fake_target_g = Parameter(g, requires_grad=True)
                self.fake_target_x = Parameter(x, requires_grad=True)

            self.fake_target_x.data = x.data.clone()
            self.fake_target_g.data = g.data.clone()
            neg_teacher_fake = self.inverse(discriminator(self.fake_target_g), d_real, self.fake_target_g)
            neg_teacher_real = self.inverse(d_fake, discriminator(self.fake_target_x), self.fake_target_x)

            reg_fake, _ = self.loss.forward(d_real, discriminator(neg_teacher_fake))
            reg_real, g_ = self.loss.forward(discriminator(neg_teacher_real), d_fake)

            return self.d_gammas[1]*(reg_fake+reg_real), self.fake_g_gamma*g_

        if form == "regularize_fake":
            if self.reg_fake_target is None:
                self.reg_fake_target = Parameter(g, requires_grad=True)
            self.reg_fake_target.data = g.data.clone()
            reg_g = self.regularize_adversarial_norm(d_real, discriminator(self.reg_fake_target), self.reg_fake_target)
            norm = (reg_g**2).mean()
            return self.d_gammas[2] * norm.mean(), None

        if form == "regularize_real":
            if self.reg_target is None:
                self.reg_target = Parameter(x, requires_grad=True)
            self.reg_target.data = x.data.clone()
            reg_x = self.regularize_adversarial_norm(d_fake, discriminator(self.reg_target), self.reg_target)
            norm = (reg_x**2).mean()
            return self.d_gammas[3] * norm.mean(), None

        raise "Invalid form"

    def inverse(self, d_real, d_fake, target):
        loss = self.loss.forward(d_fake, d_real)[0]# * self.inverse_gamma
        grads = torch_grad(outputs=loss, inputs=[target], retain_graph=True, create_graph=True, only_inputs=True)
        return target + grads[0]

    def regularize_adversarial_norm(self, d_real, d_fake, target):
        loss = ((d_real - d_fake)**2).mean()
        grads = torch_grad(outputs=loss, inputs=[target], retain_graph=True, create_graph=True, allow_unused=True)
        return grads[0]
