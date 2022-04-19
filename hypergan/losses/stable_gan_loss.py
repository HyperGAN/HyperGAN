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
        if d_fake is None:
            d_fake = discriminator(gs[0])
        if d_real is None:
            d_real = discriminator(xs[0])
        for form in ["teacher", "teacher_fake", "regularize_fake", "regularize_real"]:
            adv_d_loss, adv_g_loss = self.adversarial_norm(form, xs[0], gs[0], discriminator, d_real, d_fake) # TODO multiple
            self.gan.add_metric(form+'d', adv_d_loss.mean())
            d_losses.append(adv_d_loss)
            if adv_g_loss is not None:
                self.gan.add_metric(form+'g', adv_g_loss.mean())
                g_losses.append(adv_g_loss)
        return sum(d_losses), sum(g_losses)

    def ae_stable_loss(self, discriminator, x, g, d_fake = None, d_real = None):
        d_losses = []
        g_losses = []
        xx = torch.cat([x,x], axis=1)
        xg = torch.cat([x,g], axis=1)
        if d_fake is None:
            d_fake = discriminator(xg)
        if d_real is None:
            d_real = discriminator(xx)
        for form in ["teacher", "teacher_fake", "regularize_fake", "regularize_real"]:
            adv_d_loss, adv_g_loss = self.adversarial_ae_norm(form, x, g, discriminator, d_real, d_fake) # TODO multiple
            self.gan.add_metric(form+'d', adv_d_loss.mean())
            d_losses.append(adv_d_loss)
            if adv_g_loss is not None:
                self.gan.add_metric(form+'g', adv_g_loss.mean())
                g_losses.append(adv_g_loss)
        return sum(d_losses), sum(g_losses)

    def adversarial_ae_norm(self, form, x, g, discriminator, d_real, d_fake):
        if form == "teacher":
            if self.target_x is None:
                self.target_g = Parameter(g, requires_grad=True)
                self.target_x = Parameter(x, requires_grad=True)
                self.target_x2 = Parameter(x, requires_grad=True)
                self.target_x3 = Parameter(x, requires_grad=True)
            self.target_x.data = x.data.clone()
            self.target_x2.data = x.data.clone()
            self.target_x3.data = x.data.clone()
            self.target_g.data = g.data.clone()
            gg = torch.cat([self.target_x2, self.target_g], axis=1)
            xx = torch.cat([self.target_x3, self.target_x], axis=1)
            teacher_fake = self.ae_inverse(d_real, discriminator(gg), [self.target_x2, self.target_g])
            teacher_real = self.ae_inverse(discriminator(xx), d_fake, [self.target_x3, self.target_x])

            in_fake = torch.cat([teacher_fake[1], teacher_fake[1]], axis=1)
            in_real = torch.cat([teacher_real[0], teacher_real[1]], axis=1)
            #in_fake = torch.cat([teacher_real, teacher_fake], axis=1)
            #in_real = torch.cat([teacher_real, teacher_real], axis=1)
            #in_fake = teacher_fake
            #in_fake = torch.cat(teacher_fake, axis=1)
            #in_real = torch.cat(teacher_real, axis=1)
            #in_fake = teacher_real
            #in_real = teacher_fake
            reg_fake, g_ = self.loss.forward(discriminator(in_fake), d_fake)
            reg_real, _ = self.loss.forward(d_real, discriminator(in_real))

            return self.d_gammas[0]*(reg_fake+reg_real), self.g_gamma* g_

        if form == "teacher_fake":
            if self.fake_target_x is None:
                self.fake_target_g = Parameter(g, requires_grad=True)
                self.fake_target_x = Parameter(x, requires_grad=True)
                self.fake_target_x2 = Parameter(x, requires_grad=True)
                self.fake_target_x3 = Parameter(x, requires_grad=True)

            self.fake_target_x.data = x.data.clone()
            self.fake_target_x2.data = x.data.clone()
            self.fake_target_x3.data = x.data.clone()
            self.fake_target_g.data = g.data.clone()
            gg = torch.cat([self.fake_target_x2, self.fake_target_g], axis=1)
            xx = torch.cat([self.fake_target_x3, self.fake_target_x], axis=1)
            neg_teacher_fake = self.ae_inverse(discriminator(gg), d_real, [self.fake_target_x2, self.fake_target_g])
            neg_teacher_real = self.ae_inverse(d_fake, discriminator(xx), [self.fake_target_x3, self.fake_target_x])

            #in_fake = torch.cat([neg_teacher_real, neg_teacher_fake], axis=1)
            #in_real = torch.cat([neg_teacher_real, neg_teacher_real], axis=1)
            in_fake = torch.cat([neg_teacher_fake[1], neg_teacher_fake[1]], axis=1)
            in_real = torch.cat([neg_teacher_real[0], neg_teacher_real[1]], axis=1)
            #in_fake = neg_teacher_fake
            #in_real = neg_teacher_real
            #in_fake = teacher_real
            #in_real = teacher_fake
            reg_fake, _ = self.loss.forward(d_real, discriminator(in_fake))
            reg_real, g_ = self.loss.forward(discriminator(in_real), d_fake)

            return self.d_gammas[1]*(reg_fake+reg_real), self.fake_g_gamma*g_

        if form == "regularize_fake":
            if self.reg_fake_target is None:
                self.reg_fake_target = Parameter(g, requires_grad=True)
                self.reg_fake_target_x = Parameter(x, requires_grad=True)
            self.reg_fake_target.data = g.data.clone()
            self.reg_fake_target_x.data = x.data.clone()
            xg = torch.cat([self.reg_fake_target_x, self.reg_fake_target], axis=1)
            reg_g = self.regularize_adversarial_norm(d_real, discriminator(xg), xg)
            norm = (reg_g**2).mean()
            return self.d_gammas[2] * norm.mean(), None

        if form == "regularize_real":
            if self.reg_target is None:
                self.reg_target = Parameter(x, requires_grad=True)
                self.reg_target2 = Parameter(x, requires_grad=True)
            self.reg_target.data = x.data.clone()
            self.reg_target2.data = x.data.clone()
            xx = torch.cat([self.reg_target2, self.reg_target], axis=1)
            reg_x = self.regularize_adversarial_norm(d_fake, discriminator(xx), xx)
            norm = (reg_x**2).mean()
            return self.d_gammas[3] * norm.mean(), None

        raise "Invalid form"



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

    def ae_inverse(self, d_real, d_fake, targets):
        loss = self.loss.forward(d_fake, d_real)[0]# * self.inverse_gamma
        grads = torch_grad(outputs=loss, inputs=targets, retain_graph=True, create_graph=True, only_inputs=True)
        result = []
        for target, grad in zip(targets, grads):
            result.append(target+grad)
        return result


    def regularize_adversarial_norm(self, d_real, d_fake, target):
        loss = ((d_real - d_fake)**2).mean()
        grads = torch_grad(outputs=loss, inputs=[target], retain_graph=True, create_graph=True, allow_unused=True)
        return grads[0]
