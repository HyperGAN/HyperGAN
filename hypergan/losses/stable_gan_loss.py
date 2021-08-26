import hypergan
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class StableGANLoss:
    """
    Stablized gan loss. Subject to change.
    ```python
        loss = StableGANLoss()
        loss.stable_loss(discriminator, [x], [g])
    ```

    Solutions generated with GPT prompt:

        def stable_gan_loss(discriminator, xs, gs):
        \"\"\"
          pytorch optimized stable GAN loss
          This technique quickly converges to the target distribution. It's fast, it works across many architectures and hyperparameters and converges to the RBBR(resource bound best response).
          See our paper linked on github.

          Inputs:
            discriminator - the pytorch discriminator function
            xs - real data of the shape (bs, c, h, w)
            gs - generated data of the shape (bs, c, h, w)
          Returns:
            A tuple of (d_loss, g_loss)
        \"\"\"
        d_fake = discriminator(gs.detach())
        d_real = discriminator(xs)
    """
    def stable_loss(self, discriminator, xs, gs, d_fake=None, d_real=None, form=0):
        if d_fake is None:
            d_fake = discriminator(*gs.detach())
        if d_real is None:
            d_real = discriminator(*xs)

        if form==0:
            #GAN loss
            d_loss_real = -torch.mean(d_real, dim=1)
            d_loss_fake = torch.mean(d_fake, dim=1)
            d_loss = d_loss_real + d_loss_fake
            #Entropy loss
            uncertainty = torch.var(d_fake)
            d_loss += uncertainty
            return d_loss, -d_loss
        elif form==1:
            #Need to first calculate the GAN loss.
            #Note that the disciminator should predict ones for all real samples.
            #Then for stability we calculate the gradient of the discriminator output with respect to the input.
            loss = -torch.mean(torch.log(d_real) + torch.log(1-d_fake))
            #This gradient will be 0 if the discriminator perfectly segments between real and generated.
            #This is the ideal case for the discriminator so we're going to use this to approximate the derivative of the discriminator loss with respect to its input.
            xs[0].requires_grad = True
            input_grad = torch.autograd.grad(outputs=loss, inputs=xs, retain_graph=True)[0]
            #We then update the input using this gradient.
            #The update value is scaled such that the optimal discriminator behavior is achieved.
            xs.data.add_(-0.01 * input_grad.data)
            return real_loss, fake_loss
        elif form==2:
            #NaN
            uncertainty = torch.mean(d_real ** 2) + torch.mean((1 - d_fake) ** 2)
            divergence = torch.mean(d_real * torch.log(d_real / (d_fake + 1e-8))) + torch.mean((1 - d_fake) * torch.log((1 - d_fake) / (d_real + 1e-8)))
            loss= uncertainty - divergence
            return loss,loss
        elif form==3:
            gamma = 10
            uncertainty = torch.mean(torch.pow(d_real - torch.mean(d_fake), 2))
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gamma * uncertainty
            g_loss = -torch.mean(d_fake)


            return d_loss, g_loss

        elif form==4:
            self.device = 'cuda:0'
            # WGAN loss
            d_loss = torch.mean(d_fake) - torch.mean(d_real)
            g_loss = -torch.mean(d_fake)
            # Gradient penalty
            lambda_gp = 10.0
            alpha = torch.rand(xs[0].size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * xs[0].data + (1 - alpha) * gs[0].data).requires_grad_(True)
            d_hat = discriminator(x_hat)
            grad = torch.autograd.grad(
                outputs=d_hat,
                inputs=x_hat,
                grad_outputs=torch.ones(d_hat.size()).to(self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
            d_loss_gp = lambda_gp * ((grad_norm - 1)**2).mean()
            d_loss = d_loss + d_loss_gp
            return d_loss, g_loss
        elif form==5:
            stable_d_loss = (F.relu(1 - d_real) + F.relu(1 + d_fake)).mean()
            return stable_d_loss, -stable_d_loss
        elif form==6:
            stable_d_loss = -0.5 * torch.mean(d_real) + 0.5 * torch.mean(torch.nn.functional.softplus(d_real)) + 0.5 * torch.mean(torch.nn.functional.softplus(-d_fake))
            stable_g_loss = -torch.mean(torch.nn.functional.softplus(-d_fake))
            return stable_d_loss, stable_g_loss
        elif form == 110:
            eps = 1e-8
            d_fake = F.sigmoid(d_fake)
            d_real = F.sigmoid(d_real)
            g_mode_finding_loss = torch.log(d_fake+eps).mean()
            d_mode_finding_loss = torch.log(1 - d_fake+eps).mean() + torch.log(d_real+eps).mean()
            d_loss = -(d_mode_finding_loss + g_mode_finding_loss) / 2
            g_loss = -g_mode_finding_loss
            return d_loss, g_loss
        elif form == 115:
            # This technique solves mode collapse by introducing a regularization term that causes the loss to approach zero when the discriminator outputs are indistinguishable
            loss_d = (-torch.mean(d_real) + torch.mean(d_fake)) + (torch.mean(d_real ** 2) + torch.mean(d_fake ** 2)) / 2
            loss_g = -torch.mean(d_fake)
            return loss_d, loss_g
        elif form==7:
            uncertainty = d_fake.std() + d_real.std()
            loss = d_fake.mean() - d_real.mean() + 1.0 * uncertainty
            return loss, -loss
        elif form==8:
            d_loss = F.relu(d_real - d_fake + 1.0).mean()
            return d_loss, -d_loss
        elif form==9:
            gs = gs[0]
            xs = xs[0]
            uncertainty = (d_real ** 2).mean()
            d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean() + 0.1 * uncertainty
            g_loss = -d_fake.mean()
            # Now avoid bad neighborhoods
            noise = torch.randn_like(gs)
            perturbation = noise * (1 - gs)
            d_perturbed = discriminator(gs + perturbation)
            d_perturbed_loss = F.relu(1 + d_perturbed).mean()
            d_loss += 0.5 * d_perturbed_loss
            return d_loss, g_loss
        elif form==10:
            gs = gs[0]
            xs = xs[0]
            uncertainty = torch.std(d_real.detach(), dim=0)
            uncertainty = uncertainty.mean()
            d_loss = F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean() + 5.*uncertainty
            g_loss = -d_fake.mean() + uncertainty
            # Then to avoid mode collapse
            num_gen = gs.shape[0]
            noise = torch.randn(num_gen, 1, 1, 1, device=d_real.device)
            gs_ = gs + noise
            d_fake_ = discriminator(gs_)
            d_loss += F.relu(1. + d_fake_).mean()
            return d_loss, g_loss
        elif form==11:
            gs = gs[0]
            xs = xs[0]
            bs = xs.shape[0]
            alpha = torch.rand(bs, 1, 1, 1, device=xs.device)
            alpha = alpha.expand_as(xs)
            x_hat = alpha * xs + (1 - alpha) * gs
            x_hat = x_hat.detach()
            x_hat.requires_grad = True
            d_hat = discriminator(x_hat)
            gradients = torch_grad(
                outputs=d_hat,
                inputs=x_hat,
                grad_outputs=torch.ones_like(d_hat),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
            d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
            g_loss = -d_fake.mean()
            return d_loss, g_loss
        elif form==12:
            gs = gs[0]
            xs = xs[0]
            bs = xs.shape[0]
            d_loss = torch.mean(d_real - d_fake)
            g_loss = torch.mean(d_fake)
            d_grad = torch.autograd.grad(d_loss, discriminator.parameters(), retain_graph=True)
            d_grad_norm = torch.norm(torch.cat([g.view(-1) for g in d_grad]))
            d_grad_norm_clipped = d_grad_norm.clamp(max=0.5)
            d_grad_penalty = (d_grad_norm_clipped - 0.5).pow(2)
            d_loss = d_loss + 10 * d_grad_penalty
            return d_loss, g_loss
        elif form==13:
            uncertainty = torch.exp(torch.log(torch.abs(d_real - d_fake)))
            mode_seek = d_real * (1.0 - d_fake) + d_fake * (1.0 - d_real)
            # Loss to seek mode in the most stable manner possible. 
            d_loss = (1.0 - uncertainty) ** 2 / 2.0
            g_loss = (1.0 - d_fake) ** 2 / 2.0
            return d_loss, g_loss
        elif form==14:
            stable_d_loss = (d_fake - d_real + 1.0).mean()
            stable_g_loss = (d_fake - d_real - 1.0).mean()
            return stable_d_loss, stable_g_loss
        elif form==15:
            # Section 2.4, extremely fast convergence shortcut. This also mode seeks, avoids bad neighborhoods, and allows any D or G architecture. Works with SGD learning rate 1.
            uncertainty = 0.5 * (d_real.std() + d_fake.std())
            # Section 2.5
            mode_seeking = (d_real.mean() - d_fake.mean()) ** 2
            # Section 2.6. Or, if you're really into the flow paper and don't want to think too hard, it's just d_fake.mean() - d_real.mean()
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + 0.1 * uncertainty + 0.1 * mode_seeking
            g_loss = -torch.mean(d_fake)
            return d_loss, g_loss
        elif form==16:
            shortcut = d_real.clone()
            shortcut[shortcut > 0] = 1
            shortcut[shortcut <= 0] = 0
            d_loss = -torch.mean(torch.log(F.sigmoid(d_real) + 1e-8) + torch.log(1 - F.sigmoid(d_fake) + 1e-8)) + \
                torch.mean(torch.mul(shortcut, torch.log(F.sigmoid(d_real) + 1e-8)))
            g_loss = -torch.mean(torch.log(F.sigmoid(d_fake) + 1e-8))
            return d_loss, g_loss

