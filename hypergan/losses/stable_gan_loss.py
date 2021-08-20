import hypergan
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
