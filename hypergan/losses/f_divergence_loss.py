import numpy as np
import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss

TINY=1e-8
class FDivergenceLoss(BaseLoss):
    def __init__(self, gan, config):
        super(FDivergenceLoss, self).__init__(gan, config)
        self.tanh = torch.nn.Tanh()

    def _forward(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        gfx = None
        gfg = None

        pi = config.pi or 2

        g_loss_type = config.g_loss_type or config.type or 'gan'
        d_loss_type = config.type or 'gan'

        alpha = config.alpha or 0.5

        if d_loss_type == 'kl':
            bounded_x = torch.clamp(d_real, max=np.exp(9.))
            bounded_g = torch.clamp(d_fake, max=10.)
            gfx = bounded_x
            gfg = bounded_g
        elif d_loss_type == 'js':
            gfx = np.log(2) - (1+(-d_real).exp()).log()
            gfg = np.log(2) - (1+(-d_fake).exp()).log()
        elif d_loss_type == 'js_weighted':
            gfx = -pi*np.log(pi) - (1+(-d_real).exp()).log()
            gfg = -pi*np.log(pi) - (1+(-d_fake).exp()).log()
        elif d_loss_type == 'gan':
            gfx = -(1+(-d_real).exp()).log()
            gfg = -(1+(-d_fake).exp()).log()
        elif d_loss_type == 'reverse_kl':
            gfx = -d_real.log()
            gfg = -d_fake.log()
        elif d_loss_type == 'pearson' or d_loss_type == 'jeffrey' or d_loss_type == 'alpha2':
            gfx = d_real
            gfg = d_fake
        elif d_loss_type == 'squared_hellinger':
            gfx = 1-(-d_real).exp()
            gfg = 1-(-d_fake).exp()
        elif d_loss_type == 'neyman':
            gfx = 1-(-d_real).exp()
            gfx = torch.clamp(gfx, max=1.9)
            gfg = 1-(-d_fake).exp()

        elif d_loss_type == 'total_variation':
            gfx = 0.5*self.tanh(d_real)
            gfg = 0.5*self.tanh(d_fake)

        elif d_loss_type == 'alpha1':
            gfx = 1./(1-alpha) - (1+(-d_real).exp()).log()
            gfg = 1./(1-alpha) - (1+(-d_fake).exp()).log()

        else:
            raise "Unknown type " + d_loss_type

        conjugate = None

        if d_loss_type == 'kl':
            conjugate = (gfg-1).exp()
        elif d_loss_type == 'js':
            bounded = torch.clamp(gfg, max=np.log(2.)-TINY)
            conjugate = -(2-(bounded).exp()).log()
        elif d_loss_type == 'js_weighted':
            c = -pi*np.log(pi)-TINY
            bounded = gfg
            conjugate = (1-pi)*((1-pi)/((1-pi)*(bounded/pi).exp())).log()
        elif d_loss_type == 'gan':
            conjugate = -(1-(gfg).exp()).log()
        elif d_loss_type == 'reverse_kl':
            conjugate = -1-(-gfg).log()
        elif d_loss_type == 'pearson':
            conjugate = 0.25 * (gfg**2)+gfg
        elif d_loss_type == 'neyman':
            conjugate = 2 - 2 * torch.sqrt(self.relu(1-gfg)+1e-2)
        elif d_loss_type == 'squared_hellinger':
            conjugate = gfg/(1.-gfg)
        elif d_loss_type == 'jeffrey':
            raise "jeffrey conjugate not implemented"

        elif d_loss_type == 'alpha2' or d_loss_type == 'alpha1':
            bounded = gfg
            bounded = 1./alpha * (bounded * ( alpha - 1) + 1)
            conjugate = (bounded ** ( alpha/(alpha - 1.))) - 1. / alpha

        elif d_loss_type == 'total_variation':
            conjugate = gfg
        else:
            raise "Unknown type " + d_loss_type

        gf_threshold  = None # f' in the paper

        if d_loss_type == 'kl':
            gf_threshold = 1
        elif d_loss_type == 'js':
            gf_threshold = 0
        elif d_loss_type == 'gan':
            gf_threshold = -np.log(2)
        elif d_loss_type == 'reverse_kl':
            gf_threshold = -1
        elif d_loss_type == 'pearson':
            gf_threshold = 0
        elif d_loss_type == 'squared_hellinger':
            gf_threshold = 0

        self.gf_threshold=gf_threshold

        d_loss = -gfx+conjugate
        g_loss = -conjugate

        if g_loss_type == 'gan':
            g_loss = -conjugate
        elif g_loss_type == 'total_variation':
            # The inverse of derivative(1/2*x - 1)) = 0.5
            # so we use the -conjugate for now
            g_loss = -conjugate
        elif g_loss_type == 'js':
            # https://www.wolframalpha.com/input/?i=inverse+of+derivative(-(u%2B1)*log((1%2Bu)%2F2)%2Bu*log(u))
            g_loss = -d_fake.exp()
        elif g_loss_type == 'js_weighted':
            # https://www.wolframalpha.com/input/?i=inverse+of+derivative(-(u%2B1)*log((1%2Bu)%2F2)%2Bu*log(u))
            p = pi
            u = d_fake
            exp_bounded = p/u
            exp_bounded = torch.clamp(exp_bounded, max=4.)
            inner = (-4.*u*(exp_bounded).exp() +np.exp(2.)*(u**2)-2.*np.exp(2.)*u+np.exp(2.))/(u**2)
            inner = self.relu(inner)
            u = torch.clamp(u, max=0.1)
            sqrt = torch.sqrt(inner+1e-2) / (2*np.exp(1))
            g_loss = (1.-u)/(2.*u)# + sqrt
        elif g_loss_type == 'pearson':
            g_loss = -(d_fake-2.0)/2.0
        elif g_loss_type == 'neyman':
            g_loss = 1./torch.sqrt(1-d_fake) # does not work, causes 'nan'
        elif g_loss_type == 'squared_hellinger':
            g_loss = -1.0/(((d_fake-1)**2)+1e-2)
        elif g_loss_type == 'reverse_kl':
            g_loss = -d_fake
        elif g_loss_type == 'kl':
            g_loss = -gfg * gfg.exp()
        elif g_loss_type == 'alpha1': 
            a = alpha
            bounded = d_fake
            g_loss = (1.0/(a*(a-1))) * ((a*bounded).exp() - 1 - a*((bounded).exp() - 1))
        elif g_loss_type == 'alpha2':
            a = alpha
            bounded = torch.clamp(d_fake, max=4.)
            g_loss = -(1.0/(a*(a-1))) * ((a*bounded).exp() - 1 - a*((bounded).exp() - 1))
        else:
            raise "Unknown g_loss_type " + g_loss_type

        if self.config.regularizer:
            g_loss += self.g_regularizer(gfg, gfx)

        return [d_loss, g_loss]

    def g_regularizer(self, gfg, gfx):
        regularizer = None
        config = self.config
        pi = config.pi or 2
        alpha = config.alpha or 0.5

        ddfc = 0

        if config.regularizer == 'kl':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(exp(t-1)))
            bounded = torch.clamp(gfg, max=4.)
            ddfc = (bounded - 1).exp()
        elif config.regularizer == 'js':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-log(2-exp(t))))
            ddfc = -(2*(gfg).exp()) / ((2-gfg.exp())**2+1e-2)
        elif config.regularizer == 'js_weighted':
            # https://www.wolframalpha.com/input/?i=derivative(derivative((1-C)*log(((1-C)%2F(1-C*exp(x%2FC)))))
            ddfc = -((pi-1)*(gfg/pi).exp())/(pi*((pi*(gfg/pi).exp()-1)**2))
        elif config.regularizer == 'gan':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-log(1-exp(t))))
            ddfc = (2*gfg.exp()) / ((1-gfg.exp())**2+1e-2)
        elif config.regularizer == 'reverse_kl':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-1-log(-x)))
            ddfc = 1.0/(gfg**2)
        elif config.regularizer == 'pearson':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(0.25*x*x+%2B+x))
            ddfc = 0.5
        elif config.regularizer == 'jeffrey':
            raise "jeffrey regularizer not implemented"
        elif config.regularizer == 'squared_hellinger': 
            # https://www.wolframalpha.com/input/?i=derivative(derivative(t%2F(1-t)))
            ddfc = 2 / (((gfg - 1) ** 3)+1e-2)
            #ddfc = 0
        elif config.regularizer == 'neyman':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(2-2*sqrt(1-t)))
            ddfc = 1.0/(2*((1-gfg ** 3/2.)))
        elif config.regularizer == 'total_variation':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(t))
            ddfc = 0
        elif config.regularizer == 'alpha1' or config.regularizer == 'alpha2':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(1%2FC*(x*(C-1)%2B1)%5E(C%2FC-1)-1%2FC))
            ddfc = -(((alpha - 1)*gfg+1) ** (1/(alpha-1)-1))
        regularizer = ddfc * torch.norm(gfg, 2, 0) * (config.regularizer_lambda or 1)
        self.add_metric('fgan_regularizer', regularizer.mean())
        return regularizer 
        
    def d_regularizers(self):
        return []
