import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class FastTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        if self.config.use_generator:
            self.decoder = gan.generator
        else:
            self.decoder = gan.create_component("decoder", defn=self.config.decoder, input=self.gan.discriminator.named_layers["f1"])
        gan.decoded = torch.zeros_like(gan.inputs.next())
        self.loss = self.gan.initialize_component("loss")
        gan.decoded_g = torch.zeros_like(gan.inputs.next())
        self.re_x = None
        self.target = [Parameter(x, requires_grad=True) for x in self.gan.discriminator_real_inputs()]

    def forward(self, d_loss, g_loss):
        if self.re_x is None:
            self.re_x = self.gan.x
        if self.config.gan is None:
            pass
        else:
            f1 = self.gan.discriminator.context[self.config.discriminator_layer_name]
            rex = self.decoder(f1)
            x = self.gan.x.to(self.decoder.device)
            d_fake = self.gan.forward_discriminator([rex])
            d_real = self.gan.d_real#self.gan.forward_discriminator([x])
            d_l, g_l = self.loss.forward(d_real, d_fake)
            if self.config.exp1:
                rex2 = self.decoder(torch.rand_like(f1)*2.0 -1.0)
                self.gan.decoded_g = rex2
                d_fake2 = self.gan.forward_discriminator([rex2])
                d_l2, g_l2 = self.loss.forward(d_real, d_fake2)
                self.gan.add_metric("gl2", g_l2)
                self.gan.add_metric("dl2", d_l2)
                d_l += d_l2
                g_l += g_l2
            self.gan.add_metric("ft_df", d_fake.mean())
            d_fake = self.gan.forward_discriminator([self.re_x])
            f1 = self.gan.discriminator.context[self.config.discriminator_layer_name]
            rex2 = self.decoder(f1)
            self.gan.decoded = rex
 
            #loss, _, mod_target = self.regularize_adversarial_norm(d_fake, d_real, self.target)
            #norm = (-((mod_target[0] - rex)**2)).mean()

            #d_real = self.gan.forward_discriminator([mod_target])
            #d_fake = self.gan.forward_discriminator([rex])
            #d_l2, g_l2 = self.gan.trainable_gan.loss.forward(d_real, d_fake)
 
            return d_l,g_l#+d_l2-1e4*norm, g_l+g_l2
        f1 = self.gan.discriminator.context[self.config.discriminator_layer_name]
        l_recon = torch.nn.L1Loss(reduction="mean")(self.decoder(f1),self.gan.x.to(self.decoder.device)) * 10
        #self.gan.decoded = self.decoder(f1)
        d_fake = self.gan.forward_discriminator([self.re_x])
        f1 = self.gan.discriminator.context[self.config.discriminator_layer_name]
        rex = self.decoder(f1)
        self.gan.decoded = rex
        self.gan.add_metric('d_ae', l_recon)
        if self.config.g:
            return None, l_recon
        else:
            return l_recon, None

    def discriminator_components(self):
        if self.config.gan:
            return []
        if self.config.g:
            return []
        return [self.decoder]

    def generator_components(self):
        if self.config.gan:
            return [self.decoder]
        if self.config.g:
            return [self.decoder]
        return []

    def regularize_adversarial_norm(self, d1_logits, d2_logits, target):
        loss = self.forward_adversarial_norm(d1_logits, d2_logits)

        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True)
        mod_target = [_d1 + _t for _d1, _t in zip(d1_grads, target)]

        return loss, None, mod_target

    def forward_adversarial_norm(self, d_real, d_fake):
        #return (torch.sign(d_real-d_fake)*((d_real - d_fake)**2)).mean()
        return ((d_real - d_fake)**2).mean()
        #return 0.5 * (self.dist(d_real,d_fake) + self.dist(d_fake, d_real)).sum()
