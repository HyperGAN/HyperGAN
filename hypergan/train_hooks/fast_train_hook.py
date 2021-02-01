import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class FastTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.decoder = gan.create_component("decoder", defn=self.config.decoder, input=self.gan.discriminator.named_layers["f1"])
        gan.decoded = torch.zeros_like(gan.inputs.next())

    def forward(self, d_loss, g_loss):
        f1 = self.gan.discriminator.context[self.config.discriminator_layer_name]
        l_recon = torch.nn.L1Loss(reduction="mean")(self.decoder(f1),self.gan.x.to(self.decoder.device)) * 10
        self.gan.decoded = self.decoder(f1)
        self.gan.add_metric('d_ae', l_recon)
        return l_recon, None

    def discriminator_components(self):
        return [self.decoder]
