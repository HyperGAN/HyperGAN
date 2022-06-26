import torch
from torchvision import transforms
import hyperchamber as hc
import numpy as np
import inspect
from torch.nn import functional as F
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from PIL import Image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class LatentCrossEntropy(BaseTrainHook):

    def forward(self, d_loss, g_loss):
        latent = self.gan.augmented_latent
        target = (latent > 0).float()
        prediction = self.gan.discriminator.context["z_prediction"]
        loss = F.cross_entropy(prediction, target)# * 0.05
        self.add_metric('ce', loss)
        return loss, loss
