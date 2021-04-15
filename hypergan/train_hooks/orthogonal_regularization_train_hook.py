import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class OrthogonalRegularizationTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)

    def forward(self, d_loss, g_loss):
        with torch.enable_grad():
            reg = 1e-6
            orth_loss = torch.zeros(1, device=d_loss.device)
            for name, param in self.gan.generator.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0], device=param_flat.device)
                    orth_loss = orth_loss + (reg * sym.abs().sum())
            self.gan.add_metric('orth_loss', orth_loss.mean())
            return None, orth_loss.mean()
