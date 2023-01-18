import torch
from torchvision import transforms
import hyperchamber as hc
import open_clip
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
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts

class TeacherNetworkTrainHook(BaseTrainHook):

    def create(self):
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')
        self.perceptor = model.eval().requires_grad_(False).to('cuda:0')
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

    def forward(self, d_loss, g_loss):
        layer =  self.perceptor.encode_image(self.make_cutouts(self.normalize(self.gan.x/2+0.5)))
        #target = (layer > 0).float()
        prediction = self.gan.discriminator.context[self.config.layer_name]

        input_normed = F.normalize(layer.unsqueeze(1), dim=2)
        embed_normed = F.normalize(prediction.unsqueeze(1), dim=2)
        #loss = F.cross_entropy(prediction.view(target.shape[0], -1), target.view(target.shape[0], -1))# * 0.05
        loss = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2).mean()
        self.add_metric('ce', loss)
        return loss, loss
