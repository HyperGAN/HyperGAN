import torch
from torchvision import transforms
import hyperchamber as hc
import numpy as np
import inspect
from CLIP import clip
import kornia.augmentation as K
from torch.nn import functional as F
from operator import itemgetter
from hypergan.train_hooks.differential_augmentation_train_hook import rand_contrast, rand_saturation, rand_brightness, rand_translation, rand_cutout
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from PIL import Image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class PerceptualLossTrainHook(BaseTrainHook):
    """
    https://papers.nips.cc/paper/2016/file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf
    """
    def __init__(self, gan=None, config=None):
        super().__init__(gan, config)
        clip_model = (self.config.clip_model or "ViT-B/32")
        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self.perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(gan.device)
        cut_size = self.perceptor.visual.input_resolution
        self.make_cutouts = MakeCutouts(cut_size, 1, cut_pow=1)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

    def encode_image(self, image):
        return self.perceptor.encode_image(self.normalize(self.make_cutouts(image))).float()


    def forward(self, d_loss, g_loss):
        enc_g = self.encode_image(self.gan.g)
        enc_x = self.encode_image(self.gan.x)
        perceptual_loss = ((enc_g - enc_x)**2).mean()
        self.gan.add_metric('perceptual_loss', perceptual_loss)
        return [perceptual_loss, perceptual_loss]
