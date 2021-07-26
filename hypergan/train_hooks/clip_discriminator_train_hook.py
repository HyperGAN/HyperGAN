import torch
from torchvision import transforms
from hypergan.losses.stable_gan_loss import StableGANLoss
import hyperchamber as hc
import numpy as np
import inspect
from CLIP import clip
import kornia.augmentation as K
from torch.nn import functional as F
from operator import itemgetter
from hypergan.train_hooks.differential_augmentation_train_hook import rand_contrast, rand_saturation, rand_brightness, rand_translation, rand_cutout
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.upsample = nn.Upsample((self.cut_size,self.cut_size), mode='bilinear')

    def forward(self, input):
        return self.upsample(input)# + self.max_pool(input))/2


class ClipDiscriminatorTrainHook(BaseTrainHook):
    perceptor = None
    def __init__(self, gan=None, config=None):
        super().__init__(gan, config)
        self.device = gan.device
        self.embed = None
        self.weight = torch.as_tensor(1)
        self.stop = torch.as_tensor(-np.inf)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
        if ClipDiscriminatorTrainHook.perceptor is None:
            clip_model = (self.config.clip_model or "ViT-B/32")
            jit = True if float(torch.__version__[:3]) < 1.8 else False
            ClipDiscriminatorTrainHook.perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(gan.device)
            #ClipDiscriminatorTrainHook.perceptor = clip.load(clip_model, jit=jit)[0].eval().to(self.device)
        cut_size = ClipDiscriminatorTrainHook.perceptor.visual.input_resolution
        self.make_cutouts = MakeCutouts(cut_size, 1, cut_pow=1)

        self.clip_discriminator = gan.create_component("clip_discriminator", defn=self.config.clip_discriminator, input=torch.zeros_like(gan.latent.next()))
        self.clip_stable_loss = StableGANLoss()

    def forward(self, d_loss, g_loss):
        prompt = (self.config.prompt or "cat")

        if self.embed is None:
            self.embed = ClipDiscriminatorTrainHook.perceptor.encode_text(clip.tokenize(prompt).to(self.device)).float()
        input = ClipDiscriminatorTrainHook.perceptor.encode_image(self.normalize(self.make_cutouts(self.gan.g))).float()
        emb = self.embed.expand(input.shape[0], *self.embed.shape[1:])
        inx = [torch.cat([emb, emb], dim=1)]
        ing = [torch.cat([emb, input], dim=1)]
        clip_loss = self.clip_stable_loss.stable_loss(self.clip_discriminator, inx, ing)

        self.gan.add_metric('clip_loss_d', clip_loss[0])
        self.gan.add_metric('clip_loss_g', clip_loss[1])

        return clip_loss

    def generator_components(self):
        return []

    def discriminator_components(self):
        return [self.clip_discriminator]
