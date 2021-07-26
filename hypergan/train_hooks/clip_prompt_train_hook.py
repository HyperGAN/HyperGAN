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
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from PIL import Image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
                 #K.RandomHorizontalFlip(p=0.5),				# NR: add augmentation options
                 #K.RandomVerticalFlip(p=0.5),
                 #K.RandomSolarize(0.01, 0.01, p=0.7),
                 #K.RandomSharpness(0.3,p=0.4),
                 #K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
                 #K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),

                K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
                K.RandomPerspective(0.7,p=0.7),
                #K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
                #K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
                )

        self.noise_fac = 0.1

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.upsample = nn.Upsample((self.cut_size,self.cut_size), mode='bilinear')

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)

            # Pooling
            #cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutout = self.upsample(input)# + self.max_pool(input))/2
            cutouts.append(cutout)


        raugs = torch.cat(cutouts, dim=0)
        #raugs = torch.cat(cutouts[1:], dim=0)
        raugs = self.augs(raugs)
        #raugs = rand_brightness(raugs)
        #raugs = rand_saturation(raugs)
        #raugs = rand_contrast(raugs)
        raugs = rand_translation(raugs)
        #raugs = rand_cutout(raugs)

        #batch = torch.cat([raugs, cutouts[0]], dim=0)
        batch = raugs

        #if self.noise_fac:
        #    facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
        #    batch = batch + facs * torch.randn_like(batch)
        return batch

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


class ClipPromptTrainHook(BaseTrainHook):
    perceptor = None
    def __init__(self, gan=None, config=None):
        super().__init__(gan, config)
        self.device = gan.device
        self.embed = None
        self.weight = torch.as_tensor(1)
        self.stop = torch.as_tensor(-np.inf)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
        if ClipPromptTrainHook.perceptor is None:
            clip_model = (self.config.clip_model or "ViT-B/32")
            jit = True if float(torch.__version__[:3]) < 1.8 else False
            ClipPromptTrainHook.perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(gan.device)
            #ClipPromptTrainHook.perceptor = clip.load(clip_model, jit=jit)[0].eval().to(self.device)
        cut_size = ClipPromptTrainHook.perceptor.visual.input_resolution
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        self.gan.cutouts = self.gan.inputs.next()

    def forward(self, d_loss, g_loss):
        prompt = (self.config.prompt or "cat")
        image_prompt = self.config.image_prompt

        if self.embed is None:
            if image_prompt:
                img = Image.open(image_prompt)
                pil_image = img.convert('RGB')
                img = resize_image(pil_image, (sideX, sideY))
                batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                self.embed = ClipPromptTrainHook.perceptor.encode_image(normalize(batch)).float()
            else:
                self.embed = ClipPromptTrainHook.perceptor.encode_text(clip.tokenize(prompt).to(self.device)).float()
        cutouts = self.make_cutouts(self.gan.g)
        self.gan.cutouts = cutouts
        input = ClipPromptTrainHook.perceptor.encode_image(self.normalize(cutouts)).float()
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        clip_loss = self.weight * dists.mean() * (self.config.gamma or 1.0)#replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        self.gan.add_metric('clip_loss', clip_loss)

        return [clip_loss, clip_loss]
