from .base_gan import BaseGAN
from hypergan.losses.stable_gan_loss import StableGANLoss
from torchvision import transforms
from torch.nn import functional as F
import open_clip
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from torch import nn
from hypergan.trainers import *
from hypergan.layer_shape import LayerShape
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
import copy
import hyperchamber as hc
import hypergan as hg
import importlib
import json
import numpy as np
import os
import sys
import time
import torch
import uuid

class TextJepaGAN(BaseGAN):
    perceptor = None
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        cut_size = 224#self.perceptor.visual.image_size
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        self.x = self.inputs.next()

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_emb = nn.CosineEmbeddingLoss(margin=0.01, reduction="none")
    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        self.pred = self.create_component("pred")
        self.pred2 = self.create_component("pred2")
        encoded_shape = LayerShape(1024)
        self.discriminator = self.create_component("discriminator", context_shapes={"encoded": encoded_shape})
        self.discriminator2 = self.create_component("discriminator2", context_shapes={"encoded": encoded_shape})

        self.stable_gan_loss = StableGANLoss(gan=self)
        def mirror_loss():
            target = self.actual_mirror
            prediction = self.predicted_mirror
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('ce', loss)
            return loss, loss

        def l2_loss():
            target = self.x
            prediction = self.autoencoding
            #target = self.s0
            #prediction = self.encode_image(self.autoencoding)
            loss = F.mse_loss(prediction, target) 
            self.add_metric('mse', loss)
            return loss, loss

        def l22_loss():
            target = self.autoencoding
            prediction = self.g
            loss = F.mse_loss(prediction, target) * 0.1 
            self.add_metric('mse', loss)
            return loss, loss

        def same_x():
            image = self.encode_image(self.x)
            text = self.encoded_text
            #text2 = self.from_encoded_text
            #loss = self.arcsin(image, text).mean() * (self.config.prompt_lambda or 1.0)
            lossa = self.arcsin(image, text).clone()
            lossb = self.arcsin(self.s1, text)
            loss = lossa - lossb
            #y = self.cos(yimage, text).mean()
            #loss = self.cos_emb(image, text, torch.ones(text.shape[0], device=text.device)).unsqueeze(1)
            #loss = loss * self.config.prompt_lambda
            #loss = self.cos(image, text).mean()
            #loss = F.mse_loss(loss, self.arcsin(image, text2).mean() * (self.config.prompt_lambda or 1.0))
            self.add_metric('ab', loss.mean())
            return loss, loss


        def cos_loss():
            image = self.encode_image(self.autoencoding)
            text = self.encoded_text0
            #text2 = self.from_encoded_text
            #loss = self.arcsin(image, text).mean() * (self.config.prompt_lambda or 1.0)
            loss = self.arcsin(image, text).mean()
            #y = self.cos(image, text).mean()
            #loss = self.cos_emb(image, text, torch.ones(text.shape[0], device=text.device)).unsqueeze(1)
            #loss = loss * self.config.prompt_lambda
            #loss = self.cos(image, text).mean()
            #loss = F.mse_loss(loss, self.arcsin(image, text2).mean() * (self.config.prompt_lambda or 1.0))
            self.add_metric('cos', loss.mean())
            return loss ,loss
        def cos_loss2():
            image = self.s1
            text = self.encoded_text
            #text2 = self.from_encoded_text
            #loss = self.arcsin(image, text).mean() * (self.config.prompt_lambda or 1.0)
            loss = self.arcsin(image, text).mean()
            #y = self.cos(image, text).mean()
            #loss = self.cos_emb(image, text, torch.ones(text.shape[0], device=text.device)).unsqueeze(1)
            #loss = loss * self.config.prompt_lambda
            #loss = self.cos(image, text).mean()
            #loss = F.mse_loss(loss, self.arcsin(image, text2).mean() * (self.config.prompt_lambda or 1.0))
            self.add_metric('cos2', loss.mean())
            return loss ,loss


        def gan_loss():
            #ing = self.encode_image(self.autoencoding)
            #inx = self.encode_image(self.x)
            #ing = torch.cat([self.encode_image(self.autoencoding).unsqueeze(1), self.encode_image(self.x).unsqueeze(1)], dim=1)
            #inx = torch.cat([self.encode_image(self.x).unsqueeze(1), self.encode_image(self.x).unsqueeze(1)], dim=1)
            #ex = self.s0.unsqueeze(1).clone().detach()
            #ing = torch.cat([ex, self.encode_image(self.autoencoding).unsqueeze(1)], dim=1)
            #inx = torch.cat([ex, ex], dim=1)
            #ing = torch.cat([self.x, self.autoencoding], dim=1)
            #inx = torch.cat([self.x, self.x], dim=1)
            ing = self.autoencoding
            inx = self.x

            d_real = self.discriminator2(inx)
            target = (self.s0> 0).float()
            #targetz = (self.augmented_latent > 0).float()
            d_fake = self.discriminator2(ing)
            predicted_text = self.discriminator2.context["text_prediction"]
            #predicted_z = self.discriminator2.context["z_prediction"]
            rloss = F.cross_entropy(predicted_text, target)# * 0.05
            #zloss = F.cross_entropy(predicted_z, targetz)# * 0.05
            #loss = self.stable_gan_loss.stable_loss(self.discriminator2, [inx], [ing], d_real=d_real, d_fake=d_fake)
            d_loss = F.softplus(-d_real)+F.softplus(d_fake)
            g_loss = F.softplus(-d_fake)
            #d_loss = loss[0]
            #g_loss = loss[1]
            #loss = [d_loss + rloss, g_loss]
            loss = [d_loss, g_loss]


            self.add_metric('dl', loss[0].mean())
            self.add_metric('gl', loss[1].mean())
            return loss[0], loss[1]
        def kloss():
            prediction = self.predicted_x
            target = (self.encode_image(self.x) > 0.0 ).float()
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('cex', loss)
            self.add_metric('min', target.min().mean())
            self.add_metric('mean', target.mean())
            self.add_metric('max', target.max().mean())
            return loss, None
        def pred_loss():
            s0 = self.autoencoding
            s1 = self.pred2(s0)
            st = self.x
            ing = torch.cat([st, s0, s1], dim=1)
            inx = torch.cat([st, st, st], dim=1)

            d_fake = self.discriminator2(ing)
            target = (self.s0 > 0).float()
            predicted_text = self.discriminator2.context["text_prediction"]
            d_real = self.discriminator2(inx)
            loss = self.stable_gan_loss.stable_loss(self.discriminator2, [inx], [ing], d_real=d_real, d_fake=d_fake)
            rloss = F.cross_entropy(predicted_text, target)# * 0.05

            self.add_metric('dl', loss[0].mean())
            self.add_metric('gl', loss[1].mean())
            return loss[0] + rloss, loss[1]




        if self.config.train_decoder:
            self.add_loss(kloss)
        else:
            #self.add_loss(mirror_loss)
            self.add_loss(l2_loss)
            #self.add_loss(gan_loss)
            #self.add_loss(pred_loss)
            #self.add_loss(cos_loss)
            #self.add_loss(cos_loss2)

            #self.add_loss(same_x)


    def forward_discriminator(self, *inputs):
        if len(inputs)>1:
            return self.discriminator(inputs[0], context= {"encoded":inputs[1]})
        else:
            return self.discriminator(inputs[0])



    def next_inputs(self):
        inp = self.inputs.next()
        self.x = inp['img']
        self.text = inp['txt']
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)


    def arcsin(self, image, text):
        #return image @ text.T
        #image = image.norm(dim=-1, keepdim=True)
        #text = text.norm(dim=-1, keepdim=True)
        image = F.normalize(image.squeeze(), dim=1)
        text = F.normalize(text.squeeze(), dim=1)
        #y = self.cos(image, text)
        #return image @ text.T
        #return image.sub(text)
        #return self.cos_emb(image, text, y).unsqueeze(1)
        #self.add_metric('y', y.clone().detach().mean())
        #return y
        #return image.sub(text).norm(dim=1).div(2).arcsin().unsqueeze(1)
        return image.sub(text).norm(dim=1).div(2).arcsin().unsqueeze(1)

    def encode_image(self, img):
        return self.perceptor.encode_image(self.make_cutouts(img)).float()
        #return PromptGAN.perceptor.encode_image(self.make_cutouts(img)).float()
        #return PromptGAN.perceptor.encode_image(self.normalize(self.make_cutouts(img.add(1).div(2))).float()

    def forward_pass(self):
        prompt = self.config.prompt
        if self.perceptor is None:
            model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')

            #clip_model = (self.config.clip_model or "ViT-B/32")
            #jit = True if float(torch.__version__[:3]) < 1.8 else False
            self.perceptor = model.eval().requires_grad_(False).to(self.device)#clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(self.device)
            #jit = True if float(torch.__version__[:3]) < 1.8 else False
            #self.perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(self.device)
            #self.perceptor = clip.load(clip_model, jit=True)[0].requires_grad_(False).eval().to(self.device)

        self.s0 = self.encode_image(self.x)
        if self.config.train_decoder:
            g = self.generator(self.s0)
            self.g  = g
            self.x_args = [self.train_hooks.augment_x(self.x), self.train_hooks.augment_x(self.x)]
            self.g_args = [self.train_hooks.augment_x(self.x), self.train_hooks.augment_g(self.g)]
            self.g_args = [torch.cat(self.g_args, dim=1)]
            self.x_args = [torch.cat(self.x_args, dim=1)]
            d_real = self.forward_discriminator(*self.x_args)
            self.predicted_x = self.discriminator.context["x_prediction"]
            d_fake = self.forward_discriminator(*self.g_args)
        else:
            self.encoded_text0 = self.perceptor.encode_text(open_clip.tokenize(self.text).to(self.device)).float().clone().detach()
            g = self.generator(self.s0)
            self.encoded_text = self.perceptor.encode_text(open_clip.tokenize(self.text).to(self.device)).float().clone().detach()
            self.s1 = self.pred(torch.cat([self.encoded_text.unsqueeze(1), self.augmented_latent.unsqueeze(1)], 1))
            self.g = self.generator(self.s1)
            self.autoencoding = g
            self.st = self.encode_image(self.x)
            #self.actual_mirror = (self.augmented_latent > 0).float()
            #self.augmented_g = self.train_hooks.augment_g(self.g)
            self.g_args = [ self.encoded_text.unsqueeze(1),  self.s1]
            self.x_args = [ self.encoded_text.unsqueeze(1), self.st.unsqueeze(1)]
            self.g_args = [torch.cat(self.g_args, dim=1)]
            self.x_args = [torch.cat(self.x_args, dim=1)]
            d_real = self.forward_discriminator(*self.x_args)
            d_fake = self.forward_discriminator(*self.g_args)
        #self.autoencoding= self.discriminator.context["autoencoding"]
        #self.predicted_mirror = self.discriminator.context["z_prediction"]

        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def input_nodes(self):
        "used in hypergan build"
        return [
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
        ]

    def discriminator_components(self):
        return [self.discriminator, self.discriminator2]

    def generator_components(self):
        return [self.generator, self.pred, self.pred2]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args

