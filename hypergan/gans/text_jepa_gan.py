from .base_gan import BaseGAN
import random
import kornia.augmentation as K
from hypergan.train_hooks.diffusion_train_hook import GaussianDiffusion
from hypergan.encoders.t5_text_encoder import T5TextEncoder
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
        self.custom_encoder = self.create_component("custom_encoder")
        self.project_image = self.create_component("project_image")
        self.project_text = self.create_component("project_text")
        self.custom_text_encoder = self.create_component("custom_text_encoder")
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
            loss = F.mse_loss(prediction, target) * ( self.config.mse_lambda or 1)
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
            ing = self.generator(self.s1.clone().detach())
            inx = self.x

            d_real = self.discriminator2(inx)
            target = (self.encode_image(ing, encoder="clip")> 0).float()
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
            target = (self.encode_image(self.x, encoder="clip")> 0).float()
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('cex', loss)
            self.add_metric('min', target.min().mean())
            self.add_metric('mean', target.mean())
            self.add_metric('max', target.max().mean())
            return loss, None
        def kloss2():
            prediction = self.predicted_text
            target = (self.encode_text(self.text, encoder="clip").clone().detach() > 0.0 ).float()
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
        
        def off_diagonal(a):
            n = a.shape[1]
            return a.masked_select(~torch.eye(n, dtype=bool, device=a.device))

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


        def augment(x):
            return self.augs(x)

        def vicreg_loss_image():
            x_a = augment(self.x)
            x_b = augment(self.x)
            z_a = self.encode_image(x_a)
            z_b = self.encode_image(x_b)
            z_a = self.project_image(z_a)
            z_b = self.project_image(z_b)
            D = z_a.shape[1]
            N = z_a.shape[0]
            sim_loss = F.mse_loss(z_a, z_b)
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(torch.relu(1-std_z_a)+torch.mean(torch.relu(1-std_z_b)))
            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + \
                off_diagonal(cov_z_b).pow_(2).sum() / D

            mu = 2.5
            lam = 2.5
            nu = 1
            loss = lam * sim_loss + mu * std_loss + nu * cov_loss
            self.add_metric('vicr', loss.mean())
            return loss ,loss

        def l2_pred_loss():
            loss = F.mse_loss(self.s1, self.s0.clone().detach()) * 250
            self.add_metric('pred', loss.mean())
            return loss, loss

        def random_swap(text, offset, length):
            rot = random.randint(offset,length)
            rot2 = rot
            while rot2 == rot:
                rot2 = random.randint(offset,length)
            newtext = text[:rot] + text[rot2:rot2+1] + text[rot+1:]
            return newtext


        def vicreg_loss_text():
            newtext_a = []
            newtext_b = []
            for text in self.text:
                max_len = len(text)
                max_len = max(max_len, 64)
                newtext_a.append(random_swap(text, 0, max_len//2))
                newtext_b.append(random_swap(text, max_len//2, max_len))
            z_a = self.encode_text(newtext_a)
            z_b = self.encode_text(newtext_b)
            z_a = self.project_text(z_a)
            z_b = self.project_text(z_b)
            D = z_a.shape[1]
            N = z_a.shape[0]
            sim_loss = F.mse_loss(z_a, z_b)
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(torch.relu(1-std_z_a)+torch.mean(torch.relu(1-std_z_b)))
            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + \
                off_diagonal(cov_z_b).pow_(2).sum() / D

            mu = 2.5
            lam = 2.5
            nu = 1
            loss = lam * sim_loss + mu * std_loss + nu * cov_loss
            self.add_metric('vict', loss.mean())
            return loss ,loss




        self.diffusion = GaussianDiffusion(beta_schedule="linear", timesteps=100).to(self.device)
        if self.config.train_decoder:
            self.add_loss(kloss)
            #self.add_loss(kloss2)
            self.add_loss(vicreg_loss_image)
            self.add_loss(vicreg_loss_text)
            self.add_loss(l2_pred_loss)
            self.add_loss(l2_loss)
        else:
            #self.add_loss(mirror_loss)
            self.add_loss(vicreg_loss_image)
            self.add_loss(vicreg_loss_text)
            #self.add_loss(l2_pred_loss)
            self.add_loss(l2_loss)
            if self.config.gan_loss:
                self.add_loss(gan_loss)
            #self.add_loss(pred_loss)
            #self.add_loss(cos_loss)
            #self.add_loss(cos_loss2)

            #self.add_loss(same_x)
        if self.custom_text_encoder:
            self.text_encoder = T5TextEncoder(max_text_len=(self.config.tokens or 64))


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

    def encode_image(self, img, encoder="auto"):
        if encoder == "clip" or self.custom_encoder is None:
            return self.perceptor.encode_image(self.make_cutouts(img)).float().squeeze()
        return self.custom_encoder(img)

        #return PromptGAN.perceptor.encode_image(self.make_cutouts(img)).float()
        #return PromptGAN.perceptor.encode_image(self.normalize(self.make_cutouts(img.add(1).div(2))).float()

    def encode_text(self, text, noise = False, encoder="auto"):
        if encoder == "clip" or self.custom_encoder is None:
            return self.perceptor.encode_text(open_clip.tokenize(text).to(self.device)).float().clone().detach()
        else:
            if self.config.clip_text_encoder:
                return self.custom_text_encoder(self.perceptor.encode_text(open_clip.tokenize(text).to(self.device)).float().clone().detach().unsqueeze(1))

            text_embeddings = self.text_encoder.encode_text(text, tokens=(self.config.tokens or 64)).to(self.device)
            if noise:
                text_embeddings = text_embeddings +torch.randn_like(text_embeddings)*noise
            return self.custom_text_encoder(text_embeddings)


    def partial_noise(self, x, threshold=0.125):
        #if self.config.noise_type == "masked":
        mask = (torch.rand_like(x) > threshold).float()
        return mask*x
        #noise_level = 1-((threshold+1) / 2.0)
        #t = self.diffusion.get_times(self.batch_size(), noise_level)
        #chance = 0.5
        #mask = (torch.rand([x.shape[0], 1, 1, 1], device=x.device) > chance).float()

        #return self.diffusion.q_sample(x, t) * mask + x * (1 - mask)



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

        if self.config.denoising:
            self.s0 = self.encode_image(self.partial_noise(self.x))
        else:
            self.s0 = self.encode_image(self.x)

        if self.config.train_decoder:
            self.autoencoding = self.generator(self.s0.clone().detach())
            self.s1 =  self.pred(torch.cat([self.encode_text(self.text).unsqueeze(1), self.augmented_latent.unsqueeze(1)], 1))
            g = self.generator(self.s1.clone().detach())
            self.g  = g
            self.x_args = [self.train_hooks.augment_x(self.x),self.train_hooks.augment_x(self.x), self.train_hooks.augment_x(self.autoencoding.clone().detach())]
            self.g_args = [self.train_hooks.augment_x(self.x),self.train_hooks.augment_x(self.autoencoding), self.train_hooks.augment_g(self.g)]
            self.g_args = [torch.cat(self.g_args, dim=1)]
            self.x_args = [torch.cat(self.x_args, dim=1)]
            #self.g_args = [self.g]
            #self.x_args = [self.x]
            d_real = self.forward_discriminator(*self.x_args)
            self.predicted_x = self.discriminator.context["x_prediction"]
            d_fake = self.forward_discriminator(*self.g_args)
            #self.predicted_text = self.discriminator.context["text_prediction"]
        else:
            self.encoded_text = self.encode_text(self.text)
            g = self.generator(self.s0)
            #gin = torch.cat([self.encoded_text.unsqueeze(1),self.s0.unsqueeze(1)], dim=1)
            #g = self.generator(gin.clone().detach())
            self.autoencoding = g
            self.s1 = self.pred(torch.cat([self.encoded_text.unsqueeze(1), self.augmented_latent.unsqueeze(1)], 1))
            #self.g = self.generator(self.s1)
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
        return [self.generator, self.pred, self.pred2, self.custom_encoder, self.custom_text_encoder, self.project_image, self.project_text]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args

