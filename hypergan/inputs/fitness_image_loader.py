from hypergan.gan_component import ValidationException, GANComponent
import glob
import os
import torch
import torch.utils.data as data
import torchvision

from hypergan.inputs.image_loader import ImageLoader
import numpy as np

class FitnessImageLoader(ImageLoader):
    """
    ImageLoader loads a set of images
    """

    def __init__(self, config):
        super(FitnessImageLoader, self).__init__(config=config)

    def next(self, index=0):
        if not hasattr(self, 'best_sample'):
            self.best_sample = super(FitnessImageLoader, self).next(index)
            self.best_sample = self.next(index)
        all_samples = torch.split(self.best_sample, 1, dim=0)
        prior_best_scores = self.gan.discriminator(self.best_sample)
        all_scores = torch.split(prior_best_scores, 1, dim=0)
        all_scores = [d.mean() for d in all_scores]
        for i in range(self.config.steps or 1):
            sample = super(FitnessImageLoader, self).next(index)
            d_scores = self.gan.discriminator(sample)
            d_scores = torch.split(d_scores, 1, dim=0)
            d_scores = [d.mean() for d in d_scores]
            samples = torch.split(sample, 1, dim=0)
            all_scores += d_scores
            all_samples += samples
        all_scores = [s.item() for s in all_scores]
        sorted_idx = np.argsort(all_scores)
        if self.config.reverse:
            sorted_idx = sorted_idx[::-1]
        sorted_idx = sorted_idx[:self.gan.batch_size()]
        sorted_samples = [all_samples[idx] for idx in sorted_idx]
        self.best_sample = torch.cat(sorted_samples, dim=0).cuda()
        self.sample = self.best_sample
        return self.best_sample



