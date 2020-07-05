from hypergan.gan_component import ValidationException, GANComponent
from .unsupervised_image_folder import UnsupervisedImageFolder
import glob
import os
import torch
import torch.utils.data as data
import torchvision

class ImageLoader:
    """
    ImageLoader loads a set of images
    """

    def __init__(self, config):
        self.config = config
        self.multiple = torch.Tensor([2.0]).float()[0].cuda()
        self.offset = torch.Tensor([-1.0]).float()[0].cuda()
        self.datasets = []
        transform_list = []
        h, w = self.config.height, self.config.width
        if self.config.blank:
            return

        if config.crop:
            transform_list.append(torchvision.transforms.CenterCrop((h, w)))

        if config.resize:
            transform_list.append(torchvision.transforms.Resize((h, w)))

        if config.random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((h, w), pad_if_needed=True, padding_mode='edge'))

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)

        directories = self.config.directories or [self.config.directory]
        if(not isinstance(directories, list)):
            directories = [directories]

        self.dataloaders = []
        for directory in directories:
            mode = "RGB"
            if self.channels() == 4:
                mode = "RGBA"
            image_folder = UnsupervisedImageFolder(directory, transform=transform, mode=mode)
            self.dataloaders.append(data.DataLoader(image_folder, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=4, drop_last=True))
            self.datasets.append(iter(self.dataloaders[-1]))

    def batch_size(self):
        return self.config.batch_size

    def width(self):
        return self.config.width

    def height(self):
        return self.config.height

    def channels(self):
        return self.config.channels

    def next(self, index=0):
        if self.config.blank:
            self.sample = torch.zeros([self.config.batch_size, self.config.channels, self.config.height, self.config.width]).cuda()
            return self.sample
        try:
            self.sample = self.datasets[index].next()[0].cuda() * self.multiple + self.offset
            return self.sample
        except StopIteration:
            self.datasets[index] = iter(self.dataloaders[index])
            return self.next(index)
