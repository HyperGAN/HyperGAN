from hypergan.gan_component import ValidationException, GANComponent
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
        self.datasets = []
        transform_list = []
        h, w = self.config.height, self.config.width
        if self.config.blank:
            self.next()
            return

        if config.crop:
            transform_list.append(torchvision.transforms.CenterCrop((h, w)))

        if config.resize:
            transform_list.append(torchvision.transforms.Resize((h, w)))

        if config.random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((h, w), pad_if_needed=True, padding_mode='edge'))

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)

        directories = self.config.directories
        if(not isinstance(directories, list)):
            directories = [directories]

        self.dataloaders = []
        for directory in directories:
            #TODO channels
            image_folder = torchvision.datasets.ImageFolder(directory, transform=transform)
            self.dataloaders.append(data.DataLoader(image_folder, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=4, drop_last=True))
            self.datasets.append(iter(self.dataloaders[-1]))
        self.next()

    def next(self, index=0):
        if self.config.blank:
            self.sample = torch.zeros([self.config.batch_size, self.config.channels, self.config.height, self.config.width])
            return self.sample
        try:
            self.sample = self.datasets[index].next()[0].cuda() * 2.0 - 1.0
            return self.sample
        except StopIteration:
            self.datasets[index] = iter(self.dataloaders[index])
            return self.next(index)
