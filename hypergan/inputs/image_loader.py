import glob
import os
import torchvision
import torch.utils.data as data
from hypergan.gan_component import ValidationException, GANComponent

class ImageLoader:
    """
    ImageLoader loads a set of images
    """

    def __init__(self, batch_size, directories, channels=3, width=64, height=64, crop=False, resize=False, sequential=False, random_crop=False):
        self.batch_size = batch_size

        directory = directories[0]
        return self.image_folder_create(directories, channels=channels, width=width, height=height, crop=crop, resize=resize, sequential=sequential, random_crop=random_crop)

    def image_folder_create(self, directories, channels=3, width=64, height=64, crop=False, resize=False, sequential=False, random_crop=False):
        self.datasets = []

        transform_list = []

        if crop:
            transform_list.append(torchvision.transforms.CenterCrop((height, width)))

        if resize:
            transform_list.append(torchvision.transforms.Resize((height, width)))
        if random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((height, width), pad_if_needed=True, padding_mode='edge'))

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)

        if(not isinstance(directories, list)):
            directories = [directories]

        self.dataloaders = []
        for directory in directories:
            #TODO channels
            image_folder = torchvision.datasets.ImageFolder(directory, transform=transform)
            self.dataloaders.append(data.DataLoader(image_folder, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True))
            self.datasets.append(iter(self.dataloaders[-1]))
        self.next()

    def next(self, index=0):
        try:
            self.sample = self.datasets[index].next()[0].cuda()
            return self.sample
        except StopIteration:
            self.datasets[index] = iter(self.dataloaders[index])
            return self.next(index)
