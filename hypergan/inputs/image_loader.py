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

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)

        if(not isinstance(directories, list)):
            directories = [directories]

        for directory in directories:
            #TODO channels
            image_folder = torchvision.datasets.ImageFolder(directory, transform=transform)
            self.datasets.append(iter(data.DataLoader(image_folder, batch_size=self.batch_size, shuffle=True, num_workers=4)))
        self.next()

    def next(self):
        self.samples = [d.next()[0] for d in self.datasets]
        return self.samples
