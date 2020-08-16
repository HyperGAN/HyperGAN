from hypergan.gan_component import ValidationException, GANComponent
from .crop_resize_transform import CropResizeTransform
from .unsupervised_image_folder import UnsupervisedImageFolder
import glob
import os
import torch
import torch.utils.data as data
import torchvision
import PIL

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
            return

        if config.crop:
            transform_list.append(CropResizeTransform((h, w)))

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
        self.device=0
        for directory in directories:
            mode = "RGB"
            if self.channels() == 4:
                mode = "RGBA"
            image_folder = UnsupervisedImageFolder(directory, transform=transform, mode=mode)
            shuffle = True
            if config.shuffle is not None:
                shuffle = config.shuffle
            dataloader = data.DataLoader(image_folder, batch_size=config.batch_size, shuffle=shuffle, num_workers=4, drop_last=True, pin_memory=True)
            self.dataloaders.append(dataloader)
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
            return torch.zeros([self.config.batch_size, self.config.channels, self.config.height, self.config.width]).cuda()
        try:
            self.multiple = torch.tensor(2.0, device="cuda:"+str(self.device))
            self.offset = torch.tensor(-1.0, device="cuda:"+str(self.device))
            return self.datasets[index].next()[0].cuda(device=self.device) * self.multiple + self.offset
        except ValueError:
            return self.next(index)
        except PIL.UnidentifiedImageError:
            return self.next(index)
        except StopIteration:
            self.datasets[index] = iter(self.dataloaders[index])
            return self.next(index)
