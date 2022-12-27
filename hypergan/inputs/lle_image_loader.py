from hypergan.gan_component import ValidationException, GANComponent
from .crop_resize_transform import CropResizeTransform
from .labeled_dataset import LabeledDataset
from .lle_dataset import LLEDataset
import glob
import os
import torch
import torch.utils.data as data
import torchvision
import PIL
from random import random, randint, shuffle

class LLEImageLoader:

    def __init__(self, config, device=None):
        self.config = config
        self.datasets = []
        transform_list = []
        h, w = self.config.height, self.config.width
        if self.config.blank:
            return
        if device is None:
            device = self.config.device
        self.multiple = None
        self.offset = None

        if config.crop:
            transform_list.append(CropResizeTransform((h, w)))

        if config.resize:
            transform_list.append(torchvision.transforms.Resize((h, w)))

        if config.random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((h, w), pad_if_needed=True, padding_mode='edge'))

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)
        mode = "RGB"
        if self.channels() == 4:
            mode = "RGBA"
        #image_folder = LabeledDataset(config.directory, config.csv_path, transform=transform, mode=mode, device=device)
        image_folder = LLEDataset(config.directory, config.csv_path, transform=transform, mode=mode, device=device)
        num_workers = 2
        if self.config.num_workers is not None:
            num_workers = self.config.num_workers
        dataloader = data.DataLoader(image_folder, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True)
        self.dataloader = dataloader
        self.dataset = None
        self.device = device

    def to(self, device):
        return LabeledImageLoader(self.config, device=device)

    def batch_size(self):
        return self.config.batch_size

    def width(self):
        return self.config.width

    def height(self):
        return self.config.height

    def channels(self):
        return self.config.channels

    def next(self, index=0):
        if self.dataset is None:
            self.dataset = iter(self.dataloader)
        if self.config.blank:
            return torch.zeros([self.config.batch_size, self.config.channels, self.config.height, self.config.width]).cuda()
        try:
            if self.multiple is None:
                self.multiple = torch.tensor(2.0, device=self.device)
                self.offset = torch.tensor(-1.0, device=self.device)
            data = next(self.dataset)
            data['img'] = data['img'].to(self.device) * self.multiple + self.offset
            self.data = data
            return data['img']
        except OSError:
            return self.next(index)
        except ValueError:
            return self.next(index)
        except RuntimeError:
            return self.next(index)
        except PIL.Image.DecompressionBombError:
            return self.next(index)
        except PIL.UnidentifiedImageError:
            return self.next(index)
        except StopIteration:
            self.dataset = iter(self.dataloader)
            return self.next(index)