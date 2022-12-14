from PIL import Image
import os
import torch
import torchvision
import random
from torch.nn import functional as F
from pathlib import Path

from torch.utils import data

EXTS = ['jpg', 'jpeg', 'png']

class LLEDataset(data.Dataset):
    # adapted from https://gist.githubusercontent.com/Netruk44/38d793e6d04a53cc4d9acbfadbb04a5c/raw/ba0b53b9618be58fab8dd0e61c655fc0986c23ea/imagen_train.py
    # Given a folder of images and a csv, returns a dataset of automatically generated (image, label) pairs.
    # Image filenames are used as the key lookup into the csv file.
    # The csv file should be formatted as 'filename,label,...'
    # Labels can contain anything except ',' and can be as long as you want. '\n' will get stripped out.
    def __init__(self, folder, csv_path, labelset = {'face': 0, 'multiple people': 1, 'watermark': 2, 'cropped': 3}, transform=None, device='cpu:0', mode='RGB'):
        super().__init__()
        self.folder = folder

        self.labels = {}
        self.labelset = labelset
        self.paths = []
        with open(csv_path, 'rt') as fp:
            lines = [x.split(',') for x in fp.readlines()]
            for split in lines:
                labels = split[1:]
                labels[-1] = labels[-1].strip()
                fname = split[0]

                self.labels[fname]= labels
                self.paths.append(fname)
        print("Found csv entries", len(self.labels))
        print("Found labels", len(self.labelset))

        print("Found paths entries", len(self.paths))
        #for i,path in enumerate(self.paths):
        #    p = os.path.relpath(path, self.folder)
        #    if p not in self.texts:
        #        del self.paths[i]

        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.paths)

    def sanitize(self, t):
        replacements = {
            '\n': ' ',
        }
        
        for k, v in replacements.items():
            t = t.replace(k, v)
            
        return t
    
    # Returns a dict with 'txt' and 'img' keys.
    def __getitem__(self, index):
        img_path = self.paths[index]
        img_filename = os.path.relpath(img_path, self.folder)
        #assert img_filename in self.texts, f'Image {img_filename} not found in {img_path}'
        if img_filename not in self.labels:
            print(self.labels.keys())
            assert 1==0, "Not found" + img_filename
        # Get the labels for this image
        all_labels = self.labels[img_filename]

        # Retrieve image
        img = self.transform(Image.open(img_path))
        labels = [self.labelset[x] for x in all_labels]
        onehots = F.one_hot(torch.arange(0, len(self.labelset)), len(self.labelset))
        multilabel = sum([onehots[i] for i in labels])
        target_class = labels[0]

        return {
            'img': img,
            'labels': multilabel,
            'target_class': target_class
        }


