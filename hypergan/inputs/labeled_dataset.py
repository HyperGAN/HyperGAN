from PIL import Image
import os
import torch
import torchvision
from random import random, randint, shuffle
from pathlib import Path

from torch.utils import data

EXTS = ['jpg', 'jpeg', 'png']

class LabeledDataset(data.Dataset):
    # adapted from https://gist.githubusercontent.com/Netruk44/38d793e6d04a53cc4d9acbfadbb04a5c/raw/ba0b53b9618be58fab8dd0e61c655fc0986c23ea/imagen_train.py
    # Given a folder of images and a csv, returns a dataset of automatically generated (image, label) pairs.
    # Image filenames are used as the key lookup into the csv file.
    # The csv file should be formatted as 'filename,label,...'
    # Labels can contain anything except ',' and can be as long as you want. '\n' will get stripped out.
    def __init__(self, folder, csv_path, transform=None, device='cpu:0', mode='RGB'):
        super().__init__()
        self.folder = folder
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        
        with open(csv_path, 'rt') as fp:
            lines = [x.split(',') for x in fp.readlines()]
            self.texts = {x[0]: x[1:] for x in lines}
        print("Found csv entries", len(self.texts))

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
        if img_filename not in self.texts:
            return self.__getitem__((index+1) % len(self.texts))
        # Get the labels for this image
        all_labels = self.texts[img_filename]

        # Generate the text to be used for training by randomly shuffling the labels
        # and joining with a space
        #shuffle(all_labels)
        generated_full_description = ','.join(all_labels)
        generated_full_description = self.sanitize(generated_full_description)

        # Retrieve image
        img = self.transform(Image.open(img_path))

        return {
            'img': img,
            'txt': generated_full_description,
        }


