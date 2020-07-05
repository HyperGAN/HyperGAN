import numpy as np
import hyperchamber as hc
import torch.nn as nn

from .base_generator import BaseGenerator

class DCGANGenerator(BaseGenerator):

    def required(self):
        return []

    def create(self):
        self.linear = nn.Sequential(
                nn.Linear(100, 4*4*512),
                nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        lin = self.linear(x).view(self.gan.batch_size(), 512, 4, 4)
        net = self.net(lin)
        return net.view(self.gan.batch_size(),3,64,64)
