import argparse
import uuid
import os
import sys
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.inputs import *
from hypergan.search.random_search import RandomSearch
from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.generators.base_generator import BaseGenerator
from hypergan.gans.base_gan import BaseGAN
from common import *

from torch import optim
from torch.autograd import Variable

from torchvision import datasets, transforms

arg_parser = ArgumentParser(description='Train an MNIST classifier G(x) = label')
args = arg_parser.parse_args()

class MNISTInputLoader:
    def __init__(self, batch_size):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataset_folder = 'mnist'

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_folder, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_folder, train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_cache = self.build_cache()
        self.test_dataset = iter(self.test_loader)
        self.eye = torch.eye(10).cuda()

    def batch_size(self):
        return args.batch_size

    def width(self):
        return 28

    def height(self):
        return 28

    def channels(self):
        return 1

    def build_cache(self):
        self.train_dataset = iter(self.train_loader)
        cache = []
        while True:
            try:
                batch = self.train_dataset.next()
                xs = torch.split(batch[0], 1, 0)
                ys = torch.split(batch[1], 1, 0)
                for (x,y) in zip(xs,ys):
                    cache.append([x,y])
            except StopIteration:
                cache = sorted(cache, key=lambda element: element[1].item())
                return cache

    def next(self, index=0):
        try:
            sample = [self.train_cache.pop(0) for i in range(args.batch_size)]
            xs = torch.cat([_s[0] for _s in sample], 0)
            ys = torch.cat([_s[1] for _s in sample], 0)
            self.sample = [xs.cuda(), self.eye[ys.cuda()].cuda()]
            return self.sample
        except IndexError:
            self.train_cache = self.build_cache()
            return self.next()

    def testdata(self, index=0):
        self.test_dataset = iter(self.test_loader)
        while True:
            try:
                self.sample = self.test_dataset.next()
                self.sample = [self.sample[0].cuda(), self.eye[self.sample[1].cuda()]]
                yield self.sample
            except StopIteration:
                return

class MNISTGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.generator = None
        self.loss = None
        self.trainer = None
        BaseGAN.__init__(self, *args, **kwargs)
        self.x, self.y = self.inputs.next()

    def create(self):
        self.generator = self.create_component("generator", input=self.inputs.next()[0])
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0], {"digit": inputs[1]})

    def forward_pass(self):
        self.x, self.y = self.inputs.next()
        g = self.generator(self.x)
        self.g = g
        d_real = self.forward_discriminator([self.x, self.y])
        d_fake = self.forward_discriminator([self.x, g])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def g_parameters(self):
        return self.generator.parameters()

    def d_parameters(self):
        return self.discriminator.parameters()

    def discriminator_fake_inputs(self):
        return [[self.x, self.g]]

    def discriminator_real_inputs(self):
        if hasattr(self, 'x'):
            return [self.x, self.y]
        else:
            return self.inputs.next()

class MNISTGenerator(BaseGenerator):
    def create(self):
        self.linear = torch.nn.Linear(28*28*1, 1024)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1024, 10)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, input, context={}):
        net = input
        net = self.linear(net.reshape(self.gan.batch_size(), -1))
        net = self.relu(net)
        net = self.linear2(net)
        net = self.sigmoid(net)
        return net

class MNISTDiscriminator(BaseDiscriminator):
    def create(self):
        self.linear = torch.nn.Linear(28*28*1+10, 1024)
        self.linear2 = torch.nn.Linear(1024, 1)
        self.tanh = torch.nn.Hardtanh()
        self.relu = torch.nn.ReLU()

    def forward(self, input, context={}):
        net = torch.cat([input.reshape(self.gan.batch_size(), -1),context['digit']], 1)
        net = self.linear(net)
        net = self.relu(net)
        net = self.linear2(net)
        net = self.tanh(net)
        return net


config = lookup_config(args)

if args.action == 'search':
    search = RandomSearch({
        'generator': {'class': MNISTGenerator, 'end_features': 10},
        'discriminator': {'class': MNISTDiscriminator}
        })

    config = search.random_config()

inputs = MNISTInputLoader(args.batch_size)

def setup_gan(config, inputs, args):
    gan = MNISTGAN(config, inputs=inputs)
    return gan

def train(config, args):
    gan = setup_gan(config, inputs, args)
    test_batches = []

    for i in range(args.steps):
        gan.step()

        if i % args.sample_every == 0 and i > 0:
            correct_prediction = 0
            total = 0
            for (x,y) in gan.inputs.testdata():
                prediction = gan.generator(x)
                correct_prediction += (torch.argmax(prediction,1) == torch.argmax(y,1)).sum()
                total += y.shape[0]
            accuracy = (float(correct_prediction) / total)*100
            print("accuracy: ", accuracy)

    return sum_metrics

def search(config, args):
    metrics = train(config, args)
    config_filename = "classification-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)

    with open(args.search_output, "a") as myfile:
        print("Writing result")
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
elif args.action == 'search':
    search(config, args)
else:
    print("Unknown action: "+args.action)


