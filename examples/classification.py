import argparse
import uuid
import os
import sys
import hypergan as hg
import hyperchamber as hc
from hypergan.inputs import *
from hypergan.search.random_search import RandomSearch
from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.generators.base_generator import BaseGenerator
from hypergan.gans.base_gan import BaseGAN
from hypergan.layer_shape import LayerShape
from common import *

from torch import optim
from torch.autograd import Variable

from torchvision import datasets, transforms

arg_parser = ArgumentParser(description='Train an classifier G(x) = label')
arg_parser.parser.add_argument('--dataset', '-D', type=str, default='mnist', help='dataset to use - options are mnist / cifar10')
args = arg_parser.parse_args()
config_name = args.config
save_file = "saves/"+config_name+"/model.ckpt"
os.makedirs(os.path.expanduser(os.path.dirname(save_file)), exist_ok=True)

class InputLoader:
    def __init__(self, batch_size):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        dataset_folder = args.dataset
        dataset = datasets.MNIST
        if args.dataset == 'cifar10':
            dataset = datasets.CIFAR10

        train_loader = torch.utils.data.DataLoader(
            dataset(dataset_folder, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            dataset(dataset_folder, train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = iter(self.train_loader)
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

    def next(self, index=0):
        try:
            self.sample = self.train_dataset.next()
            self.sample = [self.sample[0].cuda(), self.eye[self.sample[1].cuda()]]
            return self.sample
        except StopIteration:
            self.train_dataset = iter(self.train_loader)
            return self.next(index)

    def testdata(self, index=0):
        self.test_dataset = iter(self.test_loader)
        while True:
            try:
                self.sample = self.test_dataset.next()
                self.sample = [self.sample[0].cuda(), self.eye[self.sample[1].cuda()]]
                yield self.sample
            except StopIteration:
                return

class GAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.generator = None
        self.loss = None
        BaseGAN.__init__(self, *args, **kwargs)
        self.x, self.y = self.inputs.next()

    def create(self):
        self.generator = self.create_component("generator", input=self.inputs.next()[0])
        if self.config.generator2:
            self.generator2 = self.create_component("generator2", input=self.inputs.next()[1])
        self.discriminator = self.create_component("discriminator", context_shapes={"digit": LayerShape(10)})
        self.loss = self.create_component("loss")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0], {"digit": inputs[1]})

    def forward_pass(self):
        self.x, self.y = self.inputs.next()
        g = self.generator(self.x)
        if self.config.generator2:
            g2 = self.generator2(self.y)
            gy = self.generator(g2)
            self.gy = gy
            self.g2 = g2
        self.g = g
        d_real = self.forward_discriminator([self.x, self.y])
        d_fake = self.forward_discriminator([self.x, g])
        correct = torch.floor((torch.round(g) == self.y).long().sum(axis=1)/10.0).view(-1,1)
        if self.config.generator2:
            d_fake += correct * self.forward_discriminator([g2, gy])
        self.d_fake = d_fake
        self.d_real = d_real
        self.adversarial_norm_fake_targets = [
            [self.x, self.g]
        ]
        if self.config.generator2:
           self.adversarial_norm_fake_targets += [
             [g2, self.gy]
           ]
        return d_real, d_fake

    def discriminator_fake_inputs(self):
        return self.adversarial_norm_fake_targets

    def discriminator_real_inputs(self):
        if hasattr(self, 'x'):
            return [self.x, self.y]
        else:
            return self.inputs.next()

    def generator_components(self):
        if self.config.generator2:
            return [self.generator2, self.generator]
        return [self.generator]
    def discriminator_components(self):
        return [self.discriminator]

class Generator(BaseGenerator):
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

class Discriminator(BaseDiscriminator):
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
        'generator': {'class': Generator, 'end_features': 10},
        'discriminator': {'class': Discriminator}
        })

    config = search.random_config()

inputs = InputLoader(args.batch_size)

def setup_gan(config, inputs, args):
    gan = GAN(config, inputs=inputs)
    return gan

def train(config, args):
    gan = setup_gan(config, inputs, args)
    trainable_gan = hg.TrainableGAN(gan, save_file = save_file, devices = args.devices, backend_name = args.backend)
    test_batches = []
    accuracy = 0

    for i in range(args.steps):
        trainable_gan.step()

        if i == args.steps-1 or i % args.sample_every == 0 and i > 0:
            correct_prediction = 0
            total = 0
            for (x,y) in gan.inputs.testdata():
                prediction = gan.generator(x)
                correct_prediction += (torch.argmax(prediction,1) == torch.argmax(y,1)).sum()
                total += y.shape[0]
            accuracy = (float(correct_prediction) / total)*100
            print("accuracy: ", accuracy)

    return accuracy

def search(config, args):
    metrics = train(config, args)
    config_filename = "classification-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)

    with open(args.search_output, "a") as myfile:
        print("Writing result")
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, args)
    print(config_name + ": resulting metrics:", metrics)
elif args.action == 'search':
    search(config, args)
else:
    print("Unknown action: "+args.action)


