from hyperchamber import Config
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.batch_walk_sampler import BatchWalkSampler
from hypergan.samplers.input_sampler import InputSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.y_sampler import YSampler
from pathlib import Path
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import hyperchamber as hc
import hypergan as hg
import inspect
import math
import numpy as np
import os
import re
import torch
import torch.nn as nn

class BaseGAN():
    def __init__(self, config=None, inputs=None):
        """ Initialized a new GAN."""
        self.steps = Variable(torch.zeros([1]))
        self.inputs = inputs
        self.inputs.gan = self
        self.components = {}
        self.destroy = False

        if config == None:
            config = hg.Configuration.default()

        self.config = config
        self._metrics = {}
        self.create()

    def add_metric(self, name, value):
        """adds metric to monitor during training
            name:string
            value:Tensor
        """
        self._metrics[name] = value
        return self._metrics

    def parameters(self):
        for param in self.g_parameters():
            yield param
        for param in self.d_parameters():
            yield param
        yield self.steps

    def g_parameters(self):
        print("Warning: BaseGAN.g_parameters() called directly.  Please override")

    def d_parameters(self):
        print("Warning: BaseGAN.d_parameters() called directly.  Please override")

    def metrics(self):
        """returns a metric : tensor hash"""
        return self._metrics

    def batch_size(self):
        return self.inputs.batch_size()

    def channels(self):
        return self.inputs.channels()

    def width(self):
        return self.inputs.width()

    def height(self):
        return self.inputs.height()

    def output_shape(self):
        return [self.width(), self.height(), self.channels()]

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def add_component(self, name, component):
        index = 0
        while(name+str(index) in self.components):
            index+=1
        self.components[name+str(index)] = component

    def create_component(self, name, *args, **kw_args):
        print("Creating component:", name)
        defn = self.config[name]
        if defn == None:
            print("No definition found for " + name)
            return None
        if defn['class'] == None:
            raise ValidationException("Component definition is missing '" + name + "'")
        klass = GANComponent.lookup_function(None, defn['class'])
        gan_component = klass(self, defn, *args, **kw_args)
        if(isinstance(gan_component, nn.Module)):
            gan_component = gan_component.set_device()
        else:
            print("Warning", name, "is not a nn.Module")
        self.add_component(name, gan_component)
        return gan_component

    def create(self):
        print("Warning: BaseGAN.create() called directly.  Please override")

    def discriminator_fake_inputs(self, discriminator_index=0):
        """
            Fake inputs to the discriminator, should be cached
        """
        []

    def discriminator_real_inputs(self, discriminator_index=0):
        """
            Real inputs to the discriminator, should be cached
        """
        []

    def forward_discriminator(self, inputs, discriminator_index=0):
        """
            Runs a forward pass through the discriminator and returns the discriminator output
        """
        print("Warning: BaseGAN.forward_discriminator() called directly.  Please override")
        return None

    def forward_loss(self):
        """
            Runs a forward pass through the GAN and returns (d_loss, g_loss)
        """
        d_real, d_fake = self.forward_pass()
        return self.loss.forward(d_real, d_fake)

    def forward_pass(self):
        """
            Runs a forward pass through the GAN and returns (d_real, d_fake)
        """
        print("Warning: BaseGAN.forward_pass() called directly.  Please override")
        return None, None

    def step(self, feed_dict={}):
        self.steps += 1
        self._metrics = {}
        return self._step(feed_dict)

    def _step(self, feed_dict={}):
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)

    def save(self, save_file):
        print("Saving..." + str(len(self.components)))
        full_path = os.path.expanduser(os.path.dirname(save_file))
        os.makedirs(full_path, exist_ok=True)
        for name, component in self.components.items():
            self._save(full_path, name, component)

    def _save(self, full_path, name, component):
        path = full_path + "/"+name+".save"
        print("Saving " + path)
        print(component.state_dict().keys())
        torch.save(component.state_dict(), path)

    def load(self, save_file):
        print("Loading..." + str(len(self.components)))
        success = True
        full_path = os.path.expanduser(os.path.dirname(save_file))
        for name, component in self.components.items():
            if not self._load(full_path, name, component):
                print("Error loading", name)
                success = False
        return success

    def _load(self, full_path, name, component):
        path = full_path + "/"+name+".save"
        if Path(path).is_file():
            print("Loading " + path)
            try:
                state_dict = torch.load(path)
                print('state_dict', state_dict.keys())
                component.load_state_dict(state_dict)
                return True
            except:
                print("Warning: Could not load component " + name)
                return False
        else:
            print("Could not load " + path)
            return False


    def parse_args(self, strs):
        options = hc.Config({})
        args = []
        for x in strs:
            if '=' in x:
                lhs, rhs = x.split('=')
                options[lhs]=rhs
            else:
                args.append(x)
        return args, options

    def exit(self):
        self.destroy = True

    def build(self, input_nodes=None, output_nodes=None):
        pass

    def get_registered_samplers(self=None):
        return {
                'static_batch': StaticBatchSampler,
                'input': InputSampler,
                #'progressive': ProgressiveSampler,
                #'random_walk': RandomWalkSampler,
                #'alphagan_random_walk': AlphaganRandomWalkSampler,
                #'style_walk': StyleWalkSampler,
                'batch_walk': BatchWalkSampler,
                'batch': BatchSampler,
                #'grid': GridSampler,
                #'sorted': SortedSampler,
                #'gang': GangSampler,
                #'began': BeganSampler,
                #'autoencode': AutoencodeSampler,
                #'debug': DebugSampler,
                'y': YSampler,
                #'segment': SegmentSampler,
                'aligned': AlignedSampler
            }

    def g_parameters(self):
        for component in self.generator_components():
            for param in component.parameters():
                yield param

    def d_parameters(self):
        for component in self.discriminator_components():
            for param in component.parameters():
                yield param

    def discriminator_components(self):
        print("Warning: BaseGAN.discriminator_components() called directly.  Please override")

    def generator_components(self):
        print("Warning: BaseGAN.generator_components() called directly.  Please override")

    def train_hooks(self):
        result = []
        for component in self.gan.components:
            if hasattr(component, "train_hooks"):
                result += component.train_hooks
        return result

    def sampler_for(self, name, default=StaticBatchSampler):
        samplers = self.get_registered_samplers()
        self.selected_sampler = name
        if name in samplers:
            return samplers[name]
        else:
            print("[hypergan] No sampler found for ", name, ".  Defaulting to", default)
            return default

    def regularize_adversarial_norm(self):
        raise ValidationException("Not implemented")

    def set_generator_trainable(self, flag):
        for c in self.generator_components():
            c.set_trainable(flag)

    def set_discriminator_trainable(self, flag):
        for c in self.discriminator_components():
            c.set_trainable(flag)
