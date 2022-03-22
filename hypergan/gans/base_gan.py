from hyperchamber import Config
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.batch_walk_sampler import BatchWalkSampler
from hypergan.samplers.factorization_batch_walk_sampler import FactorizationBatchWalkSampler
from hypergan.samplers.input_sampler import InputSampler
from hypergan.samplers.grid_sampler import GridSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.y_sampler import YSampler

from hypergan.losses.stable_gan_loss import StableGANLoss
from hypergan.train_hook_collection import TrainHookCollection
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
    def __init__(self, config=None, inputs=None, device="cuda"):
        """ Initialized a new GAN."""
        self._metrics = {}
        self.components = {}
        self.additional_losses = []
        self.destroy = False
        self.inputs = inputs
        self.steps = Variable(torch.zeros([1]))
        self.trainable_gan = None

        if config == None:
            config = hg.Configuration.default()

        self.config = config
        self.device = device
        self.create()
        self.hooks = self.setup_hooks()
        self.train_hooks = TrainHookCollection(self)

    def g_parameters(self):
        print("Warning: BaseGAN.g_parameters() called directly.  Please override")

    def d_parameters(self):
        print("Warning: BaseGAN.d_parameters() called directly.  Please override")

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

    def add_loss(self, loss_function):
        self.additional_losses.append(loss_function)

    def create_component(self, name, *args, **kw_args):
        gan_component = self.initialize_component(name, *args, **kw_args)
        self.add_component(name, gan_component)
        return gan_component

    def initialize_component(self, name, *args, **kw_args):
        print("Creating component:", name)
        if "defn" in kw_args:
            defn = kw_args["defn"]
            del(kw_args["defn"])
        else:
            defn = self.config[name]
        if defn == None:
            print("No definition found for " + name)
            return None
        if defn['class'] == None:
            raise ValidationException("Component definition is missing '" + name + "'")
        klass = GANComponent.lookup_function(None, defn['class'])
        gan_component = klass(self, defn, *args, **kw_args)
        if(isinstance(gan_component, nn.Module)):
            gan_component = gan_component.to(self.device)
        else:
            print("Warning", name, "is not a nn.Module")
        return gan_component

    def create(self):
        print("Warning: BaseGAN.create() called directly.  Please override")

    def discriminator_fake_inputs(self):
        """
            Fake inputs to the discriminator, should be cached
        """
        []

    def discriminator_real_inputs(self):
        """
            Real inputs to the discriminator, should be cached
        """
        []

    def forward_discriminator(self, *inputs):
        """
            Runs a forward pass through the discriminator and returns the discriminator output
        """
        print("Warning: BaseGAN.forward_discriminator() called directly.  Please override")
        return None

    def forward_pass(self):
        """
            Runs a forward pass through the GAN and returns (d_real, d_fake)
        """
        print("Warning: BaseGAN.forward_pass() called directly.  Please override")
        return None, None

    def forward_loss(self, loss):
        d_real, d_fake = self.forward_pass()
        if(self.config.use_stabilized_loss):
            if not hasattr(self, 'stable_gan_loss'):
                self.stable_gan_loss = StableGANLoss(gan=self, gammas=self.config.stable_gammas, offsets = self.config.stable_offsets)
            return self.stable_gan_loss.stable_loss(self.forward_discriminator, self.discriminator_real_inputs(), self.discriminator_fake_inputs()[0], d_fake = self.d_fake, d_real = self.d_real)
        d_loss, g_loss = loss.forward(d_real, d_fake)
        return [d_loss, g_loss]

    def next_inputs(self):
        return None

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
            except Exception as e:
                print("Warning: Could not load component " + name)
                print(e)
                return False
        else:
            print("Could not load " + path)
            return False

    def save(self, full_path):
        print("Saving..." + str(len(self.components)))
        for name, component in self.components.items():
            self._save(full_path, name, component)

    def _save(self, full_path, name, component):
        path = full_path + "/"+name+".save"
        print("Saving " + path)
        print(component.state_dict().keys())
        torch.save(component.state_dict(), path)

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
                'factorization_batch_walk': FactorizationBatchWalkSampler,
                'input': InputSampler,
                #'progressive': ProgressiveSampler,
                #'random_walk': RandomWalkSampler,
                #'alphagan_random_walk': AlphaganRandomWalkSampler,
                #'style_walk': StyleWalkSampler,
                'batch_walk': BatchWalkSampler,
                'batch': BatchSampler,
                'grid': GridSampler,
                #'sorted': SortedSampler,
                #'gang': GangSampler,
                #'began': BeganSampler,
                #'autoencode': AutoencodeSampler,
                #'debug': DebugSampler,
                'y': YSampler,
                #'segment': SegmentSampler,
                'aligned': AlignedSampler
            }

    def discriminator_components(self):
        print("Warning: BaseGAN.discriminator_components() called directly.  Please override")

    def generator_components(self):
        print("Warning: BaseGAN.generator_components() called directly.  Please override")

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

    def latent_parameters(self):
        params = []
        for c in self.generator_components():
            params += c.latent_parameters()
        return params

    def __getstate__(self):
        pickled = dict(self.__dict__)

        del pickled['inputs']
        del pickled['x']
        return pickled
    def __setstate__(self, d):
        self.__dict__ = d

    def to(self, device):
        self.generator = self.generator.to(device)
        self.generator.device=device
        self.discriminator = self.discriminator.to(device)
        self.discriminator.device=device
        self.device = device
        return self #TODO should create new instance

    def create_input(self, blank=False, rank=None):
        klass = GANComponent.lookup_function(None, self.input_config['class'])
        self.input_config["blank"]=blank
        self.input_config["rank"]=rank
        return klass(self.input_config)

    def g_parameters(self):
        for component in self.generator_components():
            for param in component.parameters():
                yield param

    def d_parameters(self):
        for component in self.discriminator_components():
            for param in component.parameters():
                yield param

    def parameters(self):
        for param in self.g_parameters():
            yield param
        for param in self.d_parameters():
            yield param

    def setup_hooks(self, config_name="hooks", add_to_hooks=True):
        hooks = []
        for hook_config in (self.config.trainer[config_name]):
            hook_config = hc.lookup_functions(hook_config.copy())
            defn = {k: v for k, v in hook_config.items() if k in inspect.getargspec(hook_config['class']).args}
            defn['gan']=self
            defn['config']=hook_config
            hook = hook_config["class"](**defn)
            self.add_component("hook", hook)
            if add_to_hooks:
                hooks.append(hook)

        return hooks

    def add_metric(self, name, value):
        """adds metric to the gan
            name:string
            value:Tensor
        """
        self._metrics[name] = value
        return self._metrics

    def metrics(self):
        """returns a metric : tensor hash"""
        return self._metrics
