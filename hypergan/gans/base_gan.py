import hyperchamber as hc
from hyperchamber import Config
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.skip_connections import SkipConnections

import math
import re
import os
import inspect
import hypergan as hg
import numpy as np

from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.progressive_sampler import ProgressiveSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.input_sampler import InputSampler
from hypergan.samplers.batch_walk_sampler import BatchWalkSampler
from hypergan.samplers.grid_sampler import GridSampler
from hypergan.samplers.sorted_sampler import SortedSampler
from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.autoencode_sampler import AutoencodeSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.style_walk_sampler import StyleWalkSampler
from hypergan.samplers.alphagan_random_walk_sampler import AlphaganRandomWalkSampler
from hypergan.samplers.debug_sampler import DebugSampler
from hypergan.samplers.segment_sampler import SegmentSampler
from hypergan.samplers.y_sampler import YSampler
from hypergan.samplers.gang_sampler import GangSampler

class BaseGAN(GANComponent):
    def __init__(self, config=None, inputs=None, debug=None, name="hypergan", method="train"):
        """ Initialized a new GAN."""
        self.inputs = inputs
        self.components = []
        self.method = method
        self.debug = debug
        self.name = name
        self.skip_connections = SkipConnections()
        self.destroy = False

        if config == None:
            config = hg.Configuration.default()

        GANComponent.__init__(self, self, config)

    def batch_size(self):
        return self.inputs.samples[0].size()[0]

    def channels(self):
        return self.inputs.samples[0].size()[1]

    def width(self):
        return self.inputs.samples[0].size()[3]

    def height(self):
        return self.inputs.samples[0].size()[2]

    def output_shape(self):
        return [self.width(), self.height(), self.channels()]

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def create_component(self, defn, *args, **kw_args):
        if defn == None:
            return None
        if defn['class'] == None:
            raise ValidationException("Component definition is missing '" + name + "'")
        klass = self.lookup_class(defn['class'])
        gan_component = klass(self, defn, *args, **kw_args)
        self.components.append(gan_component)
        self.components = list(set(self.components))
        return gan_component

    def create_optimizer(self, options):
        options = hc.lookup_functions(options)
        klass = options['class']
        newopts = options.copy()
        newopts['gan']=self.gan
        newopts['config']=options
        defn = {k: v for k, v in newopts.items() if k in inspect.getargspec(klass).args}
        learn_rate = options.learn_rate or options.learning_rate
        if 'learning_rate' in options:
            del defn['learning_rate']
        learn_rate = self.configurable_param(learn_rate)
        self.learn_rate = learn_rate
        gan_component = klass(learn_rate, **defn)
        self.components.append(gan_component)
        return gan_component

    def create(self):
        print("Warning: BaseGAN.create() called directly.  Please override")

    def step(self, feed_dict={}):
        #self.step_count = self.session.run(self.increment_step)
        return self._step(feed_dict)

    def _step(self, feed_dict={}):
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)

    def save(self, save_file):
        pass

    def load(self, save_file):
        pass

    def initialize_variables(self):
        pass

    def metrics(self):
        metrics = {}
        for metric in self._metrics:
            metrics[metric['name']]=metric['value']
        for c in self.components:
            try:
                metrics.update(c.metrics())
            except AttributeError:
                pass
        return metrics

    def configurable_param(self, string):
        self.param_ops = {
            "decay": self.configurable_params_decay,
            "anneal": self.configurable_params_anneal,
            "oscillate": self.configurable_params_oscillate,
            "on": self.configurable_params_turn_on
        }
        if isinstance(string, str):
            if re.match("^-?\d+$", string):
                return int(string)
            if re.match("^-?\d+?\.\d+?$", string):
                return float(string)
            if "(" not in string:
                return string

            method_name, inner = string.split("(")
            inner = inner.replace(")", "")
            if method_name not in self.param_ops:
                raise ValidationException("configurable param cannot find method: "+ method_name + " in string "+string)
            args, options = self.parse_args(inner.split(" "))
            result = self.param_ops[method_name](args, options)
            if "metric" in options:
                self.add_metric(options["metric"], result)
            return result
        return string

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

    def configurable_params_oscillate(self, args, options):
        offset = int(options.offset or 0)
        steps = int(options.T or options.steps or 1000)
        method = options.method or "sin"
        _range = options.range or "0:1"
        r1,r2 = _range.split(":")
        r1 = float(r1)
        r2 = float(r2)
 
        if method == "sin":
            t = self.gan.steps
            t = tf.dtypes.cast(t, tf.float32)
            n1_to_1 = tf.math.sin((tf.constant(math.pi) * 2 * t + tf.constant(offset, tf.float32))/ (steps))
            n = (n1_to_1+1)/2.0
            return (1-n)*r1 + n*r2
        else:
            raise ValidationException(options.method + " not a supported oscillation method")

    def configurable_params_anneal(self, args, options):
        steps = int(options.T or options.steps or 1000)
        alpha = float(args[0])
        t = self.gan.steps
        t = tf.dtypes.cast(t, tf.float32)
        return tf.pow(alpha, tf.dtypes.cast(t, tf.float32) / steps)

    def configurable_params_decay(self, args, options):
        _range = options.range or "0:1"
        steps = int(options.steps or 10000)
        start = int(options.start or 0)
        r1,r2 = _range.split(":")
        r1 = float(r1)
        r2 = float(r2)
        cycle = "cycle" in args
        repeat = "repeat" in args
        current_step = self.gan.steps
        if repeat:
            current_step %= (steps+1)
        if start == 0:
            return tf.train.polynomial_decay(r1, current_step, steps, end_learning_rate=r2, power=1, cycle=cycle)
        else:
            start = tf.constant(start, dtype=tf.int32)
            steps = tf.constant(steps, dtype=tf.int32)
            onoff = tf.minimum(1.0, tf.cast(tf.nn.relu(current_step - start), tf.float32))
            return (1.0 - onoff)*r1 + onoff * tf.train.polynomial_decay(r1, tf.to_float(current_step-start), tf.to_float(steps), end_learning_rate=r2, power=1.0, cycle=cycle)

    def configurable_params_turn_on(self, args, options):
        offset = float(options["offset"]) or 0.0
        if "random" in args:
            onvalue = float(options["onvalue"]) or 1.0
            n = tf.random_uniform([1], minval=-1, maxval=1)
            n += tf.constant(offset, dtype=tf.float32)
            return (tf.sign(n) + 1) /2 * tf.constant(float(options["onvalue"]), dtype=tf.float32)

    def exit(self):
        self.destroy = True

    def build(self, input_nodes=None, output_nodes=None):
        pass

    def get_registered_samplers(self=None):
        return {
                'static_batch': StaticBatchSampler,
                'input': InputSampler,
                'progressive': ProgressiveSampler,
                'random_walk': RandomWalkSampler,
                'alphagan_random_walk': AlphaganRandomWalkSampler,
                'style_walk': StyleWalkSampler,
                'batch_walk': BatchWalkSampler,
                'batch': BatchSampler,
                'grid': GridSampler,
                'sorted': SortedSampler,
                'gang': GangSampler,
                'began': BeganSampler,
                'autoencode': AutoencodeSampler,
                'debug': DebugSampler,
                'y': YSampler,
                'segment': SegmentSampler,
                'aligned': AlignedSampler
            }

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
