import hyperchamber as hc
import inspect
import copy
import re
import os
import operator
from functools import reduce
from math import floor

import pyparsing
import hypergan
import torch
import torch.nn as nn

from .gan_component import GANComponent
from hypergan.gan_component import ValidationException
from hypergan.layer_size import LayerSize

from hypergan.modules.adaptive_instance_norm import AdaptiveInstanceNorm
from hypergan.modules.operation import Operation
from hypergan.modules.attention import Attention
from hypergan.modules.ez_norm import EzNorm
from hypergan.modules.concat_noise import ConcatNoise
from hypergan.modules.const import Const
from hypergan.modules.learned_noise import LearnedNoise
from hypergan.modules.modulated_conv2d import ModulatedConv2d, Blur, EqualLinear
from hypergan.modules.multi_head_attention import MultiHeadAttention
from hypergan.modules.reshape import Reshape
from hypergan.modules.no_op import NoOp
from hypergan.modules.residual import Residual
from hypergan.modules.scaled_conv2d import ScaledConv2d
from hypergan.modules.variational import Variational
from hypergan.modules.pixel_norm import PixelNorm

import torchvision

class ConfigurationException(Exception):
    pass

class ConfigurableComponent(GANComponent):
    def __init__(self, gan, config, input=None, context={}):
        self.current_size = LayerSize(gan.channels(), gan.height(), gan.width())
        if isinstance(input, GANComponent):
            if hasattr(input, 'current_height'):
                self.current_size = LayerSize(input.current_channels, input.current_height, input.current_width)
            elif hasattr(input, 'current_channels'):
                self.current_size = LayerSize(input.current_channels)
            else:
                self.current_size = input.current_size
        self.layers = []
        self.layer_sizes = []
        self.untrainable_parameters = set()
        self.layer_output_sizes = {}
        self.nn_layers = []
        self.layer_options = {}
        self.parsed_layers = []
        self.parser = hypergan.parser.Parser()
        self.layer_ops = {**self.activations(),
            "adaptive_avg_pool": self.layer_adaptive_avg_pool,
            "adaptive_avg_pool1d": self.layer_adaptive_avg_pool1d,
            "adaptive_avg_pool3d": self.layer_adaptive_avg_pool3d,
            "adaptive_instance_norm": self.layer_adaptive_instance_norm,
            "add": self.layer_add,
            "attention": self.layer_attention,
            "avg_pool": self.layer_avg_pool,
            "batch_norm": self.layer_batch_norm,
            "batch_norm1d": self.layer_batch_norm1d,
            "blur": self.layer_blur,
            "cat": self.layer_cat,
            "const": self.layer_const,
            "conv": self.layer_conv,
            "conv1d": self.layer_conv1d,
            "conv2d": self.layer_conv2d,
            "conv3d": self.layer_conv3d,
            "deconv": self.layer_deconv,
            "dropout": self.layer_dropout,
            "equal_linear": self.layer_equal_linear,
            "ez_norm": self.layer_ez_norm,
            "flatten": self.layer_flatten,
            "identity": self.layer_identity,
            "instance_norm": self.layer_instance_norm,
            "instance_norm1d": self.layer_instance_norm1d,
            "instance_norm3d": self.layer_instance_norm3d,
            "latent": self.layer_latent,
            "layer": self.layer_layer,
            "layer_norm": self.layer_norm,
            "learned_noise": self.layer_learned_noise,
            "linear": self.layer_linear,
            #"make2d": self.layer_make2d,
            #"make3d": self.layer_make3d,
            "modulated_conv2d": self.layer_modulated_conv2d,
            "module": self.layer_module,
            "mul": self.layer_mul,
            "multi_head_attention": self.layer_multi_head_attention,
            "pad": self.layer_pad,
            "pixel_norm": self.layer_pixel_norm,
            "pretrained": self.layer_pretrained,
            "reshape": self.layer_reshape,
            "residual": self.layer_residual, #TODO options
            "resize_conv": self.layer_resize_conv,
            "resize_conv2d": self.layer_resize_conv2d,
            "resize_conv1d": self.layer_resize_conv1d,
            "scaled_conv2d": self.layer_scaled_conv2d,
            "split": self.layer_split,
            "subpixel": self.layer_subpixel,
            "upsample": self.layer_upsample,
            "vae": self.layer_vae
            #"add": self.layer_add, #TODO
            # "crop": self.layer_crop,
            # "dropout": self.layer_dropout,
            # "noise": self.layer_noise, #TODO
            #"attention": self.layer_attention, #TODO
            #"const": self.layer_const, #TODO
            #"gram_matrix": self.layer_gram_matrix, #TODO
            #"image_statistics": self.layer_image_statistics, #TODO
            #"knowledge_base": self.layer_knowledge_base, #TODO
            #"layer_norm": self.layer_layer_norm,#TODO
            #"mask": self.layer_mask,#TODO
            #"match_support": self.layer_match_support,#TODO
            #"minibatch": self.layer_minibatch,#TODO
            #"pixel_norm": self.layer_pixel_norm,#TODO
            #"progressive_replace": self.layer_progressive_replace,#TODO
            #"reduce_sum": self.layer_reduce_sum,#TODO might want to just do "reduce sum" instead
            #"relational": self.layer_relational,#TODO
            #"unpool": self.layer_unpool, #TODO https://arxiv.org/abs/1505.04366
            #"squash": self.layer_squash, #TODO
            #"tensorflowcv": self.layer_tensorflowcv, #TODO layer torchvision instead?
            #"turing_test": self.layer_turing_test, #TODO
            #"two_sample_stack": self.layer_two_sample_stack, #TODO
            #"zeros": self.layer_zeros, #TODO
            #"zeros_like": self.layer_zeros_like #TODO
            }
        self.named_layers = {}
        self.context = context
        if not hasattr(gan, "named_layers"):
            gan.named_layers = {}
        self.subnets = hc.Config(hc.Config(config).subnets or {})
        GANComponent.__init__(self, gan, config)
        self.const_two = torch.Tensor([2.0]).float()[0].cuda()
        self.const_one = torch.Tensor([1.0]).float()[0].cuda()

    def required(self):
        return "layers".split()

    def layer(self, name):
        if name in self.gan.named_layers:
            return self.gan.named_layers[name] 
        if name in self.named_layers:
            return self.named_layers[name]
        return None

    def build(self, net, replace_controls={}, context={}):
        self.replace_controls=replace_controls
        config = self.config

        for name, layer in self.context.items():
            self.set_layer(name, layer)

        for name, layer in context.items():
            self.set_layer(name, layer)

        for layer in config.layers:
            net = self.parse_layer(net, layer)
            self.layers += [net]

        return net

    def create(self):
        for layer in self.config.layers:
            net = self.parse_layer(layer)
            self.nn_layers.append(net)

        self.net = nn.Sequential(*self.nn_layers)

    def parse_layer(self, layer):
        config = self.config

        if isinstance(layer, list):
            ns = []
            axis = -1
            for l in layer:
                if isinstance(l, int):
                    axis = l
                    continue
                n = self.parse_layer(l)
                ns += [n]

        else:
            print("Parsing layer:", layer)
            parsed = self.parser.parse_string(layer)
            parsed.parsed_options = hc.Config(parsed.options)
            print("Parsed layer:", parsed.to_list())
            self.parsed_layers.append(parsed)
            layer = self.build_layer(parsed.layer_name, parsed.args, parsed.parsed_options)
            self.layer_sizes.append(self.current_size)
            return layer

    def build_layer(self, op, args, options):
        if self.layer_ops[op]:
            if isinstance(self.layer_ops[op], nn.Module):
                net = self.layer_ops[op]
            else:
                #before_count = self.count_number_trainable_params()
                print("Size before: ", self.current_size.dims)
                net = self.layer_ops[op](None, args, options)
                print("Size after: ", self.current_size.dims)
            if 'name' in options:
                self.set_layer(options['name'], net)
            if options.trainable == False:
                self.untrainable_parameters = self.untrainable_parameters.union(set(net.parameters()))
            return net

            #after = self.variables()
            #new = set(after) - set(before)
            #for j in new:
            #    self.layer_options[j]=options
            #after_count = self.count_number_trainable_params()
            #if not self.ops._reuse:
            #    if net == None:
            #        print("[Error] Layer resulted in null return value: ", op, args, options)
            #        raise ValidationException("Configurable layer is null")
            #    print("layer: ", self.ops.shape(net), op, args, after_count-before_count, "params")
        else:
            print("ConfigurableComponent: Op not defined", op)

    def set_layer(self, name, net):
        self.gan.named_layers[name] = net
        self.named_layers[name]     = net
        self.layer_output_sizes[name] = self.current_size

    def activations(self):
        return {
            "celu": nn.CELU(),
            "gelu": nn.GELU(),
            "lrelu": nn.LeakyReLU(0.2),
            "prelu": nn.PReLU(),
            "relu": nn.ReLU(),
            "relu6": nn.ReLU6(),
            "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus(),
            "softshrink": nn.Softshrink(),
            "softsign": nn.Softsign(),
            "hardtanh": nn.Hardtanh(),
            "tanh": nn.Tanh(),
            "tanhshrink": nn.Tanhshrink()
        }

    def layer_residual(self, net, args, options):
        return Residual(self.current_size.channels)

    def layer_dropout(self, net, args, options):
        return nn.Dropout2d(float(args[0]))

    def layer_identity(self, net, args, options):
        return NoOp()

    def layer_equal_linear(self, net, args, options):
        lr_mul = 1
        if options.lr_mul is not None:
            lr_mul = options.lr_mul
        result = EqualLinear(self.current_size.size(), args[0], lr_mul=lr_mul)
        self.current_size = LayerSize(args[0])
        return result

    def get_same_padding(self, input_rows, filter_rows, stride, dilation):
        out_rows = (input_rows + stride - 1) // stride
        return max(0, (out_rows - 1) * stride + (filter_rows - 1) * dilation + 1 - input_rows) // 2

    def layer_const(self, net, args, options):
        return Const(*self.current_size.dims)

    #from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/3
    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w

    def layer_conv(self, net, args, options):
        return self.layer_conv2d(net, args, options)

    def layer_conv2d(self, net, args, options):
        if len(args) > 0:
            channels = args[0]
        else:
            channels = self.current_size.channels
        options = hc.Config(options)
        stride = 1
        if options.stride is not None:
            stride = options.stride
        filter = 3
        if options.filter is not None:
            filter = options.filter
        padding = 1
        if options.padding is not None:
            padding = options.padding

        dilation = 1

        layer = nn.Conv2d(options.input_channels or self.current_size.channels, channels, filter, stride, padding = (padding, padding))
        self.nn_init(layer, options.initializer)
        h, w = self.conv_output_shape((self.current_size.height, self.current_size.width), filter, stride, padding, dilation)
        self.current_size = LayerSize(channels, h, w)
        return layer

    def layer_conv1d(self, net, args, options):
        if len(args) > 0:
            channels = args[0]
        else:
            channels = self.current_size.channels
        print("Options:", options)
        options = hc.Config(options)
        stride = options.stride or 1
        fltr = options.filter or 3
        dilation = 1

        padding = options.padding or 1#self.get_same_padding(self.current_width, self.current_width, stride, dilation)

        layers = [nn.Conv1d(options.input_channels or self.current_size.channels, channels, fltr, stride, padding = padding)]
        self.nn_init(layers[-1], options.initializer)
        self.current_size = LayerSize(channels, self.current_size.height // stride) #TODO better calculation of this
        return nn.Sequential(*layers)


    def layer_conv3d(self, net, args, options):
        if len(args) > 0:
            channels = args[0]
        else:
            channels = self.current_size.channels
        options = hc.Config(options)
        stride = options.stride or 1
        fltr = options.filter or 3
        dilation = 1

        padding = options.padding or 1#self.get_same_padding(self.current_width, self.current_width, stride, dilation)
        if options.padding0:
            padding = [options.padding0, padding, padding]
        if options.stride0:
            stride = [options.stride0, stride, stride]
        else:
            stride = [stride, stride, stride]

        layers = [nn.Conv3d(options.input_channels or self.current_size.channels, channels, fltr, stride, padding = padding)]
        self.nn_init(layer, options.initializer)
        self.current_size = LayerSize(frames, channels, self.current_size.height // stride[1], self.current_size.width // stride[2]) #TODO this doesn't work, what is frames? Also chw calculation like conv2d
        return nn.Sequential(*layers)

    def layer_linear(self, net, args, options):
        options = hc.Config(options)
        shape = [int(x) for x in str(args[0]).split("*")]
        bias = True
        if options.bias == False:
            bias = False
        output_size = 1
        for dim in shape:
            output_size *= dim
        layers = []
        if len(self.current_size.dims) != 1:
            layers += [nn.Flatten()]

        layers += [nn.Linear(options.input_size or self.current_size.size(), output_size, bias=bias)]
        self.nn_init(layers[-1], options.initializer)
        self.current_size = LayerSize(*list(reversed(shape)))
        if len(shape) != 1:
            layers.append(Reshape(*self.current_size.dims))

        return nn.Sequential(*layers)

    #def layer_make2d(self, net, args, options):
    #    return NoOp()

    #def layer_make3d(self, net, args, options):
    #    return NoOp()

    def layer_modulated_conv2d(self, net, args, options):
        channels = self.current_size.channels
        if len(args) > 0:
            channels = args[0]
        method = "conv"
        if len(args) > 1:
            method = args[1]
        upsample = method == "upsample"
        downsample = method == "downsample"

        demodulate = True
        if options.demodulate == False:
            demodulate = False

        filter = 3
        if options.filter:
            filter = options.filter

        lr_mul = 1.0
        if options.lr_mul:
            lr_mul = options.lr_mul
        input_channels = self.current_size.channels
        if options.input_channels:
            input_channels = options.input_channels

        result = ModulatedConv2d(input_channels, channels, filter, self.layer_output_sizes['w'].size(), upsample=upsample, demodulate=demodulate, downsample=downsample, lr_mul=lr_mul)

        if upsample:
            self.current_size = LayerSize(channels, self.current_size.height * 2, self.current_size.width * 2)
        elif downsample:
            self.current_size = LayerSize(channels, self.current_size.height // 2, self.current_size.width // 2)
        return result

    def layer_module(self, net, args, options):
        klass = GANComponent.lookup_function(None,"function:__main__."+args[0])
        instance = klass(self.gan, net, args, options, self.current_size)
        self.current_size = instance.layer_size(self.current_size)
        return instance

    def layer_blur(self, net, args, options):
        blur_kernel=[1, 3, 3, 1]
        kernel_size=3
        factor = 2
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1

        return Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

    def layer_reshape(self, net, args, options):
        dims_args = [int(x) for x in args[0].split("*")]
        dims = list(reversed(dims_args))
        self.current_size = LayerSize(*dims)
        return Reshape(*dims)

    def layer_adaptive_avg_pool(self, net, args, options):
        self.current_size = LayerSize(self.current_size.channels, self.current_size.height // 2, self.current_size.width // 2)
        return nn.AdaptiveAvgPool2d([self.current_size.height, self.current_size.width])

    def layer_adaptive_avg_pool1d (self, net, args, options):
        self.current_size = LayerSize(self.current_size.channels, self.current_size.height // 2)
        return nn.AdaptiveAvgPool1d(self.current_size.height)

    def layer_avg_pool(self, net, args, options):
        self.current_size = LayerSize(self.current_size.channels, self.current_size.height // 2, self.current_size.width // 2)
        return nn.AvgPool2d(2, 2)

    def layer_adaptive_avg_pool3d(self, net, args, options):
        frames = 4 #TODO
        self.current_size = LayerSize(frames, self.current_size.channels, self.current_size.height // 2, self.current_size.width // 2)
        return nn.AdaptiveAvgPool3d([self.current_size.frames, self.current_size.height, self.current_size.width]) #TODO looks wrong

    def layer_instance_norm(self, net, args, options):
        options = hc.Config(options)
        affine = True
        if options.affine == False:
            affine = False
        return nn.InstanceNorm2d(self.current_size.channels, affine=affine)

    def layer_instance_norm1d(self, net, args, options):
        options = hc.Config(options)
        affine = True
        if options.affine == False:
            affine = False
        return nn.InstanceNorm1d(self.current_size.channels, affine=affine)


    def layer_instance_norm3d(self, net, args, options):
        options = hc.Config(options)
        affine = True
        if options.affine == False:
            affine = False
        return nn.InstanceNorm3d(self.current_size.channels, affine=affine)

    def layer_batch_norm(self, net, args, options):
        return nn.BatchNorm2d(self.current_size.channels)

    def layer_batch_norm1d(self, net, args, options):
        return nn.BatchNorm1d(self.current_size.size())

    def get_conv_options(self, config, options):
        stride = options.stride or self.ops.config_option("stride", [1,1])
        fltr = options.filter or self.ops.config_option("filter", [3,3])
        avg_pool = options.avg_pool or self.ops.config_option("avg_pool", [1,1])

        if type(stride) != type([]):
            stride = [stride, stride]

        if type(avg_pool) != type([]):
            avg_pool = [avg_pool, avg_pool]

        if type(fltr) != type([]):
            fltr = [fltr, fltr]
        return stride, fltr, avg_pool


    def layer_deconv(self, net, args, options):
        if len(args) > 0:
            channels = args[0]
        else:
            channels = self.current_size.channels
        options = hc.Config(options)
        filter = 4 #TODO
        if options.filter:
            filter = options.filter
        stride = 2
        if options.stride:
            stride = options.stride
        padding = 1
        if options.padding:
            padding = options.padding
        layer = nn.ConvTranspose2d(options.input_channels or self.current_size.channels, channels, filter, stride, padding)
        self.nn_init(layer, options.initializer)
        self.current_size = LayerSize(channels, self.current_size.height * 2, self.current_size.width * 2)
        return layer

    def layer_pad(self, net, args, options):
        options = hc.Config(options)

        return nn.ZeroPad2d((args[0], args[1], args[2], args[3]))

    def layer_pixel_norm(self, net, args, options):
        return PixelNorm()

    def layer_pretrained(self, net, args, options):
        model = getattr(torchvision.models, args[0])(pretrained=True)
        model.train(True)
        if options.layer:
            layers = list(model.children())[:options.layer]
            if options.sublayer:
                layers[-1] = nn.Sequential(*layers[-1][:options.sublayer])
        else:
            layers = [model]
            print("List of pretrained layers:", layers)
            raise ValidationException("layer=-1 required for pretrained, sublayer=-1 optional.  Layers outputted above.")
        return nn.Sequential(*layers)

    def layer_upsample(self, net, args, options):
        w = options.w or self.current_size.width * 2
        h = options.h or self.current_size.height * 2
        result = nn.Upsample((h, w), mode="bilinear")
        self.current_size = LayerSize(self.current_size.channels, h, w)
        return result

    def layer_resize_conv(self, net, args, options):
        return self.layer_resize_conv2d(net, args, options)

    def layer_resize_conv2d(self, net, args, options):
        options = hc.Config(options)
        channels = args[0]

        w = options.w or self.current_size.width * 2
        h = options.h or self.current_size.height * 2
        layers = [nn.Upsample((h, w), mode="bilinear"),
                nn.Conv2d(options.input_channels or self.current_size.channels, channels, options.filter or 3, 1, 1)]
        self.nn_init(layers[-1], options.initializer)
        self.current_size = LayerSize(channels, h, w)
        return nn.Sequential(*layers)

    def layer_resize_conv1d(self, net, args, options):
        options = hc.Config(options)
        channels = args[0]
        h = options.h or self.current_size.height * 2

        layers = [nn.Upsample((h)),
                nn.Conv1d(options.input_channels or self.current_size.channels, channels, options.filter or 3, 1, 1)]
        self.nn_init(layers[-1], options.initializer)
        self.current_size = LayerSize(channels, h)
        return nn.Sequential(*layers)

    def layer_scaled_conv2d(self, net, args, options):
        channels = self.current_size.channels
        if len(args) > 0:
            channels = args[0]
        method = "conv"
        if len(args) > 1:
            method = args[1]
        upsample = method == "upsample"
        downsample = method == "downsample"

        demodulate = True
        if options.demodulate == False:
            demodulate = False

        filter = 3
        if options.filter:
            filter = options.filter

        lr_mul = 1.0
        if options.lr_mul:
            lr_mul = options.lr_mul
        input_channels = self.current_size.channels
        if options.input_channels:
            input_channels = options.input_channels

        result = ScaledConv2d(input_channels, channels, filter, 0, upsample=upsample, demodulate=demodulate, downsample=downsample, lr_mul=lr_mul)
        self.nn_init(result, options.initializer)

        if upsample:
            self.current_size = LayerSize(channels, self.current_size.height * 2, self.current_size.width * 2)
        else:
            self.current_size = LayerSize(channels, self.current_size.height - 2, self.current_size.width - 2)
        return result

    def layer_split(self, net, args, options):
        options = hc.Config(options)
        split_size = args[0]
        select = args[1]
        dim = -1
        if options.dim:
            dim = options.dim
        #TODO better validation
        #TODO increase dim options
        if dim == -1:
            dims = list(self.current_size.dims).copy()
            dims[0] = split_size
            if (select + 1) * split_size > self.current_size.channels:
                dims[0] = self.current_size.channels % split_size
            self.current_size = LayerSize(*dims)
        return NoOp()

    def layer_subpixel(self, net, args, options):
        options = hc.Config(options)
        channels = args[0]

        layers = [nn.Conv2d(options.input_channels or self.current_size.channels, channels*4, options.filter or 3, 1, 1), nn.PixelShuffle(2)]
        self.nn_init(layers[0], options.initializer)
        self.current_size = LayerSize(channels, self.current_size.height * 2, self.current_size.width * 2)
        return nn.Sequential(*layers)

    def layer_latent(self, net, args, options):
        self.current_size = LayerSize(self.gan.latent.current_input_size)
        return NoOp()

    def layer_layer(self, net, args, options):
        if args[0] in self.layer_output_sizes:
            self.current_size = self.layer_output_sizes[args[0]]
        return NoOp()

    def layer_linformer(self, net, args, options):
        model = Linformer(
                input_size = self.current_size.size(),
                channels = self.current_size.height # TODO wtf
        )
        return model

    def layer_vae(self, net, args, options):
        self.vae = Variational(self.current_size.channels)
        return self.vae

    def layer_add(self, net, args, options):
        return self.operation_layer(net, args, options, "+")

    def layer_cat(self, net, args, options):
        result = self.operation_layer(net, args, options, "cat")
        dims = None
        for arg in args:
            if arg == "self":
                new_size = self.current_size
            else:
                new_size = self.layer_output_sizes[args[1]]
            if dims is None:
                dims = new_size.dims
            else:
                dims = [dims[0]+new_size.dims[0]] + list(dims[1:])

        self.current_size = LayerSize(*dims)
        return result

    def layer_mul(self, net, args, options):
        return self.operation_layer(net, args, options, "*")

    def layer_multi_head_attention(self, net, args, options):
        output_size = self.current_size.size()
        if len(args) > 0:
            output_size = args[0]
        layer = MultiHeadAttention(self.current_size.size(), output_size, heads=options.heads or 4)
        self.current_size = LayerSize(output_size)
        self.nn_init(layer.o, options.initializer)
        self.nn_init(layer.h, options.initializer)
        self.nn_init(layer.g, options.initializer)
        self.nn_init(layer.f, options.initializer)
        return layer

    def operation_layer(self, net, args, options, operation):
        options = hc.Config(options)
        layers = []
        layer_names = []
        current_size = self.current_size

        for arg in args:
            self.current_size = current_size
            if arg == 'self':
                layers.append(None)
                layer_names.append("self")
            elif arg == 'noise':
                layers.append(LearnedNoise())
                layer_names.append(None)
            elif arg in self.named_layers:
                layers.append(NoOp())
                layer_names.append("layer "+arg)
            elif arg in self.gan.named_layers:
                layers.append(self.gan.named_layers[arg])
                layer_names.append(None)
            elif type(arg) == pyparsing.ParseResults and type(arg[0]) == hypergan.parser.Pattern:
                print( "Found pattern")
                parsed = arg[0]
                parsed.parsed_options = hc.Config(parsed.options)
                layers.append(self.build_layer(parsed.layer_name, parsed.args, parsed.parsed_options))
                layer_names.append(parsed.layer_name)
            else:
                print("arg", type(arg))
                raise "Could not parse add layer "
        return Operation(layers, layer_names, operation)

    def layer_attention(self, net, args, options):
        layer = Attention(self.current_size.channels)
        self.nn_init(layer.v, options.initializer)
        self.nn_init(layer.h, options.initializer)
        self.nn_init(layer.g, options.initializer)
        self.nn_init(layer.f, options.initializer)
        return layer

    def layer_norm(self, net, args, options):
        affine = True
        if options.affine == False:
            affine = False

        return nn.LayerNorm(self.current_size.dims, elementwise_affine=affine)

    def layer_learned_noise(self, net, args, options):
        return LearnedNoise(*([self.gan.batch_size(), *self.current_size.dims]))

    def layer_concat(self, net, args, options):
        options = hc.Config(options)
        if args[0] == 'noise':
            dims = self.current_size.dims.copy()
            dims[0] *= 2
            self.current_size = LayerSize(*dims)
            return NoOp()
        elif args[0] == 'layer':
            return NoOp()
        else:
            print("Got: ", args[0])
            print("Warning: only 'concat noise' and 'concat layer' is supported for now.")

    #def layer_concat3d(self, net, args, options):
    #    options = hc.Config(options)
    #    if args[0] == 'noise':
    #        print("Concat noise!")
    #        return NoOp()
    #    elif args[0] == 'layer':
    #        return NoOp()
    #    else:
    #        print("Got: ", args[0])
    #        print("Warning: only 'concat noise' and 'concat layer' is supported for now.")

    def layer_adaptive_instance_norm(self, net, args, options):
        return AdaptiveInstanceNorm(self.layer_output_sizes['w'].size(), self.current_size.channels, equal_linear=options.equal_linear)

    def layer_ez_norm(self, net, args, options):
        layer = EzNorm(self.layer_output_sizes['w'].size(), self.current_size.channels, len(self.current_size.dims), equal_linear=options.equal_linear)
        self.nn_init(layer.beta, options.initializer)
        self.nn_init(layer.conv, options.initializer)
        return layer

    def layer_flatten(self, net, args, options):
        self.current_size = LayerSize(self.current_size.size())
        return nn.Flatten()

    def layer_zeros_like(self, net, args, options):
        return Zeros(self.gan.latent.sample().shape)

    def nn_init(self, layer, initializer_option):
        if initializer_option is None:
            return
        if type(initializer_option) == pyparsing.ParseResults and type(initializer_option[0]) == hypergan.parser.Pattern:
            args = [initializer_option[0].layer_name] + initializer_option[0].args
            options = hc.Config(initializer_option[0].options)
        else:
            args = [initializer_option]
            options = hc.Config({})

        layer_data = layer.weight.data

        if args[0] == "uniform":
            a = float(args[1])
            b = float(args[2])
            nn.init.uniform_(layer_data, a, b)
        elif args[0] == "normal":
            mean = float(args[1])
            std = float(args[2])
            nn.init.normal_(layer_data, mean, std)
        elif args[0] == "constant":
            val = float(args[1])
            nn.init.constant_(layer_data, val)
        elif args[0] == "ones":
            nn.init.ones_(layer_data)
        elif args[0] == "zeros":
            nn.init.zeros_(layer_data)
        elif args[0] == "eye":
            nn.init.eye_(layer_data)
        elif args[0] == "dirac":
            nn.init.dirac_(layer_data)
        elif args[0] == "xavier_uniform":
            gain = nn.init.calculate_gain(options.gain or "relu")
            nn.init.xavier_uniform_(layer_data, gain=gain)
        elif args[0] == "xavier_normal":
            gain = nn.init.calculate_gain(options.gain or "relu")
            nn.init.xavier_normal_(layer_data, gain=gain)
        elif args[0] == "kaiming_uniform":
            a = 0 #TODO wrong
            nn.init.kaiming_uniform_(layer_data, mode=(options.mode or "fan_in"), nonlinearity=options.gain or "relu")
        elif args[0] == "kaiming_normal":
            a = 0 #TODO wrong
            nn.init.kaiming_normal_(layer_data, mode=(options.mode or "fan_in"), nonlinearity=options.gain or "relu")
        elif args[0] == "orthogonal":
            if "gain" in options:
                gain = nn.init.calculate_gain(options["gain"])
            else:
                gain = 1
            nn.init.orthogonal_(layer_data, gain=gain)
        else:
            print("Warning: No initializer found for " + args[0])
        if "gain" in options:
            layer_data.mul_(nn.init.calculate_gain(options["gain"]))
        return NoOp()


    def forward(self, input, context={}):
        self.context = context
        for module, parsed, layer_size in zip(self.net, self.parsed_layers, self.layer_sizes):
            options = parsed.parsed_options
            args = parsed.args
            layer_name = parsed.layer_name
            name = options.name
            if layer_name == "adaptive_instance_norm":
                input = module(input, self.context['w'])
            elif layer_name == "ez_norm":
                input = module(input, self.context['w'])
            elif layer_name == "add":
                input = module(input, self.context)
            elif layer_name == "mul":
                input = module(input, self.context)
            elif layer_name == "cat":
                input = module(input, self.context)
            elif layer_name == "split":
                input = torch.split(input, args[0], options.dim or -1)[args[1]]
            # elif layer_name == "concat3d":
            #    if args[0] == "layer":
            #        input = torch.cat((input, self.context[args[1]]), dim=2)
            #    elif args[0] == "noise":
            #        input = torch.cat((input, torch.randn_like(input)*0.01), dim=2)
            # elif layer_name == "make2d":
            #     input = torch.squeeze(input, dim=2) #TODO only 3d -> 2d
            # elif layer_name == "make3d":
            #     input = input[:,:,None,:,:] #TODO only 2d -> 3d

            elif layer_name == "layer":
                input = self.context[args[0]]
            elif layer_name == "latent":
                input = self.gan.latent.sample()
            elif layer_name == "modulated_conv2d":
                input = module(input, self.context['w'])
            elif layer_name == "pretrained":
                in_zero_one = (input + self.const_one) / self.const_two
                mean = torch.as_tensor([0.485, 0.456, 0.406], device='cuda:0').view(1, 3, 1, 1)
                std = torch.as_tensor([0.229, 0.224, 0.225], device='cuda:0').view(1, 3, 1, 1)

                input = module(input.clone().sub_(mean).div_(std))
            else:
                input = module(input)
            if self.gan.steps == 0:
                size = LayerSize(*list(input.shape[1:]))
                if size.squeeze_dims() != layer_size.squeeze_dims():
                    print("Error: Size error on", layer_name)
                    print("Error: Expected output size", layer_size.dims)
                    print("Error: Actual output size", size.dims)
                    raise "Layer size error, cannot continue"
                else:
                    print("Sizes as expected", input.shape[1:], layer_size.dims)
            if name is not None:
                self.context[name] = input
        self.sample = input
        return input

    def set_trainable(self, flag):
        for p in (set(list(self.parameters())) - self.untrainable_parameters):
            p.requires_grad = flag
