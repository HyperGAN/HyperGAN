import torch
import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def instance_std(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

class EvoNorm2D(nn.Module):

    def __init__(self, input, non_linear = True, version = 'S0', efficient = False, affine = True, momentum = 0.9, eps = 1e-5, groups = 32, training = True):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == 'S0':
            self.swish = MemoryEfficientSwish()
        self.groups = min(input, groups)
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
    
    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(self.v * x)   # Original Swish Implementation, however memory intensive.
                else:
                    num = self.swish(x)    # Experimental Memory Efficient Variant of Swish
                return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim = (0, 2, 3), unbiased = False, keepdim = True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std(x, eps = self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta



class EvoNorm(hg.Layer):
    """
        ---
        description: 'layer ez_norm for configurable component'
        ---

        # ez_norm layer

        `ez_norm` is a custom normalization technique that uses a conv of the input by a linear projection of a style vector.

        ## Optional arguments

            `style` - The name of the style vector to use. Defaults to "w"

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "ez_norm style=[style vector name]"
        ```

        ## examples

        ```json
          "latent name=w",
          ...
          "cat self (ez_norm style=w)"
        ```
    """
    def __init__(self, component, args, options):
        super(EvoNorm, self).__init__(component, args, options)
        channels = component.current_size.channels
        self.size = LayerShape(*component.current_size.dims)
        self.evonorm = EvoNorm2D(channels)

    def forward(self, input, context):
        return self.evonorm(input)

    def output_size(self):
        return self.size

