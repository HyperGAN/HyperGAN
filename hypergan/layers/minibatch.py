# adapted from https://github.com/AaronYALai/Generative_Adversarial_Networks_PyTorch/blob/master/ImprovedGAN/ImprovedGAN.py
import torch.nn as nn
import torch
from hypergan.layer_shape import LayerShape
import hypergan as hg

class Minibatch(hg.Layer):
    """
        ---
        description: 'layer minibatch for the discriminator'
        ---

        # improved GAN minibatch

        ## input size

        Any 1-d input

        ## output size

        output size with additional minibatch logits

        ## syntax

        ```json
          "minibatch"
        ```
    """


    def output_size(self):
        return self.size

    def __init__(self, component, args, options, n_channel=1, use_gpu=False,
                 n_B=128, n_C=16):
        super(Minibatch, self).__init__(component, args, options)
        """
        Minibatch discrimination: learn a tensor to encode side information
        from other examples in the same minibatch.
        """
        super(Minibatch, self).__init__(component, args, options)
        self.use_gpu = use_gpu
        self.n_B = n_B
        self.n_C = n_C

        T_ten_init = torch.randn(component.current_size.dims[0], n_B * n_C) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
        self.fc = nn.Linear(component.current_size.dims[0] + n_B, 1)
        self.size = LayerShape(*[component.current_size.dims[0] + n_B])

    def forward(self, input, context):
        """
        Architecture is similar to DCGANs
        Add minibatch discrimination => Improved GAN.
        """
        T_tensor = self.T_tensor
        if self.use_gpu:
            T_tensor = T_tensor.cuda()

        Ms = input.mm(T_tensor)
        Ms = Ms.view(-1, self.n_B, self.n_C)

        out_tensor = []
        for i in range(Ms.size()[0]):

            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i

            out_tensor.append(out_i)

        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.n_B)
        output = torch.cat((input, out_T), 1)

        return output
