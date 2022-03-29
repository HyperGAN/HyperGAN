from .base_generator import BaseGenerator
from .gansformer_generator import SynthesisNetwork, FullyConnectedLayer, random_dp_binary, get_embeddings
from hypergan.layers import ntm
from torch_utils import misc as torch_misc
from torch_utils import persistence
import torch
from torch import nn

# Use an NTM as a mapping layer
class Generator(BaseGenerator):
    def __init__(self, gan, config, input=None, name=None):
        super().__init__(gan, config)
        self.k = self.config.k or 1
        self.z_dim = self.config.z_dim or 512
        self.c_dim = self.config.c_dim or 512
        self.w_dim = self.config.w_dim or 512
        self.img_resolution = self.config.img_resolution or gan.width()
        self.img_channels = self.config.img_channels or 3
        self.update_emas = self.config.update_emas or False
        self.truncation_cutoff = self.config.truncation_cutoff or None
        self.truncation_psi = self.config.truncation_psi or 1
        self.component_dropout = 0.0
        self.transformer = self.config.transformer or False

        self.input_shape = [None, self.k, self.z_dim]
        self.cond_shape  = [None, self.c_dim]

        self.pos = get_embeddings(self.k - 1, self.w_dim)

        self.synthesis = SynthesisNetwork(w_dim = self.w_dim, k = self.k, img_resolution = self.img_resolution, local_noise = False,
                img_channels = self.img_channels)
        self.num_ws = self.synthesis.num_ws

        heads = []
        N=self.config.N or 1024
        M=self.config.M or 32
        num_heads = self.config.num_heads or 2
        num_layers = self.config.num_layers or 0
        self.memory = ntm.FastMemory(N, M).cuda()
        for i in range(num_heads):
            heads += [
                ntm.NTMReadHead(self.memory, self.z_dim).cuda(),
                ntm.NTMWriteHead(self.memory, self.z_dim).cuda()
            ]


        self.ntm = ntm.NTM(self.z_dim, (self.w_dim*self.num_ws)+self.c_dim, self.memory, heads).cuda()
        self.hardtanh = torch.nn.Hardtanh()
        if num_layers == 0:
            self.z_controller = FullyConnectedLayer(self.z_dim, self.z_dim)
        else:
            self.z_controller = nn.Sequential(*([FullyConnectedLayer(self.z_dim, self.z_dim, act="lrelu") for i in range(num_layers-1)] + [FullyConnectedLayer(self.z_dim, self.z_dim)]))

    def forward(self, input):
        truncation_psi = 1
        truncation_cutoff = None
        return_img = True
        return_att = False
        return_ws = False
        subnet = None
        z = self.z_controller(input)
        self.memory.reset(input.shape[0])
        z = self.ntm(self.hardtanh(z))
        z, c = z[:,:self.w_dim*self.num_ws], z[:, self.w_dim*self.num_ws:]

        return_tensor = True
        self.training = True
        if subnet is not None:
            return_ws = (subnet == "mapping")
            return_img = (subnet == "synthesis")
            return_att = False
            return_tensor = True

        _input = z if z is not None else ws
        mask = random_dp_binary([_input.shape[0], self.k - 1], self.component_dropout, self.training, _input.device)

        ws = z.view(z.shape[0], self.k, self.num_ws, self.z_dim)
        torch_misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])

        ret = ()
        if return_img or return_att:
            img, att_maps = self.synthesis(ws, pos = self.pos, mask = mask)
            if return_img:  ret += (img, )
            if return_att:  ret += (att_maps, )

        if return_ws:  ret += (ws, )

        if return_tensor:
            ret = ret[0]

        return ret

    def create(self):
        pass

    def latent_parameters(self):
        return [self.z_controller.weight]
