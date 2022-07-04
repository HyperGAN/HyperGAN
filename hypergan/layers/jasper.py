import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape
from hypergan.layers.ntm import NTMLayer




activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))

    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["masked"]
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, masked=True):
        super(MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.masked = masked

    def get_seq_len(self, lens):
        # rounding_mode not available in 20.10 container
        # return torch.div((lens + 2 * self.padding[0] - self.dilation[0]
        #                   * (self.kernel_size[0] - 1) - 1), self.stride[0], rounding_mode="floor") + 1
        return torch.floor((lens + 2 * self.padding[0] - self.dilation[0]
                            * (self.kernel_size[0] - 1) - 1) / self.stride[0]).long() + 1

    def forward(self, x, x_lens=None):
        if self.masked:
            max_len = x.size(2)
            idxs = torch.arange(max_len, dtype=x_lens.dtype, device=x_lens.device)
            mask = idxs.expand(x_lens.size(0), max_len) >= x_lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            x_lens = self.get_seq_len(x_lens)

        return super(MaskedConv1d, self).forward(x), x_lens


class JasperBlock(nn.Module):
    __constants__ = ["use_conv_masks"]

    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """
    def __init__(self, infilters, filters, repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, residual_panes=[], use_conv_masks=False, use_batch_norm=True):
        super(JasperBlock, self).__init__()

        assert padding == "same", "Only 'same' padding is supported."

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_masks = use_conv_masks
        self.use_batch_norm = use_batch_norm
        self.conv = nn.ModuleList()
        for i in range(repeat):
            self.conv.extend(self._conv_bn(infilters if i == 0 else filters,
                                           filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           padding=padding_val))
            if i < repeat - 1:
                self.conv.extend(self._act_dropout(dropout, activation))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            if len(residual_panes) == 0:
                res_panes = [infilters]
                self.dense_residual = False

            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    self._conv_bn(ip, filters, kernel_size=1)))

        self.out = nn.Sequential(*self._act_dropout(dropout, activation))

    def _conv_bn(self, in_channels, out_channels, **kw):
        if self.use_batch_norm:
            return [MaskedConv1d(in_channels, out_channels,
                                 masked=self.use_conv_masks, **kw),
                    nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)]
        else:
            return [MaskedConv1d(in_channels, out_channels,
                                 masked=self.use_conv_masks, **kw)]


    def _act_dropout(self, dropout=0.2, activation=None):
        return [activation or nn.Hardtanh(min_val=0.0, max_val=20.0),
                nn.Dropout(p=dropout)]

    def forward(self, xs, xs_lens=None):
        if not self.use_conv_masks:
            xs_lens = 0

        # forward convolutions
        out = xs[-1]
        lens = xs_lens
        for i, l in enumerate(self.conv):
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0:  #  and self.use_conv_mask:
                        res_out, _ = res_layer(res_out, xs_lens)
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_masks:
            return out, lens
        else:
            return out, None


class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_masks"]

    def __init__(self, in_feats, activation, frame_splicing=1, batch_norm=True,
                 init='xavier_uniform', use_conv_masks=False, blocks=[]):
        super(JasperEncoder, self).__init__()
        print("IN FEATS", in_feats)

        self.use_conv_masks = use_conv_masks
        self.layers = nn.ModuleList()

        in_feats *= frame_splicing
        all_residual_panes = []
        print("BLOCKS IS", blocks)
        for i,blk in enumerate(blocks):

            blk['activation'] = activations[activation]()

            has_residual_dense = blk.pop('residual_dense', False)
            if has_residual_dense:
                all_residual_panes += [in_feats]
                blk['residual_panes'] = all_residual_panes
            else:
                blk['residual_panes'] = []

            blk['use_batch_norm'] = batch_norm

            self.layers.append(
                JasperBlock(in_feats, use_conv_masks=use_conv_masks, **blk))

            in_feats = blk['filters']

        self.apply(lambda x: init_weights(x, mode=init))

    def forward(self, x, x_lens=None):
        out, out_lens = [x], x_lens
        for l in self.layers:
            out, out_lens = l(out, out_lens)

        return out, out_lens

class JasperDecoderForCTC(nn.Module):
    def __init__(self, in_feats, n_classes, init='xavier_uniform'):
        super(JasperDecoderForCTC, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_feats, n_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init))

    def forward(self, enc_out):
        out = self.layers(enc_out).transpose(1, 2)
        return out

class Jasper(hg.Layer):
    """
        ---
        description: 'layer jasper'
        ---

        ```
    """
    def __init__(self, component, args, options):
        super(Jasper, self).__init__(component, args, options)
        self.dim = options.dim or 1
        self.size = LayerShape(args[0])

        in_feats = component.current_size.channels
        activation = "relu"
        repeat = 5
        if options.repeat:
            repeat = options.repeat
        dropout = 0
        if options.dropout:
            dropout = 1
        batch_norm = True
        if options.batch_norm:
            batch_norm = options.batch_norm
        filters = options.filters or 256
        blocks = [
        {
          "filters": filters,
          "repeat": 1,
          "kernel_size": [11],
          "stride": [2],
          "dilation": [1],
          "dropout": dropout,
          "residual": False
        },
        {
          "filters": filters,
          "repeat": repeat,
          "kernel_size": [11],
          "stride": [1],
          "dilation": [1],
          "dropout": dropout,
          "residual": True,
          "residual_dense": True 
        },
        {
        "filters": int(filters * 1.5),
      "repeat": repeat,
      "kernel_size": [13],
      "stride": [1],
      "dilation": [1],
      "dropout": dropout,
        "residual": True,
      "residual_dense": True
      },
        {
      "filters": filters*2,
      "repeat": repeat,
      "kernel_size": [17],
      "stride": [1],
      "dilation": [1],
      "dropout": dropout,
      "residual": True,
      "residual_dense": True
      },
        {
      "filters": int(filters*2.5),
      "repeat": repeat,
      "kernel_size": [21],
      "stride": [1],
      "dilation": [1],
      "dropout": dropout,
      "residual": True,
      "residual_dense": True
      },
        {
      "filters": filters*3,
      "repeat": repeat,
      "kernel_size": [25],
      "stride": [1],
      "dilation": [1],
      "dropout": dropout,
      "residual": True,
      "residual_dense": True
      },
        {
      "filters": int(filters*3.5),
      "repeat": 1,
      "kernel_size": [29],
      "stride": [1],
      "dilation": [2],
      "dropout": dropout,
      "residual": False,
      },
      {
      "filters": filters*4,
      "repeat": 1,
      "kernel_size": [1],
      "stride": [1],
      "dilation": [1],
      "dropout": dropout,
      "residual": False
      }
        ]

        depth = options.depth
        if depth is not None:
            blocks = blocks[:depth]
        print("IN FEATS", in_feats)
        self.encoder = JasperEncoder(in_feats, activation, blocks=blocks, batch_norm=batch_norm)
        self.decoder = JasperDecoderForCTC(blocks[-1]["filters"], args[0])

    def forward(self, x, context, epsilon=1e-5):
        enc, lens = self.encoder(x)
        return self.decoder(enc[-1])

    def output_size(self):
        return self.size

