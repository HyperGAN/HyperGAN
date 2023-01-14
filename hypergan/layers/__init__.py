"""
Various layer definitions are defined here. The difference between layers and modules are that layers:
    * Have an input and output size
    * Have a standard interface
    * Use parsed arguments from the configurable component
    * Are attached to a Component
"""
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]

from .operation import Operation
from .add import Add
from .sub import Sub
from .cat import Cat
from .mul import Mul
from .attention import Attention
from .channel_attention import ChannelAttention
from .efficient_attention import EfficientAttention
from .ez_norm import EzNorm
from .layer import Layer
from .minibatch import Minibatch
from .multi_head_attention import MultiHeadAttention
from .noise import Noise
from .transformer import Transformer
from .pixel_shuffle import PixelShuffle
from .residual import Residual
from .resizable_stack import ResizableStack
from .evo_norm import EvoNorm
from .rnn import Rnn
from .segment_softmax import SegmentSoftmax
from .skip_connection import SkipConnection
from .slice import Slice
from .synthesis_input import SynthesisInput
from .upsample import Upsample
from .expand import Expand
from .ntm import NTMLayer
from .mlp_mixer import MlpMixer
from .embed import Embed
from .cellular_automata import CellularAutomata
from .cellular_automata_1d import CellularAutomata1D
from .cross_attention import CrossAttention
from .pretrained import Pretrained
from .convnext import ConvNext
from .vit import ViTLayer
from .xunet import XUnetLayer
from .style_swin_layer import StyleSwinLayer

from .jasper import Jasper
