"""
# HyperGAN

A composable GAN API and CLI.  Built for developers, researchers, and artists.

HyperGAN is currently in open beta.

![Colorizer 0.9 1](https://s3.amazonaws.com/hypergan-apidocs/0.9.0-images/colorizer-2.gif)

Please see [https://github.com/255BITS/HyperGAN](https://github.com/255BITS/HyperGAN) for an introduction, usage and API examples.

## License

MIT - https://opensource.org/licenses/MIT
"""
import hypergan
from .gan import GAN
from .cli import CLI
from .configuration import Configuration
import tensorflow as tf
import hypergan.cli
import hypergan as hg

