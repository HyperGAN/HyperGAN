from distutils.core import setup
from setuptools import setup

setup(
  name = 'hypergan',
  packages = ['hypergan'], # this must be the same as the name above
  version = '0.5.0',
  description = 'A customizable generative adversarial network with good defaults.  Build your own content generator.',
  author = 'Martyn Garcia, Mikkel Garcia',
  author_email = 'mikkel@255bits.com',
  url = 'https://github.com/255BITS/hypergan', 
  download_url = 'https://github.com/255BITS/hypergan/tarball/0.5.0',
  keywords = ['hypergan', 'neural network', 'procedural content generation', 'generative adversarial network'], # arbitrary keywords
  classifiers = ['Topic :: Scientific/Engineering :: Artificial Intelligence', 'Topic :: Artistic Software'],
  scripts = ['bin/hypergan']
)
