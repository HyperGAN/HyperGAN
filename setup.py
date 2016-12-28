from distutils.core import setup
from setuptools import setup
import glob

subpackages = glob.glob("hypergan/*/")
subpackages = [s.replace("/", ".") for s in subpackages]

setup(
  name = 'hypergan',
  packages = ['hypergan']+subpackages,
  version = '0.5.6',
  description = 'A customizable generative adversarial network with good defaults.  Build your own content generator.',
  author = 'Martyn Garcia, Mikkel Garcia',
  author_email = 'mikkel@255bits.com',
  maintainer = "hypergan developers",
  maintainer_email = "mikkel@255bits.com",
  license = "MIT",
  url = 'https://github.com/255BITS/hypergan', 
  keywords = ['hypergan', 'neural network', 'procedural content generation', 'generative adversarial network'], # arbitrary keywords
  classifiers = [
      'Development Status :: 4 - Beta',
      'Topic :: Scientific/Engineering :: Artificial Intelligence', 
      'Topic :: Artistic Software', 
      'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Operating System :: MacOS',
      ],
  platforms = ["Linux", "Mac OS-X", "Unix"],
  scripts = ['bin/hypergan']
)
