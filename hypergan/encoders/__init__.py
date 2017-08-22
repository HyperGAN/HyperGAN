"""
Encoders are the beginning of the network.  In `dcgan` it is a single projection of random noise.
"""
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]
