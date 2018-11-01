"""
Each `hypergan.gan_component` has access to ```self.ops```.

Ops contains our tensorflow graph operations and keeps track of our component weights.
"""
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]


from . import missing_gradients
