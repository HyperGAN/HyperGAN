import tensorflow as tf
import glob

from gan_component_test import GanComponentTest
from gan_test import GanTest
from configuration_test import ConfigurationTest

from discriminators.pyramid_discriminator_test import PyramidDiscriminatorTest
from encoders.match_discriminator_encoder_test import MatchDiscriminatorEncoderTest
from encoders.uniform_encoder_test import UniformEncoderTest

from generators.align_generator_test import AlignGeneratorTest
from generators.corridor_generator_test import CorridorGeneratorTest
from generators.resize_conv_generator_test import ResizeConvGeneratorTest

from losses.standard_gan_loss_test import StandardGanLossTest

from ops.tensorflow.ops_test import OpsTest

from trainers.alternating_trainer_test import AlternatingTrainerTest
from trainers.proportional_trainer_test import ProportionalTrainerTest

#def test_file(filename):
#    if filename == 'run.py':
#        next
#    import_name = filename[0:-3].replace('/', '.')
#    class_name = "".join(" ".join(import_name.split("_")).title().split(' '))
#    class_name = class_name.split(".")[-1]
#    #print("test file", import_name, class_name)
#    #return __import__(import_name, fromlist=[class_name])
#    print(class_name)
#    mod = __import__(import_name, fromlist=[class_name])
#    import_name = import_name + '.' + class_name
#
#    components = import_name.split('.')
#    print(components)
#    for comp in components[1:]:
#        print(comp, mod)
#        mod = getattr(mod, comp)
#        print(mod)
#    return mod
#
#classes = []
#for f in glob.glob("*test.py"):
#    classes.append(test_file(f))
#
#for f in glob.glob("**/*test.py"):
#    classes.append(test_file(f))
#
print("Running all tests", classes)
tf.test.main()
