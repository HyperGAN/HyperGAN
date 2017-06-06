import tensorflow as tf
import glob

from gan_component_test import GanComponentTest
from gan_test import GanTest
from gans.standard_gan_test import StandardGanTest
from configuration_test import ConfigurationTest
from cli_test import CliTest

from discriminators.pyramid_discriminator_test import PyramidDiscriminatorTest
from encoders.match_discriminator_encoder_test import MatchDiscriminatorEncoderTest
from encoders.uniform_encoder_test import UniformEncoderTest

from generators.align_generator_test import AlignGeneratorTest
#from generators.corridor_generator_test import CorridorGeneratorTest
from generators.resize_conv_generator_test import ResizeConvGeneratorTest

from losses.standard_gan_loss_test import StandardGanLossTest

from loaders.image_loader_test import ImageLoaderTest

from ops.tensorflow.ops_test import OpsTest

from trainers.alternating_trainer_test import AlternatingTrainerTest
from trainers.proportional_trainer_test import ProportionalTrainerTest


print("[hypergan] Running all unit tests")
tf.test.main()
