import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.skip_connections import SkipConnections

import os
import hypergan as hg
import tensorflow as tf

class BaseGAN(GANComponent):
    def __init__(self, config=None, inputs=None, device='/gpu:0', ops_config=None, ops_backend=TensorflowOps,
            batch_size=None, width=None, height=None, channels=None, debug=None, session=None):
        """ Initialized a new GAN."""
        self.inputs = inputs
        self.device = device
        self.ops_backend = ops_backend
        self.ops_config = ops_config
        self.components = []
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._channels = channels
        self.debug = debug
        self.name = "hypergan"
        self.session = session
        self.skip_connections = SkipConnections()

        if config == None:
            config = hg.Configuration.default()

        if debug and not isinstance(self.session, tf_debug.LocalCLIDebugWrapperSession):
            self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
            self.session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        else:
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            #tfconfig = tf.ConfigProto(log_device_placement=True)
            tfconfig.gpu_options.allow_growth=True

            with tf.device(self.device):
                self.session = self.session or tf.Session(config=tfconfig)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # A GAN as a component has a parent of itself
        # gan.gan.gan.gan.gan.gan
        GANComponent.__init__(self, self, config)
        self.ops.debug = debug

    def batch_size(self):
        if self._batch_size:
            return self._batch_size
        if self.inputs == None:
            raise ValidationException("gan.batch_size() requested but no inputs provided")
        return self.ops.shape(self.inputs.x)[0]

    def channels(self):
        if self._channels:
            return self._channels
        if self.inputs == None:
            raise ValidationException("gan.channels() requested but no inputs provided")
        return self.ops.shape(self.inputs.x)[-1]

    def width(self):
        if self._width:
            return self._width
        if self.inputs == None:
            raise ValidationException("gan.width() requested but no inputs provided")
        return self.ops.shape(self.inputs.x)[2]

    def height(self):
        if self._height:
            return self._height
        if self.inputs == None:
            raise ValidationException("gan.height() requested but no inputs provided")
        return self.ops.shape(self.inputs.x)[1]

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def create_component(self, defn, *args, **kw_args):
        if defn == None:
            return None
        if defn['class'] == None:
            raise ValidationException("Component definition is missing '" + name + "'")
        print('class', defn['class'], self.ops.lookup(defn['class']))
        gan_component = self.ops.lookup(defn['class'])(self, defn, *args, **kw_args)
        self.components.append(gan_component)
        return gan_component

    def create(self):
        raise ValidationException("BaseGAN.create() called directly.  Please override")

    def step(self, feed_dict={}):
        return self._step(feed_dict)

    def _step(self, feed_dict={}):
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)

    def variables(self):
        return self.ops.variables() + self.generator.variables() + self.discriminator.variables()
    def save(self, save_file):
        print("[hypergan] Saving network to ", save_file)
        os.makedirs(os.path.expanduser(os.path.dirname(save_file)), exist_ok=True)
        saver = tf.train.Saver(self.variables())
        saver.save(self.session, save_file)

    def load(self, save_file):
        save_file = os.path.expanduser(save_file)
        if os.path.isfile(save_file) or os.path.isfile(save_file + ".index" ):
            print("[hypergan] |= Loading network from "+ save_file)
            dir = os.path.dirname(save_file)
            print("[hypergan] |= Loading checkpoint from "+ dir)
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver(self.variables())
                saver.restore(self.session, save_file)
                loadedFromSave = True
                return True
            else:
                return False
        else:
            return False


    def variables(self):
        return self.ops.variables() + sum([c.variables() for c in self.components], [])

    def weights(self):
        return self.ops.weights + sum([c.ops.weights for c in self.components], [])

    def inputs(self):
        """inputs() returns any input tensors"""
        return sum([x.inputs() for x in self.components],[])
