import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.skip_connections import SkipConnections

import os
import inspect
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
        self.destroy = False

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
        GANComponent.__init__(self, self, config, name=self.name)
        self.ops.debug = debug

    def batch_size(self):
        if self._batch_size:
            return self._batch_size
        if self.inputs == None:
            raise ValidationException("gan.batch_size() requested but no inputs provided")
        return self.ops.shape(self.inputs.x)[0]

    def sample_mixture(self):
        diff = self.inputs.x - self.generator.sample
        alpha = tf.random_uniform(shape=self.ops.shape(self.generator.sample), minval=0., maxval=1.0)
        return self.inputs.x + alpha * diff

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

    def output_shape(self):
        return [self.width(), self.height(), self.channels()]

    def l1_distance(self):
        return self.inputs.x - self.generator.sample

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

    def create_optimizer(self, options):
        options = hc.lookup_functions(options)
        klass = options['class']
        newopts = options.copy()
        newopts['gan']=self.gan
        newopts['config']=options
        defn = {k: v for k, v in newopts.items() if k in inspect.getargspec(klass).args}
        learn_rate = options.learn_rate or options.learning_rate
        if 'learning_rate' in options:
            del defn['learning_rate']
        gan_component = klass(learn_rate, **defn)
        self.components.append(gan_component)
        return gan_component

    def create_generator(self, _input, reuse=False):
        return self.gan.create_component(self.gan.config.generator, name='generator', input=_input, reuse=reuse)

    def create_discriminator(self, _input, reuse=False):
        return self.gan.create_component(self.gan.config.discriminator, name="discriminator", input=_input, reuse=True)

    def create(self):
        raise ValidationException("BaseGAN.create() called directly.  Please override")

    def step(self, feed_dict={}):
        return self._step(feed_dict)

    def _step(self, feed_dict={}):
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)

    def g_vars(self):
        return self.generator.variables()
    def d_vars(self):
        return self.discriminator.variables()

    def trainable_vars(self):
        d_vars = list(set(self.d_vars()).intersection(tf.trainable_variables()))
        g_vars = list(set(self.g_vars()).intersection(tf.trainable_variables()))
        return d_vars, g_vars

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
                self.optimistic_restore(self.session, save_file, self.variables())
                return True
            else:
                return False
        else:
            return False

    def optimistic_restore(self, session, save_file, variables):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables
                if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        post_restore_vars = []
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], variables), variables))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if saved_shapes[saved_var_name] is None:
                    print(" (load) No variable found, weights discarded", saved_var_name)
                if saved_shapes[saved_var_name] != var_shape:
                    #print(" (load) Shapes do not match, weights discarded", saved_var_name, var_shape, " vs loaded ", saved_shapes[saved_var_name])
                    print(" (load) Shapes do not match, extra reinitialized", saved_var_name, var_shape, " vs loaded ", saved_shapes[saved_var_name], curr_var)
                    saved_var = tf.zeros(saved_shapes[saved_var_name])
                    s1 = self.ops.shape(curr_var)
                    s2 = saved_shapes[saved_var_name]
                    new_var = saved_var

                    for i, (_s1, _s2) in enumerate(zip(s1, s2)):
                        if _s1 > _s2:
                            s3 = self.ops.shape(new_var)
                            ns = [i for i in s3]
                            ns[i] = s1[i] - s2[i]

                            curr_var_remainder = tf.slice(curr_var, [0 for i in s1], ns)
                            new_var = tf.concat([new_var, curr_var_remainder], axis=i)

                        elif _s2 > _s1:
                            ns = [-1 for i in s1]
                            ns[i] = _s1

                            new_var = tf.slice(curr_var, [0 for i in s1], ns)

                    post_restore_op = tf.assign(curr_var, new_var)
                    post_restore_vars.append([post_restore_op, saved_var, reader.get_tensor(saved_var_name)])

                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

        for op, var, val in post_restore_vars:
            self.gan.session.run(op, {var: val})

    def variables(self):
        return list(set(self.ops.variables() + sum([c.variables() for c in self.components], [])))

    def weights(self):
        return self.ops.weights + sum([c.ops.weights for c in self.components], [])

    def inputs(self):
        """inputs() returns any input tensors"""
        return sum([x.inputs() for x in self.components],[])


    def metrics(self):
        metrics = {}
        for metric in self._metrics:
            metrics[metric['name']]=metric['value']
        for c in self.components:
            try:
                metrics.update(c.metrics())
            except AttributeError:
                pass
        return metrics

    def configurable_param(self, string):
        if isinstance(string, str):
            name, *args = string.split(" ")
            return hg.ops.decay(self, *args)
        return string

