import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.skip_connections import SkipConnections

import re
import os
import inspect
import hypergan as hg
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.progressive_sampler import ProgressiveSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.batch_walk_sampler import BatchWalkSampler
from hypergan.samplers.grid_sampler import GridSampler
from hypergan.samplers.sorted_sampler import SortedSampler
from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.autoencode_sampler import AutoencodeSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.style_walk_sampler import StyleWalkSampler
from hypergan.samplers.alphagan_random_walk_sampler import AlphaganRandomWalkSampler
from hypergan.samplers.debug_sampler import DebugSampler
from hypergan.samplers.segment_sampler import SegmentSampler
from hypergan.samplers.y_sampler import YSampler
from hypergan.samplers.gang_sampler import GangSampler

class BaseGAN(GANComponent):
    def __init__(self, config=None, inputs=None, device='/gpu:0', ops_config=None, ops_backend=TensorflowOps, graph=None,
            batch_size=None, width=None, height=None, channels=None, debug=None, session=None, name="hypergan"):
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
        self.name = name
        self.session = session
        self.skip_connections = SkipConnections()
        self.destroy = False
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

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
                self.session = self.session or tf.Session(config=tfconfig, graph=graph)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.steps = tf.Variable(0, trainable=False, name='global_step')
        self.increment_step = tf.assign(self.steps, self.steps+1)
        if config.fixed_input:
            self.feed_x = self.inputs.x
            self.inputs.x = tf.Variable(tf.zeros_like(self.feed_x))
            self.set_x = tf.assign(self.inputs.x, self.feed_x)

        if config.fixed_input_xa:
            self.feed_x = self.inputs.xa
            self.inputs.xa = tf.Variable(tf.zeros_like(self.feed_x))
            self.set_x = tf.assign(self.inputs.xa, self.feed_x)
            self.feed_x = self.inputs.xb
            self.inputs.xb = tf.Variable(tf.zeros_like(self.feed_x))
            self.set_x = tf.group([self.set_x, tf.assign(self.inputs.xb, self.feed_x)])
            self.inputs.x = self.inputs.xb



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

    def create_loss(self, discriminator, reuse=False, split=2):
        loss = self.create_component(self.config.loss, discriminator = discriminator, split=split, reuse=reuse)
        return loss
    def create_generator(self, _input, reuse=False):
        return self.gan.create_component(self.gan.config.generator, name='generator', input=_input, reuse=reuse)

    def create_discriminator(self, _input, reuse=False):
        return self.gan.create_component(self.gan.config.discriminator, name="discriminator", input=_input, reuse=True)

    def create(self):
        print("Warning: BaseGAN.create() called directly.  Please override")

    def step(self, feed_dict={}):
        self.step_count = self.session.run(self.increment_step)
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
        return self.trainable_d_vars(), self.trainable_g_vars()

    def trainable_d_vars(self):
        return list(set(self.d_vars()).intersection(tf.trainable_variables()))

    def trainable_g_vars(self):
        return list(set(self.g_vars()).intersection(tf.trainable_variables()))

    def save(self, save_file):
        if(np.any(np.isnan(self.session.run(self.loss.d_fake)))):
            print("[Error] NAN detected.  Refusing to save")
            exit()

        with self.graph.as_default():
            print("[hypergan] Saving network to ", save_file)
            os.makedirs(os.path.expanduser(os.path.dirname(save_file)), exist_ok=True)
            saver = tf.train.Saver(self.variables())
            print("Saving " +str(len(self.variables()))+ " variables: ")
            missing = set(tf.global_variables()) - set(self.variables())
            missing = [ o for o in missing if "dontsave" not in o.name ]
            if(len(missing) > 0):
                print("[hypergan] Warning: Variables on graph but not saved:", missing)
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
        return list(set(self.ops.variables() + sum([c.variables() for c in self.components], []))) + [self.global_step, self.steps]

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

    def layer_options(self, l):
        for component in self.components:
            if hasattr(component, "layer_options"):
                if l in component.layer_options:
                    return component.layer_options[l]
        return None

    def configurable_param(self, string):
        self.param_ops = {
            "decay": self.configurable_params_decay,
            "on": self.configurable_params_turn_on
        }
        if isinstance(string, str):
            if re.match("^\d+$", string):
                return int(string)
            if re.match("^\d+?\.\d+?$", string):
                return float(string)
            if "(" not in string:
                return string

            method_name, inner = string.split("(")
            inner = inner.replace(")", "")
            if method_name not in self.param_ops:
                raise ValidationException("configurable param cannot find method: "+ method_name + " in string "+string)
            args, options = self.parse_args(inner.split(" "))
            result = self.param_ops[method_name](args, options)
            if "metric" in options:
                self.add_metric(options["metric"], result)
            return result
        return string

    def parse_args(self, strs):
        options = hc.Config({})
        args = []
        for x in strs:
            if '=' in x:
                lhs, rhs = x.split('=')
                options[lhs]=rhs
            else:
                args.append(x)
        return args, options

    def configurable_params_decay(self, args, options):
        _range = options.range or "0:1"
        steps = int(options.steps or 10000)
        start = int(options.start or 0)
        r1,r2 = _range.split(":")
        r1 = float(r1)
        r2 = float(r2)
        cycle = "cycle" in args
        repeat = "repeat" in args
        current_step = self.gan.steps
        if repeat:
            current_step %= steps
        if start == 0:
            return tf.train.polynomial_decay(r1, current_step, steps, end_learning_rate=r2, power=1, cycle=cycle)
        else:
            start = tf.constant(start, dtype=tf.int32)
            steps = tf.constant(steps, dtype=tf.int32)
            onoff = tf.minimum(1.0, tf.cast(tf.nn.relu(current_step - start), tf.float32))
            return (1.0 - onoff)*r1 + onoff * tf.train.polynomial_decay(r1, tf.to_float(current_step-start), tf.to_float(steps), end_learning_rate=r2, power=1.0, cycle=cycle)

    def configurable_params_turn_on(self, args, options):
        offset = float(options["offset"]) or 0.0
        if "random" in args:
            onvalue = float(options["onvalue"]) or 1.0
            n = tf.random_uniform([1], minval=-1, maxval=1)
            n += tf.constant(offset, dtype=tf.float32)
            return (tf.sign(n) + 1) /2 * tf.constant(float(options["onvalue"], dtype=tf.float32))


    def exit(self):
        self.destroy = True

    def build(self, input_nodes=None, output_nodes=None):
        if input_nodes is None:
            input_nodes = self.gan.input_nodes()
        if output_nodes is None:
            output_nodes = self.gan.output_nodes()
        save_file_text = self.name+".pbtxt"
        build_file = os.path.expanduser("builds/"+save_file_text)
        def create_path(filename):
            return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)
        create_path(build_file)
        tf.train.write_graph(self.gan.session.graph, 'builds', save_file_text)
        inputs = [x.name.split(":")[0] for x in input_nodes]
        outputs = [x.name.split(":")[0] for x in output_nodes]

        with self.gan.session as sess:
            converter = tf.lite.TFLiteConverter.from_session(sess, self.gan.input_nodes(), self.gan.output_nodes())
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            tflite_model = converter.convert()
            tflite_file = "builds/"+self.gan.name+".tflite"
            f = open(tflite_file, "wb")
            f.write(tflite_model)
            f.close()
        tf.reset_default_graph()
        self.gan.session.close()
        [print("Input: ", x) for x in self.gan.input_nodes()]
        [print("Output: ", y) for y in self.gan.output_nodes()]
        print("Written to "+tflite_file)


    def get_registered_samplers(self=None):
        return {
                'static_batch': StaticBatchSampler,
                'progressive': ProgressiveSampler,
                'random_walk': RandomWalkSampler,
                'alphagan_random_walk': AlphaganRandomWalkSampler,
                'style_walk': StyleWalkSampler,
                'batch_walk': BatchWalkSampler,
                'batch': BatchSampler,
                'grid': GridSampler,
                'sorted': SortedSampler,
                'gang': GangSampler,
                'began': BeganSampler,
                'autoencode': AutoencodeSampler,
                'debug': DebugSampler,
                'y': YSampler,
                'segment': SegmentSampler,
                'aligned': AlignedSampler
            }
    def sampler_for(self, name, default=StaticBatchSampler):
        samplers = self.get_registered_samplers()
        self.selected_sampler = name
        if name in samplers:
            return samplers[name]
        else:
            print("[hypergan] No sampler found for ", name, ".  Defaulting to", default)
            return default


