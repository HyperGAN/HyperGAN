import os
import hyperchamber as hc
import tensorflow as tf
from hypergan.gan_component import ValidationException
from . import GAN
from .inputs import *
from .viewer import GlobalViewer
from .configuration import Configuration
import hypergan as hg
import time

import fcntl
import os
import shutil
import sys

from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.grid_sampler import GridSampler
from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.autoencode_sampler import AutoencodeSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.alphagan_random_walk_sampler import AlphaganRandomWalkSampler
from hypergan.samplers.debug_sampler import DebugSampler

from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.multi_component import MultiComponent
from time import sleep

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

class CLI:
    def __init__(self, gan, args={}):
        self.samples = 0
        self.steps = 0
        self.gan = gan

        args = hc.Config(args)
        self.args = args

        crop =  self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler = CLI.sampler_for(args.sampler)(self.gan)

        self.validate()
        if self.args.save_file:
            self.save_file = self.args.save_file
        else:
            default_save_path = os.path.abspath("saves/"+self.config_name)
            self.save_file = default_save_path + "/model.ckpt"
            self.create_path(self.save_file)

        title = "[hypergan] " + self.config_name
        GlobalViewer.title = title
        GlobalViewer.enabled = self.args.viewer

    def sampler_for(name, default=StaticBatchSampler):
        samplers = {
                'static_batch': StaticBatchSampler,
                'random_walk': RandomWalkSampler,
                'alphagan_random_walk': AlphaganRandomWalkSampler,
                'batch': BatchSampler,
                'grid': GridSampler,
                'began': BeganSampler,
                'autoencode': AutoencodeSampler,
                'debug': DebugSampler,
                'aligned': AlignedSampler
        }
        if name in samplers:
            return samplers[name]
        else:
            print("[hypergan] No sampler found for ", name, ".  Defaulting to", default)
            return default

    def sample(self, sample_file):
        """ Samples to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        sample_list = self.sampler.sample(sample_file, self.args.save_samples)

        return sample_list


    def validate(self):
        if(self.sampler == None):
            raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        self.gan.step()

        if(self.steps % self.sample_every == 0):
            sample_file="samples/%06d.png" % (self.samples)
            self.create_path(sample_file)
            sample_list = self.sample(sample_file)
            if self.args.use_hc_io:
                self.gan.config['model'] = self.args.config
                hc.io.sample(self.gan.config, sample_list)

            self.samples += 1

        self.steps+=1

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self):
        save_file_text = self.args.config+".pbtxt"
        build_file = os.path.expanduser("builds/"+save_file_text)
        self.create_path(build_file)
        tf.train.write_graph(self.gan.session.graph, 'builds', save_file_text)
        inputs = [x.name.split(":")[0] for x in self.gan.input_nodes()]
        outputs = [x.name.split(":")[0] for x in self.gan.output_nodes()]
        print("___")
        print(inputs, outputs)
        tf.reset_default_graph()
        self.gan.session.close()
        [print("Input: ", x) for x in self.gan.input_nodes()]
        [print("Output: ", y) for y in self.gan.output_nodes()]

        pbtxt_path = "builds/"+self.args.config+'.pbtxt'
        checkpoint_path = "saves/"+self.args.config+'/model.ckpt'
        input_saver_def_path = ""
        input_binary = False
        output_node_names = ",".join(outputs)
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = 'builds/frozen_'+self.args.config+'.pb'
        output_optimized_graph_name = 'builds/optimized_'+self.args.config+'.pb'
        clear_devices = True

        freeze_graph.freeze_graph(pbtxt_path, input_saver_def_path,
          input_binary, checkpoint_path, output_node_names,
          restore_op_name, filename_tensor_name,
          output_frozen_graph_name, clear_devices, "")

        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)

        print("GRAPH INPUTS", inputs, "OUTPUTS", outputs)
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def,
                inputs, # an array of the input node(s)
                outputs, # an array of output nodes
                tf.float32.as_datatype_enum)

        # Save the optimized graph

        f = tf.gfile.FastGFile(output_optimized_graph_name, "wb")
        f.write(output_graph_def.SerializeToString())
        f.flush()
        f.close()

        print("Saved generator to ", output_optimized_graph_name)

        print("Testing loading ", output_optimized_graph_name)
        with tf.gfile.FastGFile(output_optimized_graph_name, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            for input in inputs:
                print("Input: ", input, sess.graph.get_tensor_by_name(input+":0"))
            for output in outputs:
                print("Output: ", output, sess.graph.get_tensor_by_name(output+":0"))

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        while True:
            sample_file="samples/%06d.png" % (self.samples)
            self.create_path(sample_file)
            self.sample(sample_file)
            self.samples += 1
            print("Sample", self.samples)
            sleep(0.2)


    def train(self):
        i=0
        if(self.args.ipython):
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        while(i < self.total_steps or self.total_steps == -1):
            i+=1
            start_time = time.time()
            self.step()

            if (self.args.save_every != None and
                self.args.save_every != -1 and
                self.args.save_every > 0 and
                i % self.args.save_every == 0):
                print(" |= Saving network")
                self.gan.save(self.save_file)
            if self.args.ipython:
                self.check_stdin()
            end_time = time.time()

    def check_stdin(self):
        try:
            input = sys.stdin.read()
            if input[0]=="y":
                return
            from IPython import embed
            # Misc code
            embed()

        except:
            return

    def new(self):
        template = self.args.directory + '.json'
        print("[hypergan] Creating new configuration file '"+template+"' based off of '"+self.config_name+".json'")
        if os.path.isfile(template):
            raise ValidationException("File exists: " + template)
        source_configuration = Configuration.find(self.config_name+".json")
        shutil.copyfile(source_configuration, template)

        return

    def add_supervised_loss(self):
        if self.args.classloss:
            print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
            supervised_loss = SupervisedLoss(self.gan, self.gan.config.loss)
            self.gan.loss = MultiComponent(components=[supervised_loss, self.gan.loss], combine='add')
            supervised_loss.create()
            #EWW
        else:
            print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

    def run(self):
        if self.method == 'train':
            self.gan.create()
            self.add_supervised_loss()
            self.gan.session.run(tf.global_variables_initializer())

            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                print("Model loaded")
            tf.train.start_queue_runners(sess=self.gan.session)
            self.train()
            tf.reset_default_graph()
            self.gan.session.close()
        elif self.method == 'build':
            self.gan.create()
            if not self.gan.load(self.save_file):
                raise "Could not load model: "+ save_file
            else:
                print("Model loaded")
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.gan.create()
            self.add_supervised_loss()
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                print("Model loaded")

            tf.train.start_queue_runners(sess=self.gan.session)
            self.sample_forever()
            tf.reset_default_graph()
            self.gan.session.close()
