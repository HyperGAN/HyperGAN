"""
The command line interface.  Trains a directory of data.
"""
import gc
import sys
import os
import hyperchamber as hc
import tensorflow as tf
import numpy as np
from hypergan.gan_component import ValidationException
from .inputs import *
from .viewer import GlobalViewer
from .configuration import Configuration
from tensorflow.contrib import tpu
import hypergan as hg
import time

import os
import shutil
import sys

from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.multi_component import MultiComponent
from time import sleep


class CLI:
    def __init__(self, args={}, gan_fn=None, inputs_fn=None, gan_config=None):
        self.samples = 0
        self.steps = 0
        self.gan_fn = gan_fn
        self.gan_config = gan_config
        self.inputs_fn = inputs_fn

        args = hc.Config(args)
        self.args = args

        crop =  self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler_name = args.sampler
        self.sampler = None
        
        self.validate()
        
        self.loss_every = self.args.loss_every or 1
        
        if (self.args.save_losses):
            import matplotlib.pyplot as plt
            self.arr = []
            self.fig,self.ax = plt.subplots()
            self.temp = 0

        self.advSavePath = os.path.abspath("saves/"+self.config_name)+"/"
        if self.args.save_file:
            self.save_file = self.args.save_file
        else:
            default_save_path = os.path.abspath("saves/"+self.config_name)
            self.save_file = default_save_path + "/model.ckpt"
            self.create_path(self.save_file)

        title = "[hypergan] " + self.config_name
        GlobalViewer.set_options(
            enable_menu = self.args.menu,
            title = title,
            viewer_size = self.args.viewer_size,
            enabled = self.args.viewer,
            zoom = self.args.zoom)

    def sample(self, allow_save=True):
        """ Samples to a file.  Useful for visualizing the learning process.

        If allow_save is False then saves will not be created.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """
        sample_file="samples/%s/%06d.png" % (self.config_name, self.samples)
        self.create_path(sample_file)
        self.lazy_create()
        sample_list = self.sampler.sample(sample_file, allow_save and self.args.save_samples)
        self.samples += 1

        return sample_list

    def validate(self):
        return True

    def lazy_create(self):
        if(self.sampler == None):
            self.sampler = self.gan.sampler_for(self.sampler_name)(self.gan, samples_per_row=self.args.width)
            if(self.sampler == None):
                raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        bgan = self.gan
        self.gan.step()
        if hasattr(self.gan, 'newgan') and bgan.destroy:
            self.sampler=None
            self.gan = self.gan.newgan
            gc.collect()
            refs = gc.get_referrers(bgan)
            d = bgan.trainer._delegate
            bgan.trainer=None
            gc.collect()
            del bgan
            tf.reset_default_graph()

            gc.collect()

        if(self.steps % self.sample_every == 0 and self.args.sampler):
            sample_list = self.sample()

        self.steps+=1

        x = []
        if(hasattr(self.gan.loss,"sample")):
            loss = self.gan.loss.sample
            if(self.args.save_losses):
                temp2 = False
                if(len(self.arr)==0):
                    for i in range(0,len(loss)):
                        self.arr.append([]);
                for i in range(0,len(loss)):
                    self.arr[i].append(self.gan.session.run(loss[i]))
                for j in range(0,len(self.arr)):
                    if (len(self.arr[j]) > 100):
                        self.arr[j].pop(0)
                        if(not temp2 == True):
                            self.temp += 1
                            temp2 = True
                if(temp2 == True):
                    temp2 = False
        else:
            if(self.args.save_losses):
                temp2 = False
                if(len(self.arr)==0):
                    for i in range(0,len(self.gan.trainer.losses)):
                        self.arr.append([]);
                for i in range(0,len(self.gan.trainer.losses)):
                    self.arr[i].append(self.gan.session.run(self.gan.trainer.losses[i][1]))
                for j in range(0,len(self.arr)):
                    if (len(self.arr[j]) > 100):
                        self.arr[j].pop(0)
                        if(not temp2 == True):
                            self.temp += 1
                            temp2 = True
                if(temp2 == True):
                    temp2 = False
        if(self.args.save_losses and self.steps % self.loss_every == 0):
            for i in range(0,len(self.arr)):
                x2 = []
                for j in range(self.temp,self.temp+len(self.arr[i])):
                    x2.append(j)
                x.append(x2)
            self.ax.cla()
            for i in range(0,len(self.arr)):
                self.ax.plot(x[i], self.arr[i])
            self.ax.grid()
            self.ax.title.set_text("HyperGAN losses")
            self.ax.set_xlabel('Steps')
            self.ax.set_ylabel('Losses')
            self.create_path("losses/"+self.config_name+"/%06d.png" % (self.steps))
            self.fig.savefig("losses/"+self.config_name+"/%06d.png" % (self.steps))

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self):
        return self.gan.build()
    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        while not self.gan.destroy:
            self.sample()

    def train_tpu(self):
        i=0
        tf.disable_v2_behavior()
        tpu_name = self.args.device.replace("/tpu:", "")
        print("Connecting to TPU:", tpu_name)
        cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        self.cluster_resolver = cluster_resolver
        tf.tpu.experimental.initialize_tpu_system(self.cluster_resolver)
        strategy = tf.contrib.distribute.TPUStrategy(self.cluster_resolver)
        strategy.extended.experimental_enable_get_next_as_optional = False

        self.inputs = self.inputs_fn()
        input_iterator = strategy.experimental_distribute_dataset(self.inputs.dataset).make_initializable_iterator()

        with strategy.scope():
            size = [int(x) for x in self.args.size.split("x")]
            inp = hc.Config({"x": tf.zeros([self.args.batch_size/ strategy.num_replicas_in_sync, size[0], size[1], size[2]])})
            self.gan = self.gan_fn(self.gan_config, inp, distribution_strategy=strategy)
            self.gan.cli = self

        def train_step(x):
            inp = hc.Config({"x": x})
            with tf.GradientTape(persistent=True) as tape:
                replica_gan = self.gan_fn(self.gan_config, inp, distribution_strategy=strategy, reuse=True)
                d_loss = replica_gan.trainer.d_loss
                g_loss = replica_gan.trainer.g_loss
                if replica_gan.config.skip_gradient_mean is None:
                    d_loss = d_loss / strategy.num_replicas_in_sync
                    g_loss = g_loss / strategy.num_replicas_in_sync
            d_grads = tape.gradient(d_loss, replica_gan.trainable_d_vars())
            g_grads = tape.gradient(g_loss, replica_gan.trainable_g_vars())

            del tape
            optimizer = tf.tpu.CrossShardOptimizer(self.gan.trainer.optimizer)
            variables = replica_gan.trainable_d_vars() + replica_gan.trainable_g_vars()
            grads = d_grads + g_grads
            update_vars = optimizer.apply_gradients(
                            zip(grads, variables))
            with tf.control_dependencies([update_vars]):
                return tf.identity(d_loss)

        print("Creating replica graph")
        train = strategy.unwrap(strategy.experimental_run_v2(train_step, args=(next(input_iterator), )))
        print("Initializing TPU")
        iterator_init = input_iterator.initialize()

        config = tf.ConfigProto()
        cluster_spec = cluster_resolver.cluster_spec()
        if cluster_spec:
          config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

        with tf.Session(cluster_resolver.master(), config=config) as session:

            self.gan.session = session
            print("Initializing iterator")
            session.run([iterator_init])
            print("Initializing Variables")
            v_ops = [v.initializer for v in self.gan.variables()]
            session.run(v_ops)
            for v in self.gan.variables():
                session.run(v.initializer)
            print("Train hook step 0")
            for train_hook in self.gan.trainer.train_hooks:
                train_hook.before_step(0,{})
            print("Loading model, if available")
            if self.gan.load(self.save_file):
                print("Model loaded")
            else:
                print("Initializing new model")
            while((i < self.total_steps or self.total_steps == -1)):
                if i % 10 == 0:
                    self.gan.trainer.print_metrics(i)
                if i % 100 == 0:
                    self.sample()
                i+=1
                session.run(train)

                if (self.args.save_every != None and
                    self.args.save_every != -1 and
                    self.args.save_every > 0 and
                    i % self.args.save_every == 0):
                    print(" |= Saving network")
                    self.gan.save(self.save_file)  
            session.run(tpu.shutdown_system())

    def train(self):
        i=0
        if(self.args.ipython):
            import fcntl
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        if "tpu" in self.args.device:
            self.train_tpu()
            return

        self.inputs = self.inputs_fn()
        self.gan = self.gan_fn(self.gan_config, self.inputs)
        self.gan.cli = self
        self.gan.initialize_variables()
        if self.gan.load(self.save_file):
            print("Model loaded")
        else:
            print("Initializing new model")

        while((i < self.total_steps or self.total_steps == -1) and not self.gan.destroy):
            i+=1
            start_time = time.time()
            self.step()

            if (self.args.save_every != None and
                self.args.save_every != -1 and
                self.args.save_every > 0 and
                i % self.args.save_every == 0):
                print(" |= Saving network")
                self.gan.save(self.save_file)   
                self.create_path(self.advSavePath+'advSave.txt')
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'w') as the_file:
                        the_file.write(str(self.steps)+"\n")
                        the_file.write(str(self.samples)+"\n")
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
            #EWW
        else:
            print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

    def run(self):
        if self.method == 'train':
            self.train()
        elif self.method == 'build':
            self.inputs = self.inputs_fn()
            self.gan = self.gan_fn(self.gan_config, self.inputs)
            if not self.gan.load(self.save_file):
                raise ValidationException("Could not load model: "+ self.save_file)
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.steps = int(content[0])
                        self.samples = int(content[1])
                print("Model loaded")
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.inputs = self.inputs_fn()
            self.gan = self.gan_fn(self.gan_config, self.inputs)
            self.gan.initialize_variables()
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.steps = int(content[0])
                        self.samples = int(content[1])
                print("Model loaded")

            tf.train.start_queue_runners(sess=self.gan.session)
            self.sample_forever()
            tf.reset_default_graph()
            self.gan.session.close()
        elif self.method == 'test':
            print("Hooray!")
            print("Hypergan is installed correctly.  Testing tensorflow for GPU support.")
            with tf.Session() as sess:
                devices = sess.list_devices()

            if not tf.test.gpu_device_name():
                print("Warning: no default GPU device available")
                allgood=False
            else:
                print("Default GPU is available")
                allgood=True
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            print("Current available tensorflow devices:")
            for device in devices:
                print(device)
            if allgood:
                print("Congratulations!  Tensorflow and hypergan both look installed correctly.  If you still experience issues come let us know on discord.")
            else:
                print("There were errors in the test, please see the logs")

