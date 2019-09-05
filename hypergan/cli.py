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
        #local_iterator = self.inputs.dataset.make_initializable_iterator()
        distributed_dataset = strategy.experimental_distribute_dataset(self.inputs.dataset)
        input_iterator = distributed_dataset.make_initializable_iterator()

        with strategy.scope():
            size = [int(x) for x in self.args.size.split("x")]
            #inp = hc.Config({"x": tf.slice(local_iterator.get_next(), [0,0,0,0], [self.args.batch_size//strategy.num_replicas_in_sync, -1, -1, -1])})#tf.zeros([self.args.batch_size/ strategy.num_replicas_in_sync, size[0], size[1], size[2]])})
            inp = hc.Config({"x": tf.constant(-1000.0, dtype=tf.float32, shape=[self.args.batch_size/ strategy.num_replicas_in_sync, size[0], size[1], size[2]])})
            self.gan = self.gan_fn(self.gan_config, inp, distribution_strategy=strategy)
            self.gan.cli = self

        def train_step(x):
            inp = hc.Config({"x": x})
            with tf.GradientTape(persistent=True) as tape:
                replica_gan = self.gan_fn(self.gan_config, inp, distribution_strategy=strategy, reuse=True)
                self.gan.replica = replica_gan
                d_loss = replica_gan.trainer.d_loss
                g_loss = replica_gan.trainer.g_loss
            d_grads = tape.gradient(d_loss, replica_gan.trainable_d_vars())
            g_grads = tape.gradient(g_loss, replica_gan.trainable_g_vars())
            #train_hook_grads = [t.gradient(tape) for t in replica_gan.trainer.train_hooks]

            del tape
            optimizer = tf.tpu.CrossShardOptimizer(self.gan.trainer.optimizer, reduction="weighted_sum")
            variables = replica_gan.trainable_d_vars() + replica_gan.trainable_g_vars()
            grads = d_grads + g_grads
            update_vars = optimizer.apply_gradients(
                            zip(grads, variables))
            update_train_hooks = [t.update_op() for t in replica_gan.trainer.train_hooks]
            update_train_hooks = [op for op in update_train_hooks if op is not None]
            if self.gan.config.alternating:
                update_vars_g = optimizer.apply_gradients(
                                zip(g_grads, replica_gan.trainable_g_vars()))
                update_vars_d = optimizer.apply_gradients(
                                zip(d_grads, replica_gan.trainable_d_vars()))
                with tf.control_dependencies([update_vars_d] + update_train_hooks):
                    with tf.control_dependencies([update_vars_g]):
                        return tf.identity(d_loss)
                        #train_hook_updates = [replica_gan.trainer.train_hooks[i].apply_gradients(optimizer, grad) for i,grad in enumerate(train_hook_grads) if grad is not None]
                        #train_hook_updates = [op for op in train_hook_updates if op is not None]
                        #print("Train hook update count: ", len(train_hook_updates))
                        #if len(train_hook_updates) == 0:
                        #    return tf.identity(d_loss)
                        #with tf.control_dependencies(train_hook_updates):
                        #    return tf.identity(d_loss)
            else:
                with tf.control_dependencies([update_vars] + update_train_hooks):
                    return tf.identity(d_loss)

        print("Creating replica graph")
        input_iterator_next = next(input_iterator)
        train = strategy.unwrap(strategy.experimental_run_v2(train_step, args=(input_iterator_next, )))
        train_hook_update_steps = []
        for t in self.gan.trainer.train_hooks:
            train_hook_update_steps += t.distributed_step(input_iterator_next)
        train_hook_update_steps = [strategy.unwrap(t) for t in train_hook_update_steps]
        print("TR", train_hook_update_steps)

        #update_train_hooks_for_each_replica = self.gan.distribution_strategy.group(train_hook_update_steps)
        update_train_hooks_for_each_replica = train_hook_update_steps
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
            #print("Initializing local iterator")
            #session.run([local_iterator.initializer])
            print("Initializing generator variables")
            g_ops = [v.initializer for v in self.gan.generator.variables()]
            session.run(g_ops)
            print("Initializing other variables")
            v_ops = list(set([v.initializer for v in self.gan.variables()])-set(g_ops))
            session.run(v_ops)
            print("Loading model, if available")
            if self.tpu_load(self.save_file):
                print("Model loaded")
            else:
                print("Initializing new model")
            init_steps = []
            for train_hook in self.gan.trainer.train_hooks:
                init_steps += train_hook.distributed_initial_step(input_iterator_next)
            init_steps = [strategy.unwrap(t) for t in init_steps]
            #init_steps = self.gan.distribution_strategy.group(init_steps)
            self.gan.session.run(init_steps)

            update_train_hooks = [t.update_op() for t in self.gan.trainer.train_hooks]
            update_train_hooks = tf.group(*[op for op in update_train_hooks if op is not None])
            while((i < self.total_steps or self.total_steps == -1)):
                if i % self.sample_every == 0:
                    self.gan.trainer.print_metrics(i)
                    self.sample()
                i+=1
                #for train_hook in self.gan.trainer.train_hooks:
                #    train_hook.before_step(i,{})
                #session.run(update_train_hooks)
                session.run(update_train_hooks_for_each_replica+ [train])

                if (self.args.save_every != None and
                    self.args.save_every != -1 and
                    self.args.save_every > 0 and
                    i % self.args.save_every == 0):
                    print(" |= Saving network")
                    self.tpu_save(self.save_file)  
            session.run(tpu.shutdown_system())

    def tpu_load(self, save_file):
        if "gs://" in save_file:
            return self.gan.load(save_file)
        self.lazy_cpu_gan_init()
        print(" |= Loading network from local filesystem")

        self.tpu_write_ops = {}
        self.tpu_placeholders = {}
        for v in self.gan.variables():
            with self.gan.distribution_strategy.scope():
                ph = tf.zeros_like(v)
                self.tpu_placeholders[v] = ph

            def assign(v, ph):
                return v.assign(ph)
            op = self.gan.distribution_strategy.extended.call_for_each_replica(assign, args=(v,ph,))
            self.tpu_write_ops[v] = self.gan.distribution_strategy.unwrap([op])
        with tf.Session(graph=self.cpu_gan.graph) as session:
            print(" |=> Copying weights from CPU to TPU")
            self.cpu_gan.session = session
            self.cpu_gan.load(save_file)

                    #self.tpu_write_ops[v] = self.gan.distribution_strategy.extended.update(v, args=(ph,))
            for tpu_weight in self.gan.variables():
                matching_cpu_weight = None
                for cpu_placeholder, cpu_assign, cpu_weight in zip(self.cpu_placeholders, self.cpu_assigns, self.cpu_gan.variables()):
                    if cpu_weight.name.replace("save/","") == tpu_weight.name:
                        matching_cpu_weight = cpu_weight
                tpu_placeholder = self.tpu_placeholders[v]
                tpu_write_op = self.tpu_write_ops[v]
                cpu_val = self.cpu_gan.session.run(matching_cpu_weight)
                session.run(tpu_write_op, {tpu_placeholder: cpu_val})


    def lazy_cpu_gan_init(self):
        if not hasattr(self, "cpu_gan"):
            self.tpu_read_ops = {}
            for v in self.gan.variables():
                self.tpu_read_ops[v] = self.gan.distribution_strategy.extended.read_var(v)
            with tf.device('/cpu:0'):
                with tf.variable_scope("save", reuse=False) as scope:
                    size = [int(x) for x in self.args.size.split("x")]
                    graph = tf.Graph()
                    with graph.as_default():
                        inp = hc.Config({"x": tf.constant(-1000.0, dtype=tf.float32, shape=[self.args.batch_size/ self.gan.distribution_strategy.num_replicas_in_sync, size[0], size[1], size[2]])})
                        self.cpu_gan = self.gan_fn(self.gan_config, inp, graph=graph)
                        self.cpu_placeholders = [tf.zeros_like(v) for v in self.cpu_gan.variables()]
                        self.cpu_assigns = [tf.assign(v, placeholder) for v, placeholder in zip(self.cpu_gan.variables(), self.cpu_placeholders)]
            with tf.Session(graph=self.cpu_gan.graph) as session:
                self.cpu_gan.session = session
                self.cpu_gan.initialize_variables()
                for train_hook in self.cpu_gan.trainer.train_hooks:
                    train_hook.before_step(0,{})

    def tpu_save(self, save_file):
        if "gs://" in save_file:
            return self.gan.save(save_file)
        self.lazy_cpu_gan_init()

        with tf.Session(graph=self.cpu_gan.graph) as session:
            print(" |=> Copying weights from TPU to CPU")
            self.cpu_gan.session = session
            for tpu_weight in self.gan.variables():
                tpu_read_op = self.tpu_read_ops[tpu_weight]
                matching_cpu_weight = None
                matching_cpu_assign = None
                matching_cpu_placeholder = None
                for cpu_placeholder, cpu_assign, cpu_weight in zip(self.cpu_placeholders, self.cpu_assigns, self.cpu_gan.variables()):
                    if cpu_weight.name.replace("save/","") == tpu_weight.name:
                        matching_cpu_weight = cpu_weight
                        matching_cpu_assign = cpu_assign
                        matching_cpu_placeholder = cpu_placeholder
                tpu_val = self.gan.session.run(tpu_read_op)
                session.run(matching_cpu_assign, {matching_cpu_placeholder: tpu_val})
            print(" |=> Saving CPU")
            self.cpu_gan.save(save_file)


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
        if self.args.toml:
            config_format = '.toml'
        else:
            config_format = '.json'
        template = self.args.directory + config_format
        print("[hypergan] Creating new configuration file '"+template+"' based off of '"+self.config_name+config_format)
        if os.path.isfile(template):
            raise ValidationException("File exists: " + template)
        source_configuration = Configuration.find(self.config_name+config_format, config_format=config_format)
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
        GlobalViewer.close()

