import os
from hypergan.backends.hogwild_backend import HogwildBackend
from hypergan.backends.roundrobin_backend import RoundrobinBackend
from hypergan.backends.single_gpu_backend import SingleGPUBackend
from hypergan.backends.cpu_backend import CPUBackend
from hypergan.train_hook_collection import TrainHookCollection

class TrainableGAN:
    def __init__(self, gan, save_file = "default.save", devices = [], backend_name = "roundrobin"):
        self.gan = gan
        self.optimizers = []
        self.samples = 0
        self.save_file = save_file

        self._create_optimizer = True
        if backend_name == "roundrobin":
            self._create_optimizer = False

        self.train_hooks = TrainHookCollection(self)
        self.backend_name = backend_name
        self.available_backends = {
            'roundrobin': RoundrobinBackend,
            'single-gpu': SingleGPUBackend,
            'cpu': CPUBackend,
            'hogwild': HogwildBackend
        }

        chosen_backend = self.available_backends[backend_name]
        self.backend = chosen_backend(self, devices=devices)

        self.loss = self.gan.initialize_component("loss")
        self.trainer = self.gan.initialize_component("trainer", self)

    def add_optimizer(self, optimizer):
        if self._create_optimizer:
            self.optimizers.append(optimizer)

    def create_optimizer(self, klass, defn):
        optimizer = klass(self.parameters(), **defn)
        self.add_optimizer(optimizer)
        return optimizer

    def forward_loss(self):
        """
            Runs a forward pass through the GAN and returns (d_loss, g_loss)
        """
        return self.gan.forward_loss(self.loss)

    def step(self):
        self.backend.step()
        self.gan.steps += 1
        self._metrics = {}

    def load(self):
        success = self.gan.load(self.save_file)
        full_path = os.path.expanduser(os.path.dirname(self.save_file))
        for i, optimizer in enumerate(self.optimizers):
            self.gan._load(full_path, "optimizer"+str(i), optimizer)
        return success

    def save(self):
        self.backend.save()

    def save_locally(self):
        full_path = os.path.expanduser(os.path.dirname(self.save_file))
        os.makedirs(full_path, exist_ok=True)
        self.gan.save(full_path)
        for i, optimizer in enumerate(self.optimizers):
            self.gan._save(full_path, "optimizer"+str(i), optimizer)

    def set_generator_trainable(self, flag):
        for c in self.gan.generator_components():
            c.set_trainable(flag)
        for train_hook in self.gan.hooks:
            for c in train_hook.generator_components():
                c.set_trainable(flag)

    def set_discriminator_trainable(self, flag):
        for c in self.gan.discriminator_components():
            c.set_trainable(flag)
        for train_hook in self.gan.hooks:
            for c in train_hook.discriminator_components():
                c.set_trainable(flag)

    def to(self, device):
        self.gan.to(device)

    def train_hooks(self):
        result = []
        for component in self.gan.components:
            if hasattr(component, "train_hooks"):
                result += component.train_hooks
        return result

    def g_parameters(self):
        #TODO add optimizer params
        for component in self.gan.generator_components():
            for param in component.parameters():
                yield param

        for train_hook in self.gan.hooks:
            for component in train_hook.generator_components():
                for param in component.parameters():
                    yield param


    def d_parameters(self):
        #TODO add optimizer params
        for component in self.gan.discriminator_components():
            for param in component.parameters():
                yield param

        for train_hook in self.gan.hooks:
            for component in train_hook.discriminator_components():
                for param in component.parameters():
                    yield param


    def parameters(self):
        for param in self.g_parameters():
            yield param
        for param in self.d_parameters():
            yield param

    def sample(self, sampler, sample_path, save_samples=True):
        """ Samples to a file.  Useful for visualizing the learning process.

        If allow_save is False then saves will not be created.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """
        os.makedirs(os.path.expanduser(sample_path), exist_ok=True)
        sample_file="%s/%06d.png" % (sample_path, self.samples)
        sample_list = sampler.sample(sample_file, save_samples)
        self.samples += 1

        return sample_list
