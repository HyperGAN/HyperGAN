from .backend import Backend
import torch.multiprocessing as mp
from hypergan.gan_component import ValidationException, GANComponent
import torch.utils.data as data
import hyperchamber as hc
import hypergan as hg
import copy
import torch
import time

def create_input(input_config):
    klass = GANComponent.lookup_function(None, input_config['class'])
    return klass(input_config)

def train(device, gan, save_file, inputs, done_event):

    gan.inputs = inputs
    from hypergan.trainable_gan import TrainableGAN
    trainable_gan = TrainableGAN(gan, backend_name = "single-gpu", save_file = save_file)
    #torch.manual_seed(device)
    done_event.set()
    while(True):
        trainable_gan.step()

class HogwildBackend(Backend):
    """https://towardsdatascience.com/this-is-hogwild-7cc80cd9b944"""
    def __init__(self, trainable_gan, devices):
        self.trainable_gan = trainable_gan
        self.processes = []
        done_events = []
        mp.set_start_method('spawn', force=True)
        for component in trainable_gan.gan.generator_components() + trainable_gan.gan.discriminator_components():
            component.share_memory()
        if devices == "-1":
            print("Running on all available devices: ", torch.cuda.device_count())
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(d) for d in devices.split(",")]
        print("Devices:", devices)
        #gan.inputs = inputs
        num_processes=4
        for device in range(num_processes):
            done_event = mp.Event()
            inputs = self.trainable_gan.gan.inputs.to(self.trainable_gan.gan.device)
            p = mp.Process(target=train, args=(device, trainable_gan.gan, trainable_gan.save_file, inputs, done_event))
            p.start()
            self.processes.append(p)
            done_event.wait()

    def save(self):
        self.trainable_gan.save_locally()

    def step(self):
        time.sleep(0.1)
