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


def train(device, gan, inputs, done_event):
    gan.inputs = inputs
    #torch.manual_seed(device)
    done_event.set()
    gan.trainer = gan.create_component("trainer")
    while(True):
        gan.step()

class HogwildBackend(Backend):
    """https://towardsdatascience.com/this-is-hogwild-7cc80cd9b944"""
    def __init__(self, gan, cli, devices):
        self.gan = gan
        self.processes = []
        done_events = []
        mp.set_start_method('spawn', force=True)
        for component in gan.generator_components() + gan.discriminator_components():
            component.share_memory()
        if devices == "-1":
            print("Running on all available devices: ", torch.cuda.device_count())
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(d) for d in devices.split(",")]
        print("Devices:", devices)
        #inputs = cli.create_input(rank=devices[0])
        #gan.inputs = inputs
        for device in devices:
            done_event = mp.Event()
            inputs = cli.create_input(rank=device)
            p = mp.Process(target=train, args=(device, gan, inputs, done_event))
            p.start()
            self.processes.append(p)
            done_event.wait()
    def step(self):
        time.sleep(0.1)
