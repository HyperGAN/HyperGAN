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


def train(device, head_device, gan, inputs, loaded_event, report_weights_queue, set_weights_queue, report_weights_event):
    gan.inputs = inputs
    #torch.manual_seed(device)
    gan.generator = gan.generator.to(device)
    gan.generator.device="cuda:"+str(device)
    gan.discriminator = gan.discriminator.to(device)
    gan.discriminator.device="cuda:"+str(device)
    gan.inputs.device=device
    gan.latent.device=device
    gan.trainer = gan.create_component("trainer")
    loaded_event.set()
    while(True):
        gan.step()
        if report_weights_event.is_set():
            if device == head_device:
                report_weights_queue.put(list([p.clone().detach() for p in gan.parameters()]))
            else:
                report_weights_queue.put(list(gan.parameters()))

            params = set_weights_queue.get()
            for p1, p2 in zip(gan.parameters(), params):
                with torch.no_grad():
                    if(p2.device != p1.device):
                        p2 = p2.to(p1.device)
                    p1.set_(p2)
            del params
            report_weights_event.clear()


class RoundrobinBackend(Backend):
    """Trains separately then syncs each card, round robin style"""
    def __init__(self, gan, cli, devices):
        self.gan = gan
        self.cli = cli
        self.sync = 0
        self.processes = []
        self.report_weights_event = []
        self.report_weights_queue = []
        self.set_weights_queue = []
        loaded_events = []
        head_device = torch.device(list(self.gan.parameters())[0].device).index
        self.head_device=head_device
        mp.set_start_method('spawn', force=True)
        if devices == "-1":
            print("Running on all available devices: ", torch.cuda.device_count())
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(d) for d in devices.split(",")]
        self.devices = devices
        print("Devices:", devices)
        for device in devices:
            loaded_event = mp.Event()
            report_weights_event = mp.Event()
            set_weights_queue = mp.Queue()
            report_weights_queue = mp.Queue()
            inputs = cli.create_input(rank=device)
            p = mp.Process(target=train, args=(device, head_device, gan, inputs, loaded_event, report_weights_queue, set_weights_queue, report_weights_event))
            p.start()
            self.processes.append(p)
            self.report_weights_event.append(report_weights_event)
            self.report_weights_queue.append(report_weights_queue)
            self.set_weights_queue.append(set_weights_queue)
            loaded_event.wait()

    def step(self):
        time.sleep(1.0)
        selected = self.sync % len(self.processes)
        print("Syncing", selected)
        self.report_weights_event[selected].set()
        params = self.report_weights_queue[selected].get()
        for p1, p2 in zip(self.gan.parameters(), params):
            with torch.no_grad():
                if(p2.device != p1.device):
                    p2 = p2.to(p1.device)
                p1.set_((p1+p2)/2.0)
        del params
        if self.devices[selected] == self.head_device:
            self.set_weights_queue[selected].put(list([p.clone().detach() for p in self.gan.parameters()]))
        else:
            self.set_weights_queue[selected].put(list(self.gan.parameters()))
        self.sync += 1
        self.cli.sample()
