from .backend import Backend
import torch.multiprocessing as mp
from hypergan.gan_component import ValidationException, GANComponent
import torch.utils.data as data
import hyperchamber as hc
import hypergan as hg
import copy
import torch
import time
mp.set_start_method('spawn', force=True)

def train(device, head_device, gan, inputs, loaded_event, report_weights_queue, set_weights_queue, report_weights_event, save_event, save_complete_event, save_file):
    gan.inputs = inputs
    from hypergan.trainable_gan import TrainableGAN
    trainable_gan = TrainableGAN(gan, backend_name = "single-gpu", save_file = save_file)
    trainable_gan.to(device)
    if trainable_gan.load():
        print("Model loaded")
    else:
        print("Initializing new model")

    loaded_event.set()
    while(True):
        trainable_gan.step()
        if report_weights_event.is_set():
            if device == head_device:
                report_weights_queue.put(list([p.clone().detach() for p in trainable_gan.parameters()]))
            else:
                report_weights_queue.put(list(trainable_gan.parameters()))
            params = set_weights_queue.get()
            for p1, p2 in zip(trainable_gan.parameters(), params):
                with torch.no_grad():
                    if(p2.device != p1.device):
                        p2 = p2.to(p1.device)
                    p1.set_(p2)
            del params
            report_weights_event.clear()
        if save_event is not None and save_event.is_set():
            print("Saving from roundrobin")
            trainable_gan.save_locally()
            save_event.clear()
            save_complete_event.set()

class RoundrobinBackend(Backend):
    """Trains separately then syncs each card, round robin style"""
    def __init__(self, trainable_gan, devices):
        self.trainable_gan = trainable_gan
        self.sync = 0
        self.processes = []
        self.report_weights_event = []
        self.report_weights_queue = []
        self.set_weights_queue = []
        loaded_events = []
        head_device = torch.device(list(self.trainable_gan.parameters())[0].device).index
        self.head_device=head_device
        if devices == "-1":
            print("Running on all available devices: ", torch.cuda.device_count())
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(d) for d in devices.split(",")]
        self.devices = devices
        print("Devices:", devices)
        save_complete_event = mp.Event()
        save_event = mp.Event()
        self.save_event = save_event
        self.save_complete_event = save_complete_event

        for device in devices:
            loaded_event = mp.Event()
            report_weights_event = mp.Event()
            set_weights_queue = mp.Queue()
            report_weights_queue = mp.Queue()
            inputs = self.trainable_gan.gan.inputs.to(device)
            p = mp.Process(target=train, args=(device, head_device, trainable_gan.gan, inputs, loaded_event, report_weights_queue, set_weights_queue, report_weights_event, save_event, save_complete_event, self.trainable_gan.save_file))
            p.start()
            self.processes.append(p)
            self.report_weights_event.append(report_weights_event)
            self.report_weights_queue.append(report_weights_queue)
            self.set_weights_queue.append(set_weights_queue)
            loaded_event.wait()
            save_event = None

    def save(self):
        self.save_event.set()
        self.save_complete_event.wait()

    def step(self):
        time.sleep(2.0)
        selected = self.sync % len(self.processes)
        print("Syncing", selected)
        self.report_weights_event[selected].set()
        params = self.report_weights_queue[selected].get()
        for p1, p2 in zip(self.trainable_gan.parameters(), params):
            with torch.no_grad():
                if(p2.device != p1.device):
                    p2 = p2.to(p1.device)
                p1.set_((p1+p2)/2.0)
        del params
        if self.devices[selected] == self.head_device:
            self.set_weights_queue[selected].put(list([p.clone().detach() for p in self.trainable_gan.parameters()]))
        else:
            self.set_weights_queue[selected].put(list(self.trainable_gan.parameters()))

        self.sync += 1
