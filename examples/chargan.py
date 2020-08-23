import argparse
import os
import torch
import string
import uuid
import hypergan as hg
import hyperchamber as hc
from hypergan.generators import *
import torch.utils.data as data
from examples.common import *
import numpy as np

class TextData(data.Dataset):
    def __init__(self, path, max_length, device, mode="LINE"):
        if mode == "LINE":
            with open(path, errors='replace') as f:
                self.lines_raw = f.readlines()

            self.lines = [self.pad_or_truncate(line) for line in self.lines_raw]
            self.lines = [line for line in self.lines if line is not None]
        else:
            self.size = os.path.getsize(path)
            self.file = None
            self.path = path

        self.mode = mode
        self.device = device
        self.lookup = None
        self.max_length = max_length
        self.vocab = self.get_vocabulary()

    def encode_line(self, line):
        return torch.cat([self.get_encoded_value(c) for i, c in enumerate(line)])

    def get_character(self, data):
        enc = self.get_lookup_table()[1]
        if data not in enc:
            return "�"
        return enc[data]

    def get_encoded_value(self, data):
        enc = self.get_lookup_table()[0]
        if data not in enc:
            return enc[" "]
        return enc[data]

    def get_lookup_table(self):
        if self.lookup is None:
            vocabulary = self.get_vocabulary()
            values = np.arange(len(vocabulary))
            lookup = {}

            for i, key in enumerate(vocabulary):
                lookup[key]=values[i]

            #reverse the hash
            rlookup = {i[1]:i[0] for i in lookup.items()}
            for key, value in lookup.items():
                lookup[key] = torch.tensor([ lookup[key] / float(len(vocabulary)) * 2 - 1], dtype=torch.float32)
            self.lookup = [lookup, rlookup]
        return self.lookup

    def get_vocabulary(self):
        vocab = list(" ~()\"'&+#@/789zyxwvutsrqponmlkjihgfedcbaABCDEFGHIJKLMNOPQRSTUVWXYZ0123456:-,;!?.\n")
        return vocab

    def __getitem__(self, index):
        if self.mode == "LINE":
            return [self.encode_line(self.lines[index]).view(1, -1)]
        else:
            if self.file is None:
                self.file = open(self.path, 'r', errors='replace')
            self.file.seek(index)
            data = self.file.read(self.max_length)
            encoded = self.encode_line(data).view(1, -1)
            return [encoded]

    def __len__(self):
        if self.mode == "LINE":
            return len(self.lines)
        else:
            return self.size - self.max_length - 1

    def pad_or_truncate(self, line):
        line = line.rstrip()
        if len(line) == 0:
            return None
        return line.ljust(self.max_length, " ")[:self.max_length]

    def sample_output(self, val):
        vocabulary = self.get_vocabulary()
        val = (np.reshape(val, [-1]) + 1) / 2.0
        x = val[0]
        val *= len(vocabulary)
        val = np.round(val)
        val = np.minimum(val, len(vocabulary)-1)

        ox_val = [self.get_character(obj) for obj in list(val)]
        string = "".join(ox_val)
        return string


class TextInput:
    def __init__(self, config, batch_size, filename, length, one_hot=False, device=0, mode="LINE"):
        self.textdata = TextData(filename, length, device=device, mode=mode)
        self.dataloader = data.DataLoader(self.textdata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.mode = mode
        self.dataset = None
        self._batch_size = batch_size
        self.length = length
        self.filename = filename
        self.one_hot = one_hot
        self.device = device
        self.config = config

    def text_plot(self, size, filename, data, x):
        bs = x.shape[0]
        data = np.reshape(data, [bs, -1])
        x = np.reshape(x, [bs, -1])
        plt.clf()
        plt.figure(figsize=(2,2))
        data = np.squeeze(data)
        plt.plot(x)
        plt.plot(data)
        plt.xlim([0, size])
        plt.ylim([-2, 2.])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig(filename)

    def to(self, device):
        return TextInput(self.config, self._batch_size, self.filename, self.length, self.one_hot, device=device, mode=self.mode)

    def next(self, index=0):
        if self.dataset is None:
            self.dataset = iter(self.dataloader)
        try:
            self.sample = self.dataset.next()[0].to(self.device)
            return self.sample
        except StopIteration:
            self.dataset = iter(self.dataloader)
            return self.next(index)

    def batch_size(self):
        return self._batch_size

    def channels(self):
        return 1#self.length

    def width(self):
        return 1

    def height(self):
        return self.length


if __name__ == '__main__':
    arg_parser = ArgumentParser("Learn from a text file", require_directory=False)
    arg_parser.parser.add_argument('--one_hot', action='store_true', help='Use character one-hot encodings.')
    arg_parser.parser.add_argument('--filename', type=str, default='chargan.txt', help='Input dataset');
    arg_parser.parser.add_argument('--length', type=int, default=256, help='Length of string per line');
    args = arg_parser.parse_args()


    config = lookup_config(args)

    config_name = args.config
    save_file = "saves/"+config_name+"/model.ckpt"


    inputs = TextInput(config, args.batch_size, args.filename, args.length, one_hot=args.one_hot, mode="CHAR")

    def parse_size(size):
        width = int(size.split("x")[0])
        height = int(size.split("x")[1])
        channels = int(size.split("x")[2])
        return [width, height, channels]

    def lookup_config(args):
        if args.action != 'search':
            return hg.configuration.Configuration.load(args.config+".json")

    def random_config_from_list(config_list_file):
        """ Chooses a random configuration from a list of configs (separated by newline) """
        lines = tuple(open(config_list_file, 'r'))
        config_file = random.choice(lines).strip()
        print("[hypergan] config file chosen from list ", config_list_file, '  file:', config_file)
        return hg.configuration.Configuration.load(config_file+".json")



    def setup_gan(config, inputs, args):
        gan = hg.GAN(config, inputs=inputs)
        gan.load(save_file)

        return gan

    def sample(config, inputs, args):
        gan = setup_gan(config, inputs, args)

    def search(config, inputs, args):
        gan = setup_gan(config, inputs, args)

    def train(config, inputs, args):
        gan = setup_gan(config, inputs, args)
        trainable_gan = hg.TrainableGAN(gan, save_file = save_file, devices = args.devices, backend_name = args.backend)

        trainers = []

        x_0 = gan.inputs.next()
        z_0 = gan.latent.sample()

        ax_sum = 0
        ag_sum = 0
        diversity = 0.00001
        dlog = 0
        last_i = 0
        samples = 0
        steps = 0

        while(True):
            steps +=1
            if steps > args.steps and args.steps != -1:
                break
            trainable_gan.step()

            if args.action == 'train' and steps % args.save_every == 0 and steps > 0:
                print("saving " + save_file)
                trainable_gan.save()

            if steps % args.sample_every == 0:
                g = gan.generator.forward(gan.latent.sample()).cpu().detach().numpy()
                print("SAHPE", g.shape)
                x_val = gan.inputs.next().cpu().detach().numpy()
                bs = np.shape(x_val)[0]
                samples+=1
                print("X:")
                print(inputs.textdata.sample_output(x_val[0]))
                print("G:")
                print(inputs.textdata.sample_output(g[0]))

        if args.config is None:
            with open("sequence-results-10k.csv", "a") as myfile:
                myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")

    if args.action == 'train':
        metrics = train(config, inputs, args)
        print("Resulting metrics:", metrics)
    elif args.action == 'sample':
        sample(config, inputs, args)
    elif args.action == 'search':
        search(config, inputs, args)
    else:
        print("Unknown action: "+args.action)

