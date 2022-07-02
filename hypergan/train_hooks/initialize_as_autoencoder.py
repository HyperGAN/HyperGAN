import hyperchamber as hc
import numpy as np
import inspect
from hypergan.gan_component import ValidationException, GANComponent
from torch.autograd import grad as torch_grad
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook


class InitializeAsAutoencoder(BaseTrainHook):
    """
        G becomes an autoencoder on step zero.
    """
    def __init__(self, gan=None, config=None, trainer=None):
        super().__init__(config=config, gan=gan, trainer=trainer)

    def before_step(self, step, feed_dict, depth=0):
        if self.gan.steps != 1:
            return

        defn = self.config.encoder.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        encode = klass(self.gan, defn).cuda()
        defn = self.config.optimizer.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]
        self.optimizer = klass(list(encode.parameters()) + list(self.gan.generator.parameters()), **defn)

        for i in range(self.config.steps or 1000):
            self.optimizer.zero_grad()
            inp = self.gan.inputs.next()
            e = encode(inp)
            fake = self.gan.generator(e)

            loss = ((inp - fake)**2).mean()
            loss.backward()
            for p in (list(self.gan.g_parameters())+list(encode.parameters())):
                p.requires_grad = True
            #move = torch_grad(outputs=loss, inputs=list(self.gan.g_parameters())+list(encode.parameters()), retain_graph=True, create_graph=True)
            #print(list(self.gan.g_parameters())+list(encode.parameters()))
            self.optimizer.step()
            if self.config.verbose:
                print("[autoencode]", i, "loss", loss.item())
            if self.config.info and (i % 100) == 0:
                print("[autoencode]", i, "loss", loss.item())
