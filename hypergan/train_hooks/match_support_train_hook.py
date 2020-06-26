from hypergan.viewer import GlobalViewer
import hyperchamber as hc
import numpy as np
import inspect
from hypergan.gan_component import ValidationException, GANComponent
from torch.autograd import grad as torch_grad
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

from hypergan.viewer import GlobalViewer

class MatchSupportTrainHook(BaseTrainHook):
    """ 
      Makes d_fake and d_real balance by running an optimizer.
    """
    def __init__(self, gan=None, config=None, trainer=None):
        super().__init__(config=config, gan=gan, trainer=trainer)

    def before_step(self, step, feed_dict, depth=0):
        defn = self.config.optimizer.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]
        self.optimizer = klass(self.gan.generator.parameters(), **defn)

        for i in range(self.config.steps or 1):
            self.optimizer.zero_grad()
            fake = self.gan.discriminator(self.gan.generator(self.gan.latent.instance)).mean()
            real = self.gan.discriminator(self.gan.inputs.sample).mean()
            loss = self.gan.loss.forward_adversarial_norm(real, fake)
            if loss == 0.0:
                if self.config.verbose:
                    print("[match support] No loss")
                break
            move = torch_grad(outputs=loss, inputs=self.gan.g_parameters(), retain_graph=True, create_graph=True)

            if self.config.regularize:
                move = torch_grad(outputs=(loss+sum([m.abs().sum() for m in move])), inputs=self.gan.g_parameters(), retain_graph=True, create_graph=True)
            for p, g in zip(self.gan.g_parameters(), move):
                if p._grad is not None:
                    p._grad.copy_(g)
                else:
                    pass
                    #print("Missing g")
            self.optimizer.step()
            if self.config.verbose:
                print("[match support]", i, "loss", loss.item())
            if self.config.loss_threshold and loss < self.config.loss_threshold:
                if self.config.info:
                    print("[match support] loss_threshold steps", i, "loss", loss.item())
                break
        if self.config.info and i == self.config.steps-1:
            print("[match support] loss_threshold steps", i, "loss", loss.item())
