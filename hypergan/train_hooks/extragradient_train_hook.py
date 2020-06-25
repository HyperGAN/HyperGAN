import torch
from torch.autograd import grad as torch_grad
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class ExtragradientTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.relu = torch.nn.ReLU()

    def gradients(self, d_grads, g_grads):
        step_size = self.config.step_size or 1e-4
        gamma = self.config.gamma
        if gamma is None:
            gamma = 1.0
        rho = self.config.rho
        if rho is None:
            rho = 1.0

        d_grads_v1 = [g.clone().detach() for g in d_grads]
        g_grads_v1 = [g.clone().detach() for g in g_grads]

        self.step(d_grads_v1, g_grads_v1, step_size)
        d_grads_v2, g_grads_v2 = self.gan.trainer.calculate_gradients()
        self.step([-g for g in d_grads_v1], [-g for g in g_grads_v1], step_size)


        if self.config.formulation == 'v2-v1':
            d_grads = [gamma*_v1-rho*(_v2-_v1)/step_size for _v1, _v2 in zip(d_grads_v1, d_grads_v2)]
            g_grads = [gamma*_v1-rho*(_v2-_v1)/step_size for _v1, _v2 in zip(g_grads_v1, g_grads_v2)]
        elif self.config.formulation == 'agree':
            d_grads = [gamma*self.agree(_v1,_v2,gamma,rho) for _v1, _v2 in zip(d_grads_v1, d_grads_v2)]
            g_grads = [gamma*self.agree(_v1,_v2,gamma,rho) for _v1, _v2 in zip(g_grads_v1, g_grads_v2)]
        else:
            d_grads = [gamma*_v1+rho*_v2 for _v1, _v2 in zip(d_grads_v1, d_grads_v2)]
            g_grads = [gamma*_v1+rho*_v2 for _v1, _v2 in zip(g_grads_v1, g_grads_v2)]

        return [d_grads, g_grads]

    def agree(self, v1, v2, gamma, rho):
        result = (gamma*v1+rho*v2) * self.relu(torch.sign(v1*v2))
        return result

    def forward(self):
        return [None, None]

    def step(self, d_grads, g_grads, step_size):
        d_params = self.gan.d_parameters()
        g_params = self.gan.g_parameters()
        for _g, _p in zip(d_grads, d_params):
            _p.data -= step_size * _g

        for _g, _p in zip(g_grads, g_params):
            _p.data -= step_size * _g

