import torch
from hypergan.gan_component import ValidationException, GANComponent


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization for Efficiently Improving Generalization
    From https://github.com/davda54/sam
    """
    def __init__(self, params, rho=0.05, delegate={}, **kw_args):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        passed_params = list(params).copy()
        base_optimizer = self.create_optimizer(delegate, passed_params, **kw_args)

        defaults = dict(rho=rho, **kw_args)
        super(SAM, self).__init__(passed_params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    def create_component(self, name, *args, **kw_args):
        gan_component = self.initialize_component(name, *args, **kw_args)
        return gan_component

    def create_optimizer(self, defn, params):
        defn = defn.copy()
        klass = GANComponent.lookup_function(None, defn['class'])
        del defn["class"]
        return klass(params, **defn)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
