class TrainHookCollection:
    def __init__(self, gan):
        self.gan = gan

    def augment_latent(self, latent):
        for hook in self.gan.hooks:
            latent = hook.augment_latent(latent)
        return latent

    def augment_x(self, x):
        for hook in self.gan.hooks:
            x = hook.augment_x(x)
        return x

    def augment_g(self, g):
        for hook in self.gan.hooks:
            g = hook.augment_g(g)
        return g
