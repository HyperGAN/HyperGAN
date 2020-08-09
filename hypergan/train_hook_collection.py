class TrainHookCollection:
    def __init__(self, gan):
        self.gan = gan

    def augment_latent(self, latent):
        for train_hook in self.gan.trainer.train_hooks:
            latent = train_hook.augment_latent(latent)
        return latent

    def augment_x(self, x):
        for train_hook in self.gan.trainer.train_hooks:
            x = train_hook.augment_x(x)
        return x

    def augment_g(self, g):
        for train_hook in self.gan.trainer.train_hooks:
            g = train_hook.augment_g(g)
        return g
