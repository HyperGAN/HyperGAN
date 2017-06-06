from hypergan.gan_component import GANComponent

class BaseLoss(GANComponent):
    def split_batch(net):
        ops = self.ops
        s = ops.shape(net)
        net = ops.reshape(net, [s[0], -1])
        d_real = ops.slice(net, [0,0], [s[0]//2,-1])
        d_fake = ops.slice(net, [s[0]//2,0], [s[0]//2,-1])
        return [d_real, d_fake]

    def create(self):
        self._create()
